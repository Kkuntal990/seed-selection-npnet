"""ResidualEditor: lightweight text-conditioned residual noise editor (~50K params).

Applies a small, gated residual correction to noise latents conditioned on text.
Initialized to near-zero output for stable training on top of a frozen NPNet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.normalization import AdaGroupNorm

logger = logging.getLogger(__name__)


class ResidualEditor(nn.Module):
    """Apply a small text-conditioned residual edit to noise.

    Architecture:
        3x Conv2d blocks (4->16->16->4) with AdaGroupNorm + SiLU.
        Gated output: ``gate * tanh(conv_out)`` where gate starts near zero.

    Args:
        latent_channels: Number of latent channels (4 for SDXL).
        hidden_channels: Intermediate channel count (default 16).
        text_embed_dim: Dimension of text embeddings (2048 for SDXL).
        text_seq_len: Sequence length of text embeddings (77 for SDXL).

    Forward:
        noise: ``(B, C, H, W)`` base noise (NPNet output).
        prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.

    Returns:
        ``(B, C, H, W)`` residual delta (near-zero at initialization).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: int = 16,
        text_embed_dim: int = 2048,
        text_seq_len: int = 77,
    ) -> None:
        super().__init__()
        self.text_seq_len = text_seq_len
        text_flat_dim = text_embed_dim  # mean-pooled

        # Block 1: in -> hidden
        self.conv1 = nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = AdaGroupNorm(text_flat_dim, hidden_channels, 4, eps=1e-6)
        self.act1 = nn.SiLU()

        # Block 2: hidden -> hidden
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(text_flat_dim, hidden_channels, 4, eps=1e-6)
        self.act2 = nn.SiLU()

        # Block 3: hidden -> out
        self.conv3 = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

        # Learnable gate initialized to -5 => sigmoid(-5) ~ 0.007 => near-zero output at init
        self._gate_logit = nn.Parameter(torch.tensor(-5.0))

    def forward(self, noise: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Compute text-conditioned residual delta.

        Args:
            noise: ``(B, C, H, W)`` base noise.
            prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.

        Returns:
            ``(B, C, H, W)`` gated residual delta.
        """
        # Mean-pool text embeddings for conditioning
        text_cond = prompt_embeds.float().mean(dim=1)  # (B, text_embed_dim)

        x = self.conv1(noise.float())
        x = self.norm1(x, text_cond)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x, text_cond)
        x = self.act2(x)

        x = self.conv3(x)

        gate = torch.sigmoid(self._gate_logit)
        return gate * torch.tanh(x)

    def save_checkpoint(self, path: Path) -> None:
        """Save editor checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"editor": self.state_dict()}, path)
        logger.info("Editor checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load editor checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(ckpt["editor"])
        logger.info("Editor checkpoint loaded: %s", path)
