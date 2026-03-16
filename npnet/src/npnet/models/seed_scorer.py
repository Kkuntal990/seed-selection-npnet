"""SeedScorer: lightweight noise-prompt compatibility scorer (~200K params).

Predicts how well a given noise latent will work with a given text prompt.
Trained on VLM ranking data (pairwise or pointwise).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SeedScorer(nn.Module):
    """Score noise-prompt compatibility.

    Args:
        latent_channels: Number of latent channels (4 for SDXL).
        text_embed_dim: Dimension of text embeddings (2048 for SDXL).
        text_seq_len: Sequence length of text embeddings (77 for SDXL).
        hidden_dim: Hidden dimension for encoder heads (default 64).

    Forward:
        noise: ``(B, C, H, W)`` noise latent.
        prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.

    Returns:
        ``(B, 1)`` compatibility score (unbounded, higher = better).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        text_embed_dim: int = 2048,
        text_seq_len: int = 77,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.text_seq_len = text_seq_len

        # Noise encoder: spatial conv -> pool -> vector
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Text encoder: mean-pool across seq_len -> project
        self.text_encoder = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.GELU(),
        )

        # Prediction head: concat noise + text features -> score
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, noise: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Score noise-prompt compatibility.

        Args:
            noise: ``(B, C, H, W)`` noise latent.
            prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.

        Returns:
            ``(B, 1)`` compatibility score.
        """
        noise_feat = self.noise_encoder(noise.float())  # (B, hidden_dim)
        text_feat = self.text_encoder(prompt_embeds.float().mean(dim=1))  # (B, hidden_dim)
        combined = torch.cat([noise_feat, text_feat], dim=1)  # (B, hidden_dim*2)
        return self.head(combined)  # (B, 1)

    def save_checkpoint(self, path: Path) -> None:
        """Save scorer checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"scorer": self.state_dict()}, path)
        logger.info("Scorer checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load scorer checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(ckpt["scorer"])
        logger.info("Scorer checkpoint loaded: %s", path)
