"""Combined NPNet module: SVD + Swin Transformer + text embedding.

Adapted from Golden-Noise-for-Diffusion-Models/inference/npnet_pipeline.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.normalization import AdaGroupNorm

from npnet.models.noise_transformer import NoiseTransformer
from npnet.models.svd_noise_unet import SVDNoiseUnet

logger = logging.getLogger(__name__)


class NPNet(nn.Module):
    """Noise Prompt Network: transforms random noise into golden noise.

    Combines three sub-networks::

        golden_noise = unet_svd(noise)
                     + (2·sigmoid(α) - 1) · text_emb
                     + β · unet_embedding(noise + text_emb)
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_resolution: int = 128,
        text_embed_dim: int = 2048,
        text_seq_len: int = 77,
    ) -> None:
        super().__init__()
        self.unet_svd = SVDNoiseUnet(
            in_channels=latent_channels,
            out_channels=latent_channels,
            resolution=latent_resolution,
        )
        self.unet_embedding = NoiseTransformer(
            resolution=latent_resolution,
            channels=latent_channels,
        )
        self.text_embedding = AdaGroupNorm(
            text_embed_dim * text_seq_len,
            latent_channels,
            1,
            eps=1e-6,
        )
        self._alpha = nn.Parameter(torch.zeros(1))
        self._beta = nn.Parameter(torch.zeros(1))

    def forward(self, initial_noise: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Transform initial noise into golden noise.

        Args:
            initial_noise: ``(B, C, H, W)`` random noise latent.
            prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings from SDXL.

        Returns:
            Golden noise tensor with same shape as ``initial_noise``.
        """
        prompt_flat = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
        text_emb = self.text_embedding(initial_noise.float(), prompt_flat)

        golden_noise = (
            self.unet_svd(initial_noise.float())
            + (2 * torch.sigmoid(self._alpha) - 1) * text_emb
            + self._beta * self.unet_embedding((initial_noise + text_emb).float())
        )
        return golden_noise

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint in format compatible with original Golden Noise repo."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "unet_svd": self.unet_svd.state_dict(),
                "unet_embedding": self.unet_embedding.state_dict(),
                "embeeding": self.text_embedding.state_dict(),  # original repo typo preserved
                "alpha": self._alpha,
                "beta": self._beta,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint from original Golden Noise repo format."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.unet_svd.load_state_dict(ckpt["unet_svd"])
        self.unet_embedding.load_state_dict(ckpt["unet_embedding"])
        self.text_embedding.load_state_dict(ckpt["embeeding"])
        self._alpha.data = ckpt["alpha"].data if isinstance(ckpt["alpha"], torch.Tensor) else torch.tensor([ckpt["alpha"]])
        self._beta.data = ckpt["beta"].data if isinstance(ckpt["beta"], torch.Tensor) else torch.tensor([ckpt["beta"]])
        logger.info("Checkpoint loaded: %s", path)
