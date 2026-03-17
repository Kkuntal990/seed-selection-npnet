"""HybridNPNet: NPNet + SeedScorer + ResidualEditor combined module.

Hybrid formula:
    z_pred = NPNet(z, c) + editor(NPNet(z, c), c)

In delta inference mode:
    z_pred = z + NPNet(z, c) + editor(z + NPNet(z, c), c)

NPNet stays frozen. Editor adds a small residual on top.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from npnet.models.npnet import NPNet
from npnet.models.residual_editor import ResidualEditor
from npnet.models.seed_scorer import SeedScorer

logger = logging.getLogger(__name__)


class HybridNPNet(nn.Module):
    """Combined NPNet + scorer + editor for hybrid golden noise generation.

    Args:
        latent_channels: Number of latent channels (4 for SDXL).
        latent_resolution: Spatial resolution of latent (128 for 1024px SDXL).
        text_embed_dim: Dimension of text embeddings (2048 for SDXL).
        text_seq_len: Sequence length of text embeddings (77 for SDXL).
        inference_mode: ``"replacement"`` uses NPNet output directly,
            ``"delta"`` applies ``z + NPNet(z, c)``.
        scorer_hidden_dim: Hidden dimension for SeedScorer.
        editor_hidden_channels: Hidden channel count for ResidualEditor.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        latent_resolution: int = 128,
        text_embed_dim: int = 2048,
        text_seq_len: int = 77,
        inference_mode: str = "replacement",
        scorer_hidden_dim: int = 64,
        editor_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        self.inference_mode = inference_mode

        self.npnet = NPNet(
            latent_channels=latent_channels,
            latent_resolution=latent_resolution,
            text_embed_dim=text_embed_dim,
            text_seq_len=text_seq_len,
        )
        self.scorer = SeedScorer(
            latent_channels=latent_channels,
            text_embed_dim=text_embed_dim,
            text_seq_len=text_seq_len,
            hidden_dim=scorer_hidden_dim,
        )
        self.editor = ResidualEditor(
            latent_channels=latent_channels,
            hidden_channels=editor_hidden_channels,
            text_embed_dim=text_embed_dim,
            text_seq_len=text_seq_len,
        )

    def forward(
        self,
        initial_noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Transform initial noise into golden noise via NPNet + editor.

        Args:
            initial_noise: ``(B, C, H, W)`` random noise latent.
            prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.
            return_components: If True, return a dict with all intermediate values.

        Returns:
            Golden noise ``(B, C, H, W)`` or dict of components.
        """
        npnet_out = self.npnet(initial_noise, prompt_embeds)
        base = initial_noise.float() + npnet_out if self.inference_mode == "delta" else npnet_out

        delta = self.editor(base, prompt_embeds)
        golden = base + delta

        if return_components:
            # Score the editor OUTPUT so gradients flow back through the editor
            score = self.scorer(golden, prompt_embeds)
            return {
                "npnet_output": npnet_out,
                "editor_delta": delta,
                "golden_noise": golden,
                "score": score,
            }

        return golden

    def score_seeds(self, noise_batch: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Score a batch of seed noises for reranking.

        Args:
            noise_batch: ``(B, C, H, W)`` noise latents.
            prompt_embeds: ``(B, seq_len, embed_dim)`` text embeddings.

        Returns:
            ``(B, 1)`` scores.
        """
        return self.scorer(noise_batch, prompt_embeds)

    def save_checkpoint(self, path: Path) -> None:
        """Save hybrid checkpoint preserving NPNet format internally."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # Preserve original NPNet checkpoint format
                "unet_svd": self.npnet.unet_svd.state_dict(),
                "unet_embedding": self.npnet.unet_embedding.state_dict(),
                "embeeding": self.npnet.text_embedding.state_dict(),  # preserve original typo
                "alpha": self.npnet._alpha,
                "beta": self.npnet._beta,
                # Hybrid additions
                "scorer": self.scorer.state_dict(),
                "editor": self.editor.state_dict(),
                "inference_mode": self.inference_mode,
            },
            path,
        )
        logger.info("Hybrid checkpoint saved: %s", path)

    def load_checkpoint(
        self,
        path: Path,
        load_npnet: bool = True,
        load_scorer: bool = True,
        load_editor: bool = True,
    ) -> None:
        """Load hybrid checkpoint with selective component loading.

        Args:
            path: Checkpoint file path.
            load_npnet: Whether to load NPNet weights.
            load_scorer: Whether to load scorer weights.
            load_editor: Whether to load editor weights.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if load_npnet and "unet_svd" in ckpt:
            self.npnet.load_checkpoint(path)

        if load_scorer and "scorer" in ckpt:
            self.scorer.load_state_dict(ckpt["scorer"])
            logger.info("Scorer weights loaded from %s", path)

        if load_editor and "editor" in ckpt:
            self.editor.load_state_dict(ckpt["editor"])
            logger.info("Editor weights loaded from %s", path)

        if "inference_mode" in ckpt:
            self.inference_mode = ckpt["inference_mode"]

    def load_component_checkpoints(
        self,
        npnet_path: Path | None = None,
        scorer_path: Path | None = None,
        editor_path: Path | None = None,
    ) -> None:
        """Load individual component checkpoints from separate files."""
        if npnet_path is not None:
            self.npnet.load_checkpoint(npnet_path)
        if scorer_path is not None:
            self.scorer.load_checkpoint(scorer_path)
        if editor_path is not None:
            self.editor.load_checkpoint(editor_path)
