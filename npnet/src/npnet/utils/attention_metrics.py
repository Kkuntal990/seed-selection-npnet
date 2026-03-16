"""Attention-based guidance losses for hybrid editor training.

Approach 1 (MVP): Denoising consistency loss — push predicted noise to produce
the same UNet denoising prediction as the target noise at a random timestep.

Approach 2 (Enhancement): Cross-attention map losses via custom AttnProcessor.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approach 1: Denoising consistency loss
# ---------------------------------------------------------------------------


def denoising_consistency_loss(
    frozen_unet: Any,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    timestep: torch.Tensor,
    added_cond_kwargs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute denoising consistency between predicted and target noise.

    Gradients flow through ``predicted_noise`` (from the editor) but not
    through the UNet parameters (frozen) or the target path.

    Args:
        frozen_unet: Frozen SDXL UNet (no grad on parameters).
        predicted_noise: ``(B, 4, H, W)`` editor output (requires grad).
        target_noise: ``(B, 4, H, W)`` target golden noise (no grad).
        prompt_embeds: ``(B, 77, 2048)`` text embeddings.
        pooled_prompt_embeds: ``(B, 1280)`` pooled text embeddings.
        timestep: ``(B,)`` random diffusion timestep.
        added_cond_kwargs: Additional conditioning for SDXL UNet.

    Returns:
        Scalar MSE loss between UNet predictions on predicted vs target noise.
    """
    if added_cond_kwargs is None:
        # SDXL requires time_ids; use default 1024x1024 crop
        batch_size = predicted_noise.shape[0]
        device = predicted_noise.device
        time_ids = torch.tensor(
            [[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]],
            device=device,
        ).expand(batch_size, -1)
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }

    # Forward through frozen UNet with predicted noise (grads flow to input)
    eps_pred = frozen_unet(
        predicted_noise,
        timestep,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # Forward through frozen UNet with target noise (no grads)
    with torch.no_grad():
        eps_target = frozen_unet(
            target_noise,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        ).sample

    return F.mse_loss(eps_pred, eps_target)


def sample_random_timestep(
    batch_size: int,
    num_train_timesteps: int = 1000,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample a random diffusion timestep for each batch element."""
    return torch.randint(0, num_train_timesteps, (batch_size,), device=device)


# ---------------------------------------------------------------------------
# Approach 2: Cross-attention map extraction (enhancement)
# ---------------------------------------------------------------------------


class AttnMapProcessor:
    """Custom attention processor that stores cross-attention maps.

    Install on specific UNet layers to capture where each text token
    attends in the spatial domain.
    """

    def __init__(self) -> None:
        self.attn_maps: list[torch.Tensor] = []

    def clear(self) -> None:
        self.attn_maps.clear()

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process attention and store cross-attention maps."""
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(kv_input)
        value = attn.to_v(kv_input)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_weights = torch.bmm(query, key.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = attn_weights.softmax(dim=-1)

        # Store cross-attention maps only
        if is_cross:
            self.attn_maps.append(attn_weights.detach())

        hidden_states = torch.bmm(attn_weights, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def cross_attention_response_loss(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    """Compute response loss: maximize peak spatial activation per subject token.

    Encourages each text token to have a focused spatial attention peak.
    """
    if not attn_maps:
        return torch.tensor(0.0)

    losses = []
    for attn_map in attn_maps:
        # attn_map: (B*heads, spatial, text_tokens)
        # Max spatial activation per text token
        max_response = attn_map.max(dim=1).values  # (B*heads, text_tokens)
        # Maximize (minimize negative)
        losses.append(-max_response.mean())

    return torch.stack(losses).mean()
