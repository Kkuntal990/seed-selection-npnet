"""Swin Transformer-based noise processor for NPNet.

Adapted from Golden-Noise-for-Diffusion-Models/training/model/NoiseTransformer.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm import create_model
from torch.nn import functional as F


class NoiseTransformer(nn.Module):
    """Process noise latents through a pretrained Swin Transformer.

    Pipeline: Conv2d(C→3) → upsample(224) → Swin features → downsample(res) → Conv2d(7→C)
    """

    def __init__(self, resolution: int = 128, channels: int = 4) -> None:
        super().__init__()
        self.resolution = resolution
        self.downconv = nn.Conv2d(channels, 3, (1, 1), (1, 1), (0, 0))
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True)
        # Swin feature dim: 768 for swin_tiny
        swin_feat_dim = self.swin.num_features
        self.upconv = nn.Conv2d(swin_feat_dim, channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.downconv(x)
        out = F.interpolate(out, [224, 224])
        out = self.swin.forward_features(out)  # type: ignore[operator]
        # timm >= 1.0 returns (B, H, W, C) in NHWC format — permute to NCHW
        if out.ndim == 4 and out.shape[-1] != out.shape[-2]:
            out = out.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        elif out.ndim == 3:
            # timm 0.6.x returns (B, H*W, C) — reshape to (B, C, H, W)
            b, hw, c = out.shape
            h = w = int(hw**0.5)
            out = out.permute(0, 2, 1).reshape(b, c, h, w)
        out = F.interpolate(out, [self.resolution, self.resolution])
        result: torch.Tensor = self.upconv(out)
        return result
