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
        self.upconv = nn.Conv2d(7, channels, (1, 1), (1, 1), (0, 0))
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.downconv(x)
        out = F.interpolate(out, [224, 224])
        out = self.swin.forward_features(out)
        out = F.interpolate(out, [self.resolution, self.resolution])
        out = self.upconv(out)
        return out
