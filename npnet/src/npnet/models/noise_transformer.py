"""Swin Transformer-based noise processor for NPNet.

Adapted from Golden-Noise-for-Diffusion-Models/inference/model/NoiseTransformer.py
Matches the original architecture exactly.
"""

from __future__ import annotations

import torch.nn as nn
from timm import create_model
from torch.nn import functional as F


class NoiseTransformer(nn.Module):
    """Process noise latents through a pretrained Swin Transformer.

    Pipeline: Conv2d(4→3) → upsample(224) → Swin features → downsample(res) → Conv2d(7→4)

    Note: The Swin outputs (B, 7, 7, 768) in NHWC format. F.interpolate treats
    this as (B, C=7, H=7, W=768) and resizes to (B, 7, res, res), giving
    7 "channels" which are fed to upconv(7→4).
    """

    def __init__(self, resolution: int = 128, channels: int = 4) -> None:
        super().__init__()
        self.resolution = resolution
        self.upconv = nn.Conv2d(7, channels, (1, 1), (1, 1), (0, 0))
        self.downconv = nn.Conv2d(channels, 3, (1, 1), (1, 1), (0, 0))
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.downconv(x)
        out = F.interpolate(out, [224, 224])
        out = self.swin.forward_features(out)  # type: ignore[operator]
        # timm 0.6.x: (B, 49, 768) — reshape to (B, 7, 7, 768)
        if out.ndim == 3:
            b, hw, c = out.shape
            h = w = int(hw**0.5)
            out = out.reshape(b, h, w, c)
        # Now (B, 7, 7, 768) in NHWC — F.interpolate treats as (B, C=7, H=7, W=768)
        # resizes to (B, 7, res, res), giving 7 "channels" for upconv(7→4)
        out = F.interpolate(out, [self.resolution, self.resolution])
        result: "torch.Tensor" = self.upconv(out)
        return result
