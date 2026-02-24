"""SVD-based noise processor for NPNet.

Adapted from Golden-Noise-for-Diffusion-Models/training/model/SVDNoiseUnet.py
Uses the ``_wo_attention`` variant (recommended baseline).
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn


class SVDNoiseUnet(nn.Module):
    """Process noise latents via SVD decomposition and per-component MLPs.

    Rearranges ``(B, C, H, W)`` to ``(B, M, N)`` matrix, decomposes via SVD,
    processes ``U``, ``s``, ``V`` with separate MLPs, then reconstructs.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        resolution: int = 128,
    ) -> None:
        super().__init__()
        _in = int(resolution * in_channels // 2)
        _out = int(resolution * out_channels // 2)

        self.mlp1 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, _out),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(_in, _out),
        )
        self.mlp4 = nn.Sequential(
            nn.Linear(_out, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, _out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b (a c) h w -> b (a h) (c w)", a=2, c=2)
        U, s, V = torch.linalg.svd(x)
        U_T = U.permute(0, 2, 1)

        out = self.mlp1(U_T) + self.mlp2(V) + self.mlp3(s).unsqueeze(1)
        out = self.mlp4(out).mean(1) + s

        pred = U @ torch.diag_embed(out) @ V
        result: torch.Tensor = einops.rearrange(pred, "b (a h) (c w) -> b (a c) h w", a=2, c=2)
        return result
