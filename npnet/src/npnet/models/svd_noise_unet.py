"""SVD-based noise processor for NPNet.

Adapted from Golden-Noise-for-Diffusion-Models/inference/model/SVDNoiseUnet.py
Matches the original architecture with self-attention and batch norm.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Final

try:
    from timm.layers import use_fused_attn
except ImportError:
    # timm < 1.0 compatibility
    def use_fused_attn() -> bool:  # type: ignore[misc]
        return hasattr(F, "scaled_dot_product_attention")


class Attention(nn.Module):
    """Multi-head self-attention (from original SVDNoiseUnet)."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVDNoiseUnet(nn.Module):
    """Process noise latents via SVD decomposition, self-attention, and per-component MLPs.

    Rearranges ``(B, C, H, W)`` to ``(B, M, N)`` matrix, decomposes via SVD,
    processes ``U``, ``s``, ``V`` with separate MLPs + self-attention, then reconstructs.
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

        self.attention = Attention(_out)
        self.bn = nn.BatchNorm2d(_out)

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
        out = self.attention(out).mean(1)
        out = self.mlp4(out) + s

        pred = U @ torch.diag_embed(out) @ V
        result: torch.Tensor = einops.rearrange(pred, "b (a h) (c w) -> b (a c) h w", a=2, c=2)
        return result
