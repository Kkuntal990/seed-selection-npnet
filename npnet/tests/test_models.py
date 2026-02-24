"""Tests for NPNet model architecture: shapes, gradients, checkpoint round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from npnet.models.npnet import NPNet
from npnet.models.svd_noise_unet import SVDNoiseUnet

# SDXL latent dimensions
B, C, H, W = 2, 4, 128, 128
TEXT_DIM, TEXT_SEQ = 2048, 77


class TestSVDNoiseUnet:
    def test_output_shape(self) -> None:
        model = SVDNoiseUnet(in_channels=C, out_channels=C, resolution=H)
        x = torch.randn(B, C, H, W)
        out = model(x)
        assert out.shape == (B, C, H, W)

    def test_gradients_flow(self) -> None:
        model = SVDNoiseUnet(in_channels=C, out_channels=C, resolution=H)
        x = torch.randn(B, C, H, W, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (B, C, H, W)


class TestNoiseTransformer:
    """Swin Transformer tests are skipped by default (downloads pretrained weights)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="NoiseTransformer needs GPU + pretrained weight download",
    )
    def test_output_shape(self) -> None:
        from npnet.models.noise_transformer import NoiseTransformer

        model = NoiseTransformer(resolution=H, channels=C).cuda()
        x = torch.randn(B, C, H, W).cuda()
        out = model(x)
        assert out.shape == (B, C, H, W)


class TestNPNet:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="NPNet includes Swin Transformer (needs GPU + pretrained weights)",
    )
    def test_output_shape(self) -> None:
        model = NPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        noise = torch.randn(B, C, H, W).cuda()
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM).cuda()
        out = model(noise, prompt_embeds)
        assert out.shape == (B, C, H, W)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="NPNet includes Swin Transformer (needs GPU + pretrained weights)",
    )
    def test_gradients_flow(self) -> None:
        model = NPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        noise = torch.randn(B, C, H, W, requires_grad=True).cuda()
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM).cuda()
        out = model(noise, prompt_embeds)
        out.sum().backward()
        assert noise.grad is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="NPNet includes Swin Transformer (needs GPU + pretrained weights)",
    )
    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        model_a = NPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()

        ckpt_path = tmp_path / "test_ckpt.pth"
        model_a.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        model_b = NPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        model_b.load_checkpoint(ckpt_path)

        # Verify weights match
        for (name_a, p_a), (name_b, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters(), strict=True
        ):
            assert name_a == name_b
            assert torch.equal(p_a, p_b), f"Mismatch in {name_a}"


class TestSVDNoiseUnetCPU:
    """CPU-only tests for SVDNoiseUnet (no pretrained download needed)."""

    def test_different_batch_sizes(self) -> None:
        model = SVDNoiseUnet(in_channels=C, out_channels=C, resolution=H)
        for batch in [1, 4]:
            x = torch.randn(batch, C, H, W)
            out = model(x)
            assert out.shape == (batch, C, H, W)

    def test_param_count(self) -> None:
        model = SVDNoiseUnet(in_channels=C, out_channels=C, resolution=H)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # Verify it's a reasonable size (not accidentally huge)
        assert n_params < 5_000_000, f"SVDNoiseUnet has {n_params} params, unexpectedly large"
