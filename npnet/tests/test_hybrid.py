"""Tests for hybrid scorer, editor, and combined HybridNPNet."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from npnet.models.residual_editor import ResidualEditor
from npnet.models.seed_scorer import SeedScorer

# SDXL latent dimensions
B, C, H, W = 2, 4, 128, 128
TEXT_DIM, TEXT_SEQ = 2048, 77


# ---------------------------------------------------------------------------
# SeedScorer
# ---------------------------------------------------------------------------


class TestSeedScorer:
    def test_output_shape(self) -> None:
        scorer = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        noise = torch.randn(B, C, H, W)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)
        out = scorer(noise, prompt_embeds)
        assert out.shape == (B, 1)

    def test_gradients_flow(self) -> None:
        scorer = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        noise = torch.randn(B, C, H, W, requires_grad=True)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)
        out = scorer(noise, prompt_embeds)
        out.sum().backward()
        assert noise.grad is not None
        assert noise.grad.shape == (B, C, H, W)

    def test_param_count(self) -> None:
        scorer = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        n_params = sum(p.numel() for p in scorer.parameters())
        assert n_params > 0
        # Should be ~200K params
        assert n_params < 500_000, f"SeedScorer has {n_params} params, unexpectedly large"

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        scorer_a = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        ckpt_path = tmp_path / "scorer.pth"
        scorer_a.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        scorer_b = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        scorer_b.load_checkpoint(ckpt_path)

        for (na, pa), (nb, pb) in zip(
            scorer_a.named_parameters(), scorer_b.named_parameters(), strict=True
        ):
            assert na == nb
            assert torch.equal(pa, pb), f"Mismatch in {na}"

    def test_different_batch_sizes(self) -> None:
        scorer = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        for batch in [1, 4, 8]:
            noise = torch.randn(batch, C, H, W)
            prompt_embeds = torch.randn(batch, TEXT_SEQ, TEXT_DIM)
            out = scorer(noise, prompt_embeds)
            assert out.shape == (batch, 1)

    def test_pairwise_ranking_loss(self) -> None:
        """Verify scorer can be used with MarginRankingLoss."""
        scorer = SeedScorer(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        good_noise = torch.randn(B, C, H, W)
        bad_noise = torch.randn(B, C, H, W)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)

        good_score = scorer(good_noise, prompt_embeds)
        bad_score = scorer(bad_noise, prompt_embeds)
        target = torch.ones(B)

        loss = torch.nn.functional.margin_ranking_loss(
            good_score.squeeze(-1), bad_score.squeeze(-1), target, margin=0.5
        )
        loss.backward()
        # Verify gradients reach scorer parameters
        for p in scorer.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# ResidualEditor
# ---------------------------------------------------------------------------


class TestResidualEditor:
    def test_output_shape(self) -> None:
        editor = ResidualEditor(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        noise = torch.randn(B, C, H, W)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)
        out = editor(noise, prompt_embeds)
        assert out.shape == (B, C, H, W)

    def test_near_zero_init(self) -> None:
        """Editor output should be near-zero at initialization due to gate."""
        editor = ResidualEditor(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        noise = torch.randn(B, C, H, W)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)

        with torch.no_grad():
            delta = editor(noise, prompt_embeds)

        # Gate sigmoid(-5) ~ 0.007, so output should be very small
        assert delta.abs().max().item() < 0.1, (
            f"Editor output max={delta.abs().max().item()}, expected near-zero"
        )

    def test_gradients_flow(self) -> None:
        editor = ResidualEditor(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        noise = torch.randn(B, C, H, W, requires_grad=True)
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM)
        out = editor(noise, prompt_embeds)
        out.sum().backward()
        assert noise.grad is not None

    def test_param_count(self) -> None:
        editor = ResidualEditor(latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ)
        n_params = sum(p.numel() for p in editor.parameters())
        assert n_params > 0
        # Should be ~50K params
        assert n_params < 200_000, f"ResidualEditor has {n_params} params, unexpectedly large"

    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        editor_a = ResidualEditor(
            latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ
        )
        ckpt_path = tmp_path / "editor.pth"
        editor_a.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        editor_b = ResidualEditor(
            latent_channels=C, text_embed_dim=TEXT_DIM, text_seq_len=TEXT_SEQ
        )
        editor_b.load_checkpoint(ckpt_path)

        for (na, pa), (nb, pb) in zip(
            editor_a.named_parameters(), editor_b.named_parameters(), strict=True
        ):
            assert na == nb
            assert torch.equal(pa, pb), f"Mismatch in {na}"


# ---------------------------------------------------------------------------
# HybridNPNet
# ---------------------------------------------------------------------------


class TestHybridNPNet:
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="HybridNPNet includes NPNet (Swin Transformer needs GPU)",
    )
    def test_output_shape(self) -> None:
        from npnet.models.hybrid_npnet import HybridNPNet

        model = HybridNPNet(
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
        reason="HybridNPNet includes NPNet (Swin Transformer needs GPU)",
    )
    def test_return_components(self) -> None:
        from npnet.models.hybrid_npnet import HybridNPNet

        model = HybridNPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        noise = torch.randn(B, C, H, W).cuda()
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM).cuda()
        result = model(noise, prompt_embeds, return_components=True)
        assert isinstance(result, dict)
        assert "npnet_output" in result
        assert "editor_delta" in result
        assert "golden_noise" in result
        assert "score" in result
        assert result["golden_noise"].shape == (B, C, H, W)
        assert result["score"].shape == (B, 1)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="HybridNPNet includes NPNet (Swin Transformer needs GPU)",
    )
    def test_editor_only_grad_when_npnet_frozen(self) -> None:
        """When NPNet is frozen, only editor params should have gradients."""
        from npnet.models.hybrid_npnet import HybridNPNet

        model = HybridNPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        model.npnet.requires_grad_(False)
        model.scorer.requires_grad_(False)

        noise = torch.randn(B, C, H, W).cuda()
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM).cuda()
        out = model(noise, prompt_embeds)
        out.sum().backward()

        # Editor params should have gradients
        for name, p in model.editor.named_parameters():
            assert p.grad is not None, f"Editor param {name} has no gradient"

        # NPNet params should NOT have gradients
        for name, p in model.npnet.named_parameters():
            assert p.grad is None, f"NPNet param {name} should not have gradient"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="HybridNPNet includes NPNet (Swin Transformer needs GPU)",
    )
    def test_checkpoint_round_trip(self, tmp_path: Path) -> None:
        from npnet.models.hybrid_npnet import HybridNPNet

        model_a = HybridNPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()

        ckpt_path = tmp_path / "hybrid.pth"
        model_a.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        model_b = HybridNPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
        ).cuda()
        model_b.load_checkpoint(ckpt_path)

        for (na, pa), (nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters(), strict=True
        ):
            assert na == nb
            assert torch.equal(pa, pb), f"Mismatch in {na}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="HybridNPNet includes NPNet (Swin Transformer needs GPU)",
    )
    def test_delta_mode(self) -> None:
        from npnet.models.hybrid_npnet import HybridNPNet

        model = HybridNPNet(
            latent_channels=C,
            latent_resolution=H,
            text_embed_dim=TEXT_DIM,
            text_seq_len=TEXT_SEQ,
            inference_mode="delta",
        ).cuda()
        noise = torch.randn(B, C, H, W).cuda()
        prompt_embeds = torch.randn(B, TEXT_SEQ, TEXT_DIM).cuda()
        out = model(noise, prompt_embeds)
        assert out.shape == (B, C, H, W)


# ---------------------------------------------------------------------------
# Attention metrics
# ---------------------------------------------------------------------------


class TestAttentionMetrics:
    def test_sample_random_timestep(self) -> None:
        from npnet.utils.attention_metrics import sample_random_timestep

        t = sample_random_timestep(4, num_train_timesteps=1000)
        assert t.shape == (4,)
        assert (t >= 0).all()
        assert (t < 1000).all()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestHybridConfigs:
    def test_scorer_config_mode_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from npnet.config import ScorerConfig

        monkeypatch.setattr(
            sys, "argv", ["prog", "--ranking_dir", "/tmp/ranking", "--mode", "invalid"]
        )
        with pytest.raises(Exception):  # noqa: B017
            ScorerConfig()

    def test_scorer_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from npnet.config import ScorerConfig

        monkeypatch.setattr(sys, "argv", ["prog", "--ranking_dir", "/tmp/ranking"])
        config = ScorerConfig()
        assert config.mode == "pairwise"
        assert config.epochs == 50
        assert config.scorer_hidden_dim == 64

    def test_hybrid_inference_config_mode_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from npnet.config import HybridInferenceConfig

        monkeypatch.setattr(sys, "argv", ["prog", "--hybrid_mode", "invalid"])
        with pytest.raises(Exception):  # noqa: B017
            HybridInferenceConfig()

    def test_hybrid_inference_config_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from npnet.config import HybridInferenceConfig

        monkeypatch.setattr(sys, "argv", ["prog"])
        config = HybridInferenceConfig()
        assert config.hybrid_mode == "edit"
        assert config.rerank_k == 16
        assert config.inference_mode == "replacement"

    def test_hybrid_training_config_dataset_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from npnet.config import HybridTrainingConfig

        monkeypatch.setattr(
            sys, "argv", ["prog", "--dataset_type", "inversion_local"]
        )
        with pytest.raises(Exception):  # noqa: B017
            HybridTrainingConfig()
