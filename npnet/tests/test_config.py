"""Tests for NPNet configuration classes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from npnet.config import (
    DataCollectionConfig,
    DeltaNoiseBuildConfig,
    InferenceConfig,
    TrainingConfig,
)


class TestDataCollectionConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        config = DataCollectionConfig()
        assert config.model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        assert config.num_inference_steps == 50
        assert config.num_inversion_steps == 10
        assert config.guidance_scale == 5.5
        assert config.inversion_guidance_scale == 1.0
        assert config.height == 1024
        assert config.width == 1024
        assert config.seed_range == 100
        assert config.dry_run is False

    def test_seeds_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--seed_start", "5", "--seed_range", "3"])
        config = DataCollectionConfig()
        assert config.seeds == [5, 6, 7]

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog"])
        monkeypatch.setenv("NPDC_DRY_RUN", "true")
        config = DataCollectionConfig()
        assert config.dry_run is True


class TestTrainingConfig:
    def test_from_cli_ddim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--noise_pairs_dir",
                "/tmp/pairs",
                "--prompt_manifest_path",
                "/tmp/manifest.jsonl",
            ],
        )
        config = TrainingConfig()
        assert config.noise_pairs_dir == Path("/tmp/pairs")
        assert config.prompt_manifest_path == Path("/tmp/manifest.jsonl")
        assert config.dataset_type == "ddim"
        assert config.epochs == 30
        assert config.batch_size == 64
        assert config.lr == 1e-4
        assert config.val_split == 0.1

    def test_from_cli_delta_noise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--dataset_type",
                "delta_noise",
                "--delta_noise_dir",
                "/tmp/delta",
            ],
        )
        config = TrainingConfig()
        assert config.dataset_type == "delta_noise"
        assert config.delta_noise_dir == Path("/tmp/delta")

    def test_invalid_dataset_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--dataset_type", "nearest_good"],
        )
        with pytest.raises(Exception):  # noqa: B017
            TrainingConfig()

    def test_architecture_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--noise_pairs_dir",
                "/tmp/pairs",
                "--prompt_manifest_path",
                "/tmp/manifest.jsonl",
            ],
        )
        config = TrainingConfig()
        assert config.latent_channels == 4
        assert config.latent_resolution == 128
        assert config.text_embed_dim == 2048
        assert config.text_seq_len == 77


class TestInferenceConfig:
    def test_from_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--npnet_checkpoint", "/tmp/npnet_best.pth"],
        )
        config = InferenceConfig()
        assert config.npnet_checkpoint == Path("/tmp/npnet_best.pth")
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 5.5
        assert config.image_format == "jpg"

    def test_image_format_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--npnet_checkpoint", "/tmp/x.pth", "--image_format", "bmp"],
        )
        with pytest.raises(Exception):  # noqa: B017
            InferenceConfig()

    def test_seeds_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--npnet_checkpoint",
                "/tmp/x.pth",
                "--seed_start",
                "10",
                "--seed_range",
                "5",
            ],
        )
        config = InferenceConfig()
        assert config.seeds == [10, 11, 12, 13, 14]

    def test_inference_mode_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--npnet_checkpoint", "/tmp/x.pth"])
        config = InferenceConfig()
        assert config.inference_mode == "delta"

    def test_inference_mode_replacement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--npnet_checkpoint", "/tmp/x.pth", "--inference_mode", "replacement"],
        )
        config = InferenceConfig()
        assert config.inference_mode == "replacement"

    def test_inference_mode_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--npnet_checkpoint", "/tmp/x.pth", "--inference_mode", "bad"],
        )
        with pytest.raises(Exception):  # noqa: B017
            InferenceConfig()


class TestDeltaNoiseBuildConfig:
    def test_from_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--ranking_dir", "/tmp/ranking"],
        )
        config = DeltaNoiseBuildConfig()
        assert config.ranking_dir == Path("/tmp/ranking")
        assert config.num_good_seeds == 3
        assert config.num_bad_seeds == 3
        assert config.min_accuracy == 0.10
        assert config.max_accuracy == 0.90
        assert config.train_frac == 0.8
        assert config.val_frac == 0.1
