"""Configuration for NPNet data collection, training, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, CliSettingsSource


class DataCollectionConfig(BaseSettings):
    """Configuration for collecting noise pairs via DDIM inversion."""

    model_config = {"env_prefix": "NPDC_", "cli_parse_args": True}

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # --- Prompts (reuse seed-mining prompt grid) ---
    prompt_dataset_dir: Path = Field(
        default=Path("prompt_dataset"), description="Directory with prompt data files"
    )
    split: str = "train"
    num_objects: int | None = 15
    num_settings: int | None = 4
    append_background_to_spatial: bool = False

    # --- DDIM inversion ---
    num_inference_steps: int = 50
    num_inversion_steps: int = 10
    guidance_scale: float = 5.5
    inversion_guidance_scale: float = 1.0

    # --- Seeds ---
    seed_start: int = 0
    seed_range: int = 100
    height: int = 1024
    width: int = 1024

    # --- Output ---
    out_dir: Path = Field(
        default=Path("noise_pairs"), description="Output directory for .npz files"
    )
    batch_size: int = 1

    # --- Runtime ---
    enable_cpu_offload: bool = True
    dry_run: bool = False

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_start + self.seed_range))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True),
            env_settings,
        )


class TrainingConfig(BaseSettings):
    """Configuration for NPNet training."""

    model_config = {"env_prefix": "NPTR_", "cli_parse_args": True}

    # --- Model (frozen, used for encode_prompt) ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # --- Data ---
    noise_pairs_dir: Path = Field(description="Directory with .npz noise pair files")
    prompt_manifest_path: Path = Field(description="JSONL file mapping prompt_id to text")

    # --- Architecture ---
    latent_channels: int = 4
    latent_resolution: int = 128
    text_embed_dim: int = 2048
    text_seq_len: int = 77

    # --- Training hyperparameters ---
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    grad_accumulation_steps: int = 1
    num_workers: int = 4
    val_split: float = 0.1

    # --- Checkpointing ---
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"), description="Directory for saving model checkpoints"
    )
    pretrained_path: Path | None = None
    save_every_n_epochs: int = 5

    # --- Runtime ---
    seed: int = 42
    enable_cpu_offload: bool = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True),
            env_settings,
        )


class InferenceConfig(BaseSettings):
    """Configuration for generating images with golden noise."""

    model_config = {"env_prefix": "NPINF_", "cli_parse_args": True}

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    npnet_checkpoint: Path = Field(description="Path to trained NPNet .pth checkpoint")

    # --- Prompts ---
    prompt_dataset_dir: Path = Field(
        default=Path("prompt_dataset"), description="Directory with prompt data files"
    )
    split: str = "train"
    num_objects: int | None = 15
    num_settings: int | None = 4
    append_background_to_spatial: bool = False

    # --- Generation ---
    seed_start: int = 0
    seed_range: int = 100
    num_inference_steps: int = 50
    guidance_scale: float = 5.5
    height: int = 1024
    width: int = 1024
    batch_size: int = 4
    generator_device: str = "cpu"

    # --- Output ---
    out_dir: Path = Field(default=Path("golden_output"), description="Output directory for images")
    image_format: str = "jpg"

    # --- Runtime ---
    enable_cpu_offload: bool = False
    dry_run: bool = False

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, v: str) -> str:
        if v not in ("jpg", "png"):
            raise ValueError(f"image_format must be 'jpg' or 'png', got '{v}'")
        return v

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_start + self.seed_range))

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True),
            env_settings,
        )
