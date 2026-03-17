"""Configuration for NPNet data collection, training, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
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
    noise_pairs_dir: Path | None = Field(
        default=None, description="Directory with .npz noise pair files (required for ddim)"
    )
    prompt_manifest_path: Path | None = Field(
        default=None, description="JSONL file mapping prompt_id to text (required for ddim)"
    )
    dataset_type: str = Field(
        default="ddim",
        description="Dataset type: 'ddim' (original .npz) or 'inversion_local' (.pt)",
    )
    dataset_dir: Path | None = Field(
        default=None, description="Dataset dir with train/val splits (required for inversion_local)"
    )

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
    early_stopping_patience: int = 10

    # --- Runtime ---
    seed: int = 42
    enable_cpu_offload: bool = True

    @model_validator(mode="after")
    def _check_dataset_fields(self) -> TrainingConfig:
        if self.dataset_type == "ddim":
            if self.noise_pairs_dir is None or self.prompt_manifest_path is None:
                raise ValueError(
                    "--noise_pairs_dir and --prompt_manifest_path are required "
                    "when --dataset_type=ddim"
                )
        elif self.dataset_type == "inversion_local":
            if self.dataset_dir is None:
                raise ValueError("--dataset_dir is required when --dataset_type=inversion_local")
        else:
            raise ValueError(
                f"dataset_type must be 'ddim' or 'inversion_local', got '{self.dataset_type}'"
            )
        return self

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

    # --- Inference mode ---
    inference_mode: str = Field(
        default="replacement",
        description="'replacement' uses NPNet(z) directly, 'delta' applies z + NPNet(z)",
    )

    # --- Output ---
    out_dir: Path = Field(default=Path("golden_output"), description="Output directory for images")
    image_format: str = "jpg"

    # --- Runtime ---
    enable_cpu_offload: bool = False
    dry_run: bool = False

    @field_validator("inference_mode")
    @classmethod
    def validate_inference_mode(cls, v: str) -> str:
        if v not in ("delta", "replacement"):
            raise ValueError(f"inference_mode must be 'delta' or 'replacement', got '{v}'")
        return v

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


class InversionLocalBuildConfig(BaseSettings):
    """Configuration for building inversion-local perturbation pairs."""

    model_config = {"env_prefix": "NPIL_", "cli_parse_args": True}

    ranking_dir: Path = Field(description="VLM ranking output directory")
    out_dir: Path = Field(
        default=Path("data/inversion_local"),
        description="Output directory for .pt pair files",
    )
    categories: list[str] = Field(default=["numeracy", "spatial"])
    min_accuracy: float = 0.10
    max_accuracy: float = 0.90
    top_k: int = 3
    num_perturbations: int = 5
    perturbation_sigma: float = 0.05
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 50
    num_inversion_steps: int = 10
    guidance_scale: float = 5.5
    inversion_guidance_scale: float = 1.0
    height: int = 1024
    width: int = 1024
    train_frac: float = 0.8
    val_frac: float = 0.1
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


class InversionLocalDiagnosticsConfig(BaseSettings):
    """Configuration for inversion-local dataset analysis."""

    model_config = {"env_prefix": "NPDIAG_", "cli_parse_args": True}

    dataset_dir: Path = Field(description="Inversion-local dataset directory")
    output_dir: Path = Field(
        default=Path("diagnostics_output"),
        description="Output directory for reports",
    )

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


class ScorerConfig(BaseSettings):
    """Configuration for SeedScorer training (Phase A)."""

    model_config = {"env_prefix": "NPSC_", "cli_parse_args": True}

    # --- Data ---
    ranking_dir: Path = Field(description="VLM ranking output directory")
    categories: list[str] = Field(default=["numeracy", "spatial"])
    mode: str = Field(
        default="pairwise",
        description="Training mode: 'pairwise' (MarginRankingLoss) or 'pointwise' (BCE)",
    )
    max_pairs_per_prompt: int = 50

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    text_embed_dim: int = 2048
    text_seq_len: int = 77
    scorer_hidden_dim: int = 64
    height: int = 1024
    width: int = 1024

    # --- Training ---
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    margin: float = 0.5
    seed: int = 42

    # --- Checkpointing ---
    checkpoint_dir: Path = Field(default=Path("checkpoints/scorer"))
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("pairwise", "pointwise"):
            raise ValueError(f"mode must be 'pairwise' or 'pointwise', got '{v}'")
        return v

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


class HybridTrainingConfig(BaseSettings):
    """Configuration for hybrid editor training (Phase B)."""

    model_config = {"env_prefix": "NPHB_", "cli_parse_args": True}

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # --- Pretrained checkpoints ---
    npnet_checkpoint: Path | None = Field(
        default=None, description="Path to pretrained NPNet checkpoint"
    )
    scorer_checkpoint: Path | None = Field(
        default=None, description="Path to pretrained SeedScorer checkpoint"
    )

    # --- Data (reuses TrainingConfig pattern) ---
    dataset_type: str = Field(default="inversion_local")
    dataset_dir: Path | None = Field(default=None, description="Dataset directory")
    noise_pairs_dir: Path | None = None
    prompt_manifest_path: Path | None = None
    val_split: float = 0.1

    # --- Architecture ---
    latent_channels: int = 4
    latent_resolution: int = 128
    text_embed_dim: int = 2048
    text_seq_len: int = 77
    inference_mode: str = "replacement"

    # --- Loss weights ---
    lambda_edit: float = 1.0
    lambda_score: float = 1.0
    lambda_attn: float = 0.1
    lambda_norm: float = 0.01
    attn_loss_type: str = Field(
        default="denoising_consistency",
        description="Attention loss type: 'denoising_consistency' or 'none'",
    )

    # --- Freeze flags ---
    freeze_npnet: bool = True
    freeze_scorer: bool = True

    # --- Training ---
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    grad_accumulation_steps: int = 1
    num_workers: int = 4
    seed: int = 42

    # --- Checkpointing ---
    checkpoint_dir: Path = Field(default=Path("checkpoints/hybrid"))
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10

    @model_validator(mode="after")
    def _check_dataset_fields(self) -> HybridTrainingConfig:
        if self.dataset_type == "inversion_local" and self.dataset_dir is None:
            raise ValueError("--dataset_dir is required when --dataset_type=inversion_local")
        if (
            self.dataset_type == "ddim"
            and (self.noise_pairs_dir is None or self.prompt_manifest_path is None)
        ):
            raise ValueError(
                "--noise_pairs_dir and --prompt_manifest_path required for dataset_type=ddim"
            )
        # Prevent silent training on frozen random weights
        if self.freeze_npnet and self.npnet_checkpoint is None:
            raise ValueError(
                "--npnet_checkpoint is required when --freeze_npnet=true "
                "(otherwise training freezes random NPNet weights)"
            )
        if self.freeze_scorer and self.scorer_checkpoint is None and self.lambda_score > 0:
            raise ValueError(
                "--scorer_checkpoint is required when --freeze_scorer=true and lambda_score > 0 "
                "(otherwise training uses frozen random scorer weights)"
            )
        return self

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


class HybridInferenceConfig(BaseSettings):
    """Configuration for hybrid inference evaluation."""

    model_config = {"env_prefix": "NPHBI_", "cli_parse_args": True}

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # --- Checkpoints ---
    hybrid_checkpoint: Path | None = Field(
        default=None, description="Path to trained HybridNPNet checkpoint"
    )
    scorer_checkpoint: Path | None = Field(
        default=None, description="Path to trained SeedScorer checkpoint"
    )

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
    generator_device: str = "cpu"

    # --- Hybrid mode ---
    hybrid_mode: str = Field(
        default="edit",
        description="Inference mode: 'baseline', 'rerank', 'edit', 'rerank_edit'",
    )
    inference_mode: str = Field(
        default="replacement",
        description="NPNet mode: 'replacement' or 'delta'",
    )
    rerank_k: int = Field(default=16, description="Number of candidate seeds for reranking")

    # --- Output ---
    out_dir: Path = Field(default=Path("hybrid_output"), description="Output directory")
    image_format: str = "jpg"

    # --- Runtime ---
    enable_cpu_offload: bool = False
    dry_run: bool = False

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed_start, self.seed_start + self.seed_range))

    @field_validator("hybrid_mode")
    @classmethod
    def validate_hybrid_mode(cls, v: str) -> str:
        valid = ("baseline", "rerank", "edit", "rerank_edit")
        if v not in valid:
            raise ValueError(f"hybrid_mode must be one of {valid}, got '{v}'")
        return v

    @field_validator("inference_mode")
    @classmethod
    def validate_inference_mode(cls, v: str) -> str:
        if v not in ("delta", "replacement"):
            raise ValueError(f"inference_mode must be 'delta' or 'replacement', got '{v}'")
        return v

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


class BenchmarkConfig(BaseSettings):
    """Configuration for SDXL NPNet benchmarking."""

    model_config = {"env_prefix": "NPBM_", "cli_parse_args": True}

    # --- NPNet ---
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

    # --- Generation (must match evaluated images) ---
    seed_start: int = 0
    seed_range: int = 150
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    batch_size: int = 4
    generator_device: str = "cpu"
    image_format: str = "jpg"
    scheduler: str = "euler"

    # --- VLM eval ---
    vlm_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantize: str | None = "4bit"

    # --- Inference mode ---
    inference_mode: str = Field(
        default="replacement",
        description="'replacement' uses NPNet(z) directly, 'delta' applies z + NPNet(z)",
    )

    # --- Baseline comparison ---
    baseline_images_dir: Path | None = Field(
        default=None,
        description="Baseline SDXL images dir for human preference comparison",
    )

    # --- Output ---
    out_dir: Path = Field(default=Path("benchmark_output"), description="Output directory")

    # --- Phase control ---
    skip_generation: bool = False
    skip_evaluation: bool = False
    skip_ranking: bool = False

    # --- Runtime ---
    enable_cpu_offload: bool = False
    dry_run: bool = False

    @field_validator("inference_mode")
    @classmethod
    def validate_inference_mode(cls, v: str) -> str:
        if v not in ("delta", "replacement"):
            raise ValueError(f"inference_mode must be 'delta' or 'replacement', got '{v}'")
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
