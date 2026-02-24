"""Configuration for VLM evaluation and seed ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, CliSettingsSource


class EvalConfig(BaseSettings):
    """Configuration for evaluating generated images with a VLM."""

    model_config = {"env_prefix": "SMEVAL_", "cli_parse_args": True}

    # --- VLM model ---
    vlm_model_id: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="HuggingFace model ID for the VLM",
    )
    quantize: str | None = Field(
        default="4bit", description="Quantization: '4bit', '8bit', or None"
    )

    # --- Input (must match generation config) ---
    images_dir: Path = Field(
        description="Directory containing generated images (generation out_dir)"
    )
    prompt_dataset_dir: Path = Field(
        default=Path("prompt_dataset"), description="Directory with prompt data files"
    )
    split: str = "train"
    num_objects: int | None = 15
    num_settings: int | None = 4
    append_background_to_spatial: bool = False
    seed_start: int = 0
    seed_range: int = 100
    image_format: str = "jpg"

    # --- Output ---
    eval_out_dir: Path = Field(
        default=Path("eval_output"), description="Evaluation results directory"
    )

    # --- Scope ---
    category: str = Field(
        default="all", description="Category to evaluate: 'numeracy', 'spatial', or 'all'"
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in ("numeracy", "spatial", "all"):
            raise ValueError(f"category must be 'numeracy', 'spatial', or 'all', got '{v}'")
        return v

    @field_validator("quantize")
    @classmethod
    def validate_quantize(cls, v: str | None) -> str | None:
        if v is not None and v not in ("4bit", "8bit"):
            raise ValueError(f"quantize must be '4bit', '8bit', or None, got '{v}'")
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


class RankingConfig(BaseSettings):
    """Configuration for seed ranking analysis."""

    model_config = {"env_prefix": "SMRANK_", "cli_parse_args": True}

    eval_results_dir: Path = Field(description="Directory with evaluation JSONL results")
    output_dir: Path = Field(default=Path("ranking_output"), description="Ranking output directory")
    top_k: int = Field(default=10, description="Number of top/bottom seeds to report")

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
