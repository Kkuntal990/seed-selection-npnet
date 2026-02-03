"""Configuration for seed-mining image generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, CliSettingsSource


class SeedMiningConfig(BaseSettings):
    """All generation parameters.

    Supports CLI args (``--model_id ...``), environment variables
    (prefix ``SM_``), and programmatic construction.
    """

    model_config = {"env_prefix": "SM_", "cli_parse_args": True}

    # --- Model ---
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"

    # --- Generation ---
    seed_start: int = Field(default=0, description="First seed index (inclusive)")
    seed_range: int = Field(default=100, description="Number of seeds starting from seed_start")
    seeds_file: Path | None = Field(default=None, description="File with one seed per line")
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    batch_size: int = 16
    generator_device: str = Field(
        default="cpu", description="Device for torch.Generator ('cpu' or 'cuda')"
    )
    scheduler: str = Field(default="ddim", description="Scheduler name (ddim, dpmsolver++)")
    disable_safety_checker: bool = True
    enable_xformers: bool = False
    enable_cpu_offload: bool = False

    # --- I/O ---
    out_dir: Path = Path("output")
    prompt_dataset_dir: Path = Field(
        default=Path("prompt_dataset"), description="Directory with prompt data files"
    )
    image_format: str = Field(default="jpg", description="Image format: jpg or png")

    # --- Dataset scope ---
    split: str = Field(
        default="train", description="Which split to use: 'train', 'eval', or 'all'"
    )
    num_objects: int | None = Field(
        default=15, description="Limit number of objects (None = use all)"
    )
    num_settings: int | None = Field(
        default=4, description="Limit number of backgrounds (None = use all)"
    )
    append_background_to_spatial: bool = Field(
        default=False, description="Append background settings to spatial prompts"
    )

    # --- Runtime ---
    dry_run: bool = False

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in ("train", "eval", "all"):
            raise ValueError(f"split must be 'train', 'eval', or 'all', got '{v}'")
        return v

    @field_validator("image_format")
    @classmethod
    def validate_image_format(cls, v: str) -> str:
        if v not in ("jpg", "png"):
            raise ValueError(f"image_format must be 'jpg' or 'png', got '{v}'")
        return v

    @field_validator("generator_device")
    @classmethod
    def validate_generator_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda"):
            raise ValueError(f"generator_device must be 'cpu' or 'cuda', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_paths(self) -> SeedMiningConfig:
        if not self.dry_run and self.seeds_file is not None and not self.seeds_file.exists():
            raise ValueError(f"seeds_file does not exist: {self.seeds_file}")
        return self

    @property
    def seeds(self) -> list[int]:
        """Return the list of seeds to generate."""
        if self.seeds_file is not None:
            return [int(line.strip()) for line in self.seeds_file.read_text().splitlines() if line.strip()]
        return list(range(self.seed_start, self.seed_start + self.seed_range))

    @property
    def torch_dtype_str(self) -> str:
        return "float16"

    def resolved_dict(self) -> dict[str, Any]:
        """Dump config as JSON-serializable dict with extra runtime info."""
        import diffusers
        import torch
        import transformers

        data = self.model_dump(mode="json")
        data["_versions"] = {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
        }
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            data["_git_hash"] = git_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            data["_git_hash"] = None
        return data

    def save_resolved(self, path: Path) -> None:
        """Write resolved config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.resolved_dict(), indent=2, default=str) + "\n")

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
