"""CLI entrypoint for seed-mining image generation."""

from __future__ import annotations

from seed_mining.config import SeedMiningConfig
from seed_mining.generator import run_generation


def main() -> None:
    """Parse CLI args via pydantic-settings and run generation."""
    config = SeedMiningConfig()
    run_generation(config)


if __name__ == "__main__":
    main()
