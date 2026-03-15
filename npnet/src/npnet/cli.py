"""CLI entrypoints for NPNet data collection, training, and inference."""

from __future__ import annotations


def collect() -> None:
    """Collect noise pairs via DDIM inversion."""
    from npnet.config import DataCollectionConfig
    from npnet.data_collection import run_data_collection

    config = DataCollectionConfig()
    run_data_collection(config)


def train() -> None:
    """Train NPNet on collected noise pairs."""
    from npnet.config import TrainingConfig
    from npnet.trainer import run_training

    config = TrainingConfig()  # type: ignore[call-arg]  # CLI provides required args
    run_training(config)


def generate() -> None:
    """Generate images using golden noise from a trained NPNet."""
    from npnet.config import InferenceConfig
    from npnet.inference import run_golden_generation

    config = InferenceConfig()  # type: ignore[call-arg]  # CLI provides required args
    run_golden_generation(config)


def build_delta_noise() -> None:
    """Build delta-noise training pairs from VLM rankings."""
    from npnet.config import DeltaNoiseBuildConfig
    from npnet.delta_noise import DeltaNoiseConfig, run_build_delta_pairs

    cli_cfg = DeltaNoiseBuildConfig()  # type: ignore[call-arg]
    config = DeltaNoiseConfig(
        ranking_dir=cli_cfg.ranking_dir,
        out_dir=cli_cfg.out_dir,
        categories=cli_cfg.categories,
        min_accuracy=cli_cfg.min_accuracy,
        max_accuracy=cli_cfg.max_accuracy,
        num_good_seeds=cli_cfg.num_good_seeds,
        num_bad_seeds=cli_cfg.num_bad_seeds,
        latent_shape=(
            cli_cfg.latent_channels,
            cli_cfg.latent_resolution,
            cli_cfg.latent_resolution,
        ),
        train_frac=cli_cfg.train_frac,
        val_frac=cli_cfg.val_frac,
        seed=cli_cfg.seed,
    )
    run_build_delta_pairs(config)


def benchmark() -> None:
    """Run end-to-end SDXL benchmark with NPNet golden noise."""
    from npnet.benchmark import run_benchmark
    from npnet.config import BenchmarkConfig

    config = BenchmarkConfig()  # type: ignore[call-arg]
    run_benchmark(config)


def analyze_delta() -> None:
    """Analyze delta-noise dataset statistics."""
    from npnet.config import DeltaDiagnosticsConfig
    from npnet.delta_diagnostics import run_diagnostics

    config = DeltaDiagnosticsConfig()  # type: ignore[call-arg]
    run_diagnostics(config)
