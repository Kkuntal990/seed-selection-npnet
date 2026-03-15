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


def build_inversion_local() -> None:
    """Build inversion-local perturbation training pairs."""
    from npnet.config import InversionLocalBuildConfig
    from npnet.inversion_local import InversionLocalConfig, run_build_inversion_local_pairs

    cli_cfg = InversionLocalBuildConfig()  # type: ignore[call-arg]
    config = InversionLocalConfig(
        ranking_dir=cli_cfg.ranking_dir,
        out_dir=cli_cfg.out_dir,
        categories=cli_cfg.categories,
        min_accuracy=cli_cfg.min_accuracy,
        max_accuracy=cli_cfg.max_accuracy,
        top_k=cli_cfg.top_k,
        num_perturbations=cli_cfg.num_perturbations,
        perturbation_sigma=cli_cfg.perturbation_sigma,
        model_id=cli_cfg.model_id,
        num_inference_steps=cli_cfg.num_inference_steps,
        num_inversion_steps=cli_cfg.num_inversion_steps,
        guidance_scale=cli_cfg.guidance_scale,
        inversion_guidance_scale=cli_cfg.inversion_guidance_scale,
        height=cli_cfg.height,
        width=cli_cfg.width,
        train_frac=cli_cfg.train_frac,
        val_frac=cli_cfg.val_frac,
        seed=cli_cfg.seed,
        enable_cpu_offload=cli_cfg.enable_cpu_offload,
    )
    run_build_inversion_local_pairs(config)


def benchmark() -> None:
    """Run end-to-end SDXL benchmark with NPNet golden noise."""
    from npnet.benchmark import run_benchmark
    from npnet.config import BenchmarkConfig

    config = BenchmarkConfig()  # type: ignore[call-arg]
    run_benchmark(config)


def analyze_inversion_local() -> None:
    """Analyze inversion-local dataset statistics."""
    from npnet.config import InversionLocalDiagnosticsConfig
    from npnet.inversion_local_diagnostics import run_diagnostics

    config = InversionLocalDiagnosticsConfig()  # type: ignore[call-arg]
    run_diagnostics(config)
