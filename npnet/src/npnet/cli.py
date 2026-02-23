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

    config = TrainingConfig()
    run_training(config)


def generate() -> None:
    """Generate images using golden noise from a trained NPNet."""
    from npnet.config import InferenceConfig
    from npnet.inference import run_golden_generation

    config = InferenceConfig()
    run_golden_generation(config)
