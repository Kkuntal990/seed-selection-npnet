"""Datasets for seed scorer training from VLM ranking data.

Two modes:
- Pairwise: for each prompt, enumerate (correct_seed, incorrect_seed) pairs
- Pointwise: each sample is (noise, prompt_text, label)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from npnet.inversion_local import (
    load_prompt_accuracy,
    load_prompt_texts,
    load_seed_prompt_matrix,
    split_prompts_deterministic,
)

logger = logging.getLogger(__name__)


def _noise_from_seed(
    seed: int,
    latent_channels: int = 4,
    height: int = 1024,
    width: int = 1024,
) -> torch.Tensor:
    """Generate deterministic noise from a seed (CPU generator for reproducibility)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(latent_channels, height // 8, width // 8, generator=gen)


class PairwiseSeedScoreDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Pairwise dataset: (good_noise, bad_noise, prompt_text) for ranking loss.

    For each prompt, enumerate all (correct_seed, incorrect_seed) pairs from the
    VLM ranking matrix. Noise is generated on-the-fly from the seed.
    """

    def __init__(
        self,
        ranking_dir: Path,
        categories: list[str],
        prompt_ids: list[int] | None = None,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
        max_pairs_per_prompt: int = 50,
    ) -> None:
        self.latent_channels = latent_channels
        self.height = height
        self.width = width

        # Collect (good_seed, bad_seed, prompt_text) tuples
        self.pairs: list[tuple[int, int, str]] = []

        for category in categories:
            cat_dir = ranking_dir / category
            if not cat_dir.exists():
                logger.warning("Ranking dir %s does not exist, skipping", cat_dir)
                continue

            matrix = load_seed_prompt_matrix(cat_dir / "seed_prompt_matrix.csv")
            prompt_texts = load_prompt_texts(cat_dir / "prompt_accuracy.json")

            # Use all prompt_ids from matrix if not specified
            all_pids = sorted(
                {pid for seed_map in matrix.values() for pid in seed_map}
            )
            if prompt_ids is not None:
                pids = [pid for pid in all_pids if pid in prompt_ids]
            else:
                pids = all_pids

            for pid in pids:
                text = prompt_texts.get(pid, "")
                if not text:
                    continue

                good_seeds = [s for s, pm in matrix.items() if pm.get(pid, False)]
                bad_seeds = [s for s, pm in matrix.items() if not pm.get(pid, False)]

                if not good_seeds or not bad_seeds:
                    continue

                # Limit pairs per prompt to avoid imbalance
                count = 0
                for gs in good_seeds:
                    for bs in bad_seeds:
                        if count >= max_pairs_per_prompt:
                            break
                        self.pairs.append((gs, bs, text))
                        count += 1
                    if count >= max_pairs_per_prompt:
                        break

        logger.info("PairwiseSeedScoreDataset: %d pairs from %s", len(self.pairs), categories)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        good_seed, bad_seed, prompt_text = self.pairs[idx]
        good_noise = _noise_from_seed(good_seed, self.latent_channels, self.height, self.width)
        bad_noise = _noise_from_seed(bad_seed, self.latent_channels, self.height, self.width)
        return good_noise, bad_noise, prompt_text


class PointwiseSeedScoreDataset(Dataset[tuple[torch.Tensor, str, float]]):
    """Pointwise dataset: (noise, prompt_text, label) for binary classification.

    Each sample is a (seed, prompt) with label 1.0 if the seed is correct for
    the prompt, 0.0 otherwise.
    """

    def __init__(
        self,
        ranking_dir: Path,
        categories: list[str],
        prompt_ids: list[int] | None = None,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
    ) -> None:
        self.latent_channels = latent_channels
        self.height = height
        self.width = width

        # Collect (seed, prompt_text, label) tuples
        self.samples: list[tuple[int, str, float]] = []

        for category in categories:
            cat_dir = ranking_dir / category
            if not cat_dir.exists():
                logger.warning("Ranking dir %s does not exist, skipping", cat_dir)
                continue

            matrix = load_seed_prompt_matrix(cat_dir / "seed_prompt_matrix.csv")
            prompt_texts = load_prompt_texts(cat_dir / "prompt_accuracy.json")

            all_pids = sorted(
                {pid for seed_map in matrix.values() for pid in seed_map}
            )
            if prompt_ids is not None:
                pids = [pid for pid in all_pids if pid in prompt_ids]
            else:
                pids = all_pids

            for pid in pids:
                text = prompt_texts.get(pid, "")
                if not text:
                    continue
                for seed, seed_map in matrix.items():
                    if pid in seed_map:
                        label = 1.0 if seed_map[pid] else 0.0
                        self.samples.append((seed, text, label))

        logger.info(
            "PointwiseSeedScoreDataset: %d samples from %s", len(self.samples), categories
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, float]:
        seed, prompt_text, label = self.samples[idx]
        noise = _noise_from_seed(seed, self.latent_channels, self.height, self.width)
        return noise, prompt_text, label


class PrecomputedEmbedScorerDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Wraps a pairwise scorer dataset, replacing prompt strings with pre-computed embeddings.

    Returns (good_noise, bad_noise, prompt_embed) for pairwise mode.
    """

    def __init__(
        self,
        base_dataset: PairwiseSeedScoreDataset,
        embed_cache: dict[str, torch.Tensor],
    ) -> None:
        self.base = base_dataset
        self.cache = embed_cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        good_noise, bad_noise, prompt_text = self.base[idx]
        return good_noise, bad_noise, self.cache[prompt_text]


class PrecomputedEmbedPointwiseDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Wraps a pointwise scorer dataset, replacing prompt strings with pre-computed embeddings.

    Returns (noise, prompt_embed, label_tensor) for pointwise mode.
    """

    def __init__(
        self,
        base_dataset: PointwiseSeedScoreDataset,
        embed_cache: dict[str, torch.Tensor],
    ) -> None:
        self.base = base_dataset
        self.cache = embed_cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise, prompt_text, label = self.base[idx]
        return noise, self.cache[prompt_text], torch.tensor([label])


def get_scorer_prompt_ids(
    ranking_dir: Path,
    categories: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Get deterministic train/val/test prompt ID splits for scorer training.

    Collects all prompt IDs across categories and splits them.
    """
    all_pids: set[int] = set()
    for category in categories:
        cat_dir = ranking_dir / category
        if not cat_dir.exists():
            continue
        prompt_acc = load_prompt_accuracy(cat_dir / "prompt_accuracy.json")
        all_pids.update(prompt_acc.keys())

    sorted_pids = sorted(all_pids)
    return split_prompts_deterministic(sorted_pids, train_frac, val_frac, seed)
