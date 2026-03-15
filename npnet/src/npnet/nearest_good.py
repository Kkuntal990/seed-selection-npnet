"""Build nearest-good seed transport pairs from VLM ranking data."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger("npnet.nearest_good")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NearestGoodPair:
    """One source -> target noise transport pair."""

    source_seed: int
    target_seed: int
    prompt_id: int
    category: str
    prompt_text: str
    l2_distance: float


@dataclass
class NearestGoodConfig:
    """Configuration for building nearest-good seed transport pairs."""

    ranking_dir: Path
    out_dir: Path
    categories: list[str] = field(default_factory=lambda: ["numeracy", "spatial"])
    min_accuracy: float = 0.10
    max_accuracy: float = 0.90
    top_k_golden: int = 5
    num_source_seeds: int = 20
    latent_shape: tuple[int, ...] = (4, 128, 128)
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")


# ---------------------------------------------------------------------------
# Loading ranking data
# ---------------------------------------------------------------------------


def load_seed_prompt_matrix(csv_path: Path) -> dict[int, dict[int, bool]]:
    """Load seed_prompt_matrix.csv -> {seed: {prompt_id: correct}}.

    CSV header: ``seed,p0,p1,p2,...``  —  values are 0/1.
    Seeds are discovered from the rows (handles gaps).
    """
    matrix: dict[int, dict[int, bool]] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        assert all(col.startswith("p") for col in header[1:]), (
            f"Expected seed_prompt_matrix.csv columns to be p0,p1,..., got: {header[1:5]}"
        )
        prompt_ids = [int(col[1:]) for col in header[1:]]  # strip 'p' prefix
        for row in reader:
            seed = int(row[0])
            matrix[seed] = {pid: val == "1" for pid, val in zip(prompt_ids, row[1:])}
    return matrix


def load_prompt_accuracy(json_path: Path) -> dict[int, float]:
    """Load prompt_accuracy.json -> {prompt_id: accuracy}."""
    data = json.loads(json_path.read_text())
    return {p["prompt_id"]: p["accuracy"] for p in data["prompts_ranked"]}


def load_prompt_texts(json_path: Path) -> dict[int, str]:
    """Load prompt_accuracy.json -> {prompt_id: prompt_text}."""
    data = json.loads(json_path.read_text())
    return {p["prompt_id"]: p["prompt_text"] for p in data["prompts_ranked"]}


def load_seed_overall_accuracy(json_path: Path) -> dict[int, float]:
    """Load ranked_seeds.json -> {seed: accuracy} for tiebreaking."""
    data = json.loads(json_path.read_text())
    return {s["seed"]: s["accuracy"] for s in data["seeds_ranked"]}


# ---------------------------------------------------------------------------
# Filtering & selection
# ---------------------------------------------------------------------------


def filter_prompts_by_difficulty(
    prompt_accuracies: dict[int, float],
    min_accuracy: float = 0.10,
    max_accuracy: float = 0.90,
) -> list[int]:
    """Return prompt_ids with accuracy in [min_accuracy, max_accuracy]."""
    return sorted(
        pid
        for pid, acc in prompt_accuracies.items()
        if min_accuracy <= acc <= max_accuracy
    )


def get_golden_seeds_for_prompt(
    matrix: dict[int, dict[int, bool]],
    prompt_id: int,
    top_k: int,
    seed_overall_acc: dict[int, float],
) -> list[int]:
    """Return top-K seeds that are correct for *prompt_id*.

    Correct seeds are ranked by overall accuracy (tiebreaker), then truncated
    to *top_k*.  Returns empty list if no seed is correct for this prompt.
    """
    correct_seeds = [
        seed for seed, pmap in matrix.items() if pmap.get(prompt_id, False)
    ]
    if not correct_seeds:
        return []
    # Sort by overall accuracy descending, then seed ascending for stability
    correct_seeds.sort(key=lambda s: (-seed_overall_acc.get(s, 0.0), s))
    return correct_seeds[:top_k]


def sample_source_seeds(
    non_golden_seeds: list[int],
    n: int = 20,
    rng: torch.Generator | None = None,
) -> list[int]:
    """Randomly sample *n* source seeds from the non-golden pool.

    If the pool is smaller than *n*, return all of them.
    """
    if len(non_golden_seeds) <= n:
        return list(non_golden_seeds)
    indices = torch.randperm(len(non_golden_seeds), generator=rng)[: n].tolist()
    return [non_golden_seeds[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Latent generation
# ---------------------------------------------------------------------------


def generate_latent(seed: int, shape: tuple[int, ...] = (4, 128, 128)) -> torch.Tensor:
    """Generate deterministic latent noise with CPU generator.

    Matches ``inference.py:81-88``: generates in float16 with a CPU generator,
    then upcasts to float32 for training (``trainer.py:109`` calls ``.float()``).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(1, *shape, generator=generator, dtype=torch.float16)
    return latent.squeeze(0).float()


# ---------------------------------------------------------------------------
# Nearest-golden search
# ---------------------------------------------------------------------------


def find_nearest_golden(
    source_latent: torch.Tensor,
    golden_latents: dict[int, torch.Tensor],
) -> tuple[int, float]:
    """Find golden seed whose latent is nearest to *source_latent* in L2.

    Returns ``(golden_seed, l2_distance)``.
    """
    best_seed = -1
    best_dist = float("inf")
    for seed, golden in golden_latents.items():
        dist = torch.norm(source_latent.float() - golden.float(), p=2).item()
        if dist < best_dist:
            best_seed = seed
            best_dist = dist
    return best_seed, best_dist


# ---------------------------------------------------------------------------
# Prompt splitting
# ---------------------------------------------------------------------------


def split_prompts_deterministic(
    prompt_ids: list[int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split *prompt_ids* into train / val / test deterministically.

    ``test_frac = 1 - train_frac - val_frac``.
    """
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(prompt_ids), generator=rng).tolist()

    n_train = int(len(prompt_ids) * train_frac)
    n_val = int(len(prompt_ids) * val_frac)

    train_ids = sorted(prompt_ids[i] for i in indices[:n_train])
    val_ids = sorted(prompt_ids[i] for i in indices[n_train : n_train + n_val])
    test_ids = sorted(prompt_ids[i] for i in indices[n_train + n_val :])
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Pair building
# ---------------------------------------------------------------------------


def build_pairs_for_prompt(
    prompt_id: int,
    category: str,
    prompt_text: str,
    matrix: dict[int, dict[int, bool]],
    seed_overall_acc: dict[int, float],
    all_seeds: list[int],
    top_k_golden: int = 5,
    num_source_seeds: int = 20,
    latent_shape: tuple[int, ...] = (4, 128, 128),
    rng: torch.Generator | None = None,
) -> list[tuple[NearestGoodPair, torch.Tensor, torch.Tensor]]:
    """Build all nearest-good pairs for one prompt.

    Returns list of ``(pair, source_latent, target_latent)`` to avoid
    regenerating latents in the caller.
    """
    golden_seeds = get_golden_seeds_for_prompt(
        matrix, prompt_id, top_k_golden, seed_overall_acc
    )
    if not golden_seeds:
        logger.warning("[%s] prompt %d has no correct seeds, skipping", category, prompt_id)
        return []

    golden_set = set(golden_seeds)
    non_golden = [s for s in all_seeds if s not in golden_set]
    source_seeds = sample_source_seeds(non_golden, num_source_seeds, rng)

    # Pre-generate golden latents
    golden_latents = {s: generate_latent(s, latent_shape) for s in golden_seeds}

    results: list[tuple[NearestGoodPair, torch.Tensor, torch.Tensor]] = []
    for src_seed in source_seeds:
        src_latent = generate_latent(src_seed, latent_shape)
        tgt_seed, l2_dist = find_nearest_golden(src_latent, golden_latents)
        pair = NearestGoodPair(
            source_seed=src_seed,
            target_seed=tgt_seed,
            prompt_id=prompt_id,
            category=category,
            prompt_text=prompt_text,
            l2_distance=l2_dist,
        )
        results.append((pair, src_latent, golden_latents[tgt_seed]))
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_build_pairs(config: NearestGoodConfig) -> None:
    """Build nearest-good transport pairs for all categories."""
    # Set up logging so output is visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    rng = torch.Generator().manual_seed(config.seed)

    # Save run config
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "config.json").write_text(
        json.dumps(
            {
                "ranking_dir": str(config.ranking_dir),
                "out_dir": str(config.out_dir),
                "categories": config.categories,
                "min_accuracy": config.min_accuracy,
                "max_accuracy": config.max_accuracy,
                "top_k_golden": config.top_k_golden,
                "num_source_seeds": config.num_source_seeds,
                "latent_shape": list(config.latent_shape),
                "train_frac": config.train_frac,
                "val_frac": config.val_frac,
                "seed": config.seed,
            },
            indent=2,
        )
        + "\n"
    )

    total_pairs = 0

    # Clear any existing manifests to avoid duplicates on re-run
    for split_name in ("train", "val", "test"):
        manifest_path = config.out_dir / split_name / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

    for category in config.categories:
        cat_dir = config.ranking_dir / category
        if not cat_dir.exists():
            logger.warning("Ranking directory %s does not exist, skipping", cat_dir)
            continue

        # 1. Load ranking data
        matrix = load_seed_prompt_matrix(cat_dir / "seed_prompt_matrix.csv")
        prompt_acc = load_prompt_accuracy(cat_dir / "prompt_accuracy.json")
        prompt_texts = load_prompt_texts(cat_dir / "prompt_accuracy.json")
        seed_overall_acc = load_seed_overall_accuracy(cat_dir / "ranked_seeds.json")
        all_seeds = sorted(matrix.keys())

        logger.info(
            "[%s] Loaded %d seeds, %d prompts", category, len(all_seeds), len(prompt_acc)
        )

        # 2. Filter prompts by difficulty
        filtered_pids = filter_prompts_by_difficulty(
            prompt_acc, config.min_accuracy, config.max_accuracy
        )
        logger.info(
            "[%s] %d / %d prompts pass difficulty filter [%.2f, %.2f]",
            category,
            len(filtered_pids),
            len(prompt_acc),
            config.min_accuracy,
            config.max_accuracy,
        )

        if not filtered_pids:
            logger.warning("[%s] No prompts passed difficulty filter", category)
            continue

        # 3. Split prompts into train/val/test
        train_pids, val_pids, test_pids = split_prompts_deterministic(
            filtered_pids, config.train_frac, config.val_frac, config.seed
        )
        logger.info(
            "[%s] Split: train=%d, val=%d, test=%d",
            category,
            len(train_pids),
            len(val_pids),
            len(test_pids),
        )

        # Save filtered prompts info
        (config.out_dir / f"{category}_filtered_prompts.json").write_text(
            json.dumps(
                {
                    "category": category,
                    "total_prompts": len(prompt_acc),
                    "filtered_prompts": len(filtered_pids),
                    "train": train_pids,
                    "val": val_pids,
                    "test": test_pids,
                },
                indent=2,
            )
            + "\n"
        )

        # 4. Build pairs for each split
        for split_name, pids in [
            ("train", train_pids),
            ("val", val_pids),
            ("test", test_pids),
        ]:
            manifest_records: list[dict[str, Any]] = []

            for pid in tqdm(pids, desc=f"{category}/{split_name}", unit="prompt"):
                triplets = build_pairs_for_prompt(
                    prompt_id=pid,
                    category=category,
                    prompt_text=prompt_texts.get(pid, ""),
                    matrix=matrix,
                    seed_overall_acc=seed_overall_acc,
                    all_seeds=all_seeds,
                    top_k_golden=config.top_k_golden,
                    num_source_seeds=config.num_source_seeds,
                    latent_shape=config.latent_shape,
                    rng=rng,
                )

                for pair, src_latent, tgt_latent in triplets:
                    rel_path = (
                        f"{category}/prompt_{pid:04d}"
                        f"/src={pair.source_seed:03d}_tgt={pair.target_seed:03d}.pt"
                    )
                    pt_path = config.out_dir / split_name / rel_path
                    pt_path.parent.mkdir(parents=True, exist_ok=True)

                    torch.save(
                        {
                            "source_noise": src_latent,
                            "target_noise": tgt_latent,
                            "prompt_text": pair.prompt_text,
                            "source_seed": pair.source_seed,
                            "target_seed": pair.target_seed,
                            "prompt_id": pair.prompt_id,
                            "category": pair.category,
                            "l2_distance": pair.l2_distance,
                        },
                        pt_path,
                    )

                    manifest_records.append(
                        {
                            "path": rel_path,
                            "source_seed": pair.source_seed,
                            "target_seed": pair.target_seed,
                            "prompt_id": pair.prompt_id,
                            "category": pair.category,
                            "prompt_text": pair.prompt_text,
                            "l2_distance": pair.l2_distance,
                        }
                    )

            # Write manifest (overwrite to avoid duplicates on re-run)
            manifest_path = config.out_dir / split_name / "manifest.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a") as f:
                for rec in manifest_records:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")

            total_pairs += len(manifest_records)
            logger.info(
                "[%s/%s] Wrote %d pairs", category, split_name, len(manifest_records)
            )

    logger.info("Done. Total pairs: %d", total_pairs)
