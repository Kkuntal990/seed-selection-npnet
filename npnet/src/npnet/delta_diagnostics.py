"""Diagnostics and analysis for delta-noise datasets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from npnet.config import DeltaDiagnosticsConfig

logger = logging.getLogger("npnet.delta_diagnostics")


def _load_all_pt_metadata(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load metadata from all .pt files across train/val/test splits."""
    records: list[dict[str, Any]] = []
    for pt_path in sorted(dataset_dir.rglob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        records.append(
            {
                "path": str(pt_path.relative_to(dataset_dir)),
                "bad_seed": int(data["bad_seed"]),
                "good_seed": int(data["good_seed"]),
                "prompt_id": int(data["prompt_id"]),
                "category": str(data["category"]),
                "delta_norm": float(data["delta_norm"]),
                "split": pt_path.relative_to(dataset_dir).parts[0],
            }
        )
    return records


def analyze_delta_norm_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze delta norm distribution across all pairs."""
    norms = [r["delta_norm"] for r in records]
    if not norms:
        return {"error": "no norms"}

    t = torch.tensor(norms)
    return {
        "count": len(norms),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p25": t.quantile(0.25).item(),
        "p50": t.quantile(0.50).item(),
        "p75": t.quantile(0.75).item(),
        "p95": t.quantile(0.95).item(),
    }


def analyze_good_seed_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze how many unique good seeds are targeted."""
    good_counts = Counter(r["good_seed"] for r in records)
    total = len(records)

    cat_counts: dict[str, Counter[int]] = {}
    for r in records:
        cat = r["category"]
        if cat not in cat_counts:
            cat_counts[cat] = Counter()
        cat_counts[cat][r["good_seed"]] += 1

    top_seed, top_count = good_counts.most_common(1)[0] if good_counts else (-1, 0)
    concentration = top_count / total if total > 0 else 0.0

    result: dict[str, Any] = {
        "unique_good_seeds": len(good_counts),
        "total_pairs": total,
        "most_common_good_seed": top_seed,
        "most_common_good_count": top_count,
        "most_common_good_fraction": concentration,
    }

    for cat, counts in cat_counts.items():
        top_s, top_c = counts.most_common(1)[0] if counts else (-1, 0)
        cat_total = sum(counts.values())
        result[f"{cat}_unique_good_seeds"] = len(counts)
        result[f"{cat}_most_common_good_seed"] = top_s
        result[f"{cat}_most_common_good_fraction"] = top_c / cat_total if cat_total > 0 else 0.0

    return result


def analyze_prompt_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze prompt coverage and split integrity."""
    split_prompts: dict[str, set[tuple[str, int]]] = {}
    for r in records:
        split = r["split"]
        key = (r["category"], r["prompt_id"])
        if split not in split_prompts:
            split_prompts[split] = set()
        split_prompts[split].add(key)

    # Check for prompt leakage across splits
    splits = list(split_prompts.keys())
    leaks: list[str] = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = split_prompts[s1] & split_prompts[s2]
            if overlap:
                leaks.append(f"{s1} & {s2}: {len(overlap)} shared prompts")

    # Pairs per prompt
    prompt_pair_counts = Counter((r["category"], r["prompt_id"]) for r in records)
    counts = list(prompt_pair_counts.values())

    return {
        "splits": {s: len(ps) for s, ps in split_prompts.items()},
        "total_unique_prompts": len(set().union(*split_prompts.values())),
        "prompt_leaks": leaks if leaks else "none",
        "pairs_per_prompt_mean": sum(counts) / len(counts) if counts else 0,
        "pairs_per_prompt_min": min(counts) if counts else 0,
        "pairs_per_prompt_max": max(counts) if counts else 0,
    }


def analyze_category_balance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze balance between categories."""
    cat_counts = Counter(r["category"] for r in records)
    total = len(records)
    return {
        "category_counts": dict(cat_counts),
        "category_fractions": (
            {c: cnt / total for c, cnt in cat_counts.items()} if total > 0 else {}
        ),
    }


def analyze_unique_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze unique (bad_seed, good_seed) pair diversity."""
    pairs = {(r["bad_seed"], r["good_seed"]) for r in records}
    bad_seeds = {r["bad_seed"] for r in records}
    good_seeds = {r["good_seed"] for r in records}
    return {
        "unique_bad_good_pairs": len(pairs),
        "unique_bad_seeds": len(bad_seeds),
        "unique_good_seeds": len(good_seeds),
        "total_records": len(records),
    }


def analyze_delta_statistics(dataset_dir: Path, max_samples: int = 500) -> dict[str, Any]:
    """Load actual delta vectors and compute detailed statistics.

    Samples up to *max_samples* files to avoid loading the full dataset.
    """
    pt_files = sorted(dataset_dir.rglob("*.pt"))
    if not pt_files:
        return {"error": "no .pt files"}

    sample_files = pt_files[:max_samples]
    deltas: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []

    for pt_path in sample_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        deltas.append(data["delta_noise"])
        sources.append(data["source_noise"])

    delta_stack = torch.stack(deltas)  # (N, C, H, W)
    source_stack = torch.stack(sources)

    # Per-channel statistics
    per_channel: list[dict[str, float]] = []
    for c in range(delta_stack.shape[1]):
        ch = delta_stack[:, c]
        per_channel.append(
            {
                "channel": c,
                "mean": ch.mean().item(),
                "std": ch.std().item(),
                "abs_mean": ch.abs().mean().item(),
            }
        )

    # Delta norm vs source norm ratio
    delta_norms = delta_stack.flatten(1).norm(dim=1)
    source_norms = source_stack.flatten(1).norm(dim=1)
    ratio = delta_norms / source_norms.clamp(min=1e-8)

    # Pairwise cosine similarity (sample)
    n_cos = min(100, len(deltas))
    flat = delta_stack[:n_cos].flatten(1)
    flat_norm = F.normalize(flat, dim=1)
    cos_matrix = flat_norm @ flat_norm.T
    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n_cos, n_cos, dtype=torch.bool), diagonal=1)
    cos_values = cos_matrix[mask]

    return {
        "num_samples": len(deltas),
        "per_channel": per_channel,
        "delta_norm_mean": delta_norms.mean().item(),
        "delta_norm_std": delta_norms.std().item(),
        "source_norm_mean": source_norms.mean().item(),
        "delta_source_ratio_mean": ratio.mean().item(),
        "delta_source_ratio_std": ratio.std().item(),
        "pairwise_cosine_mean": cos_values.mean().item() if len(cos_values) > 0 else 0.0,
        "pairwise_cosine_std": cos_values.std().item() if len(cos_values) > 0 else 0.0,
        "pairwise_cosine_min": cos_values.min().item() if len(cos_values) > 0 else 0.0,
        "pairwise_cosine_max": cos_values.max().item() if len(cos_values) > 0 else 0.0,
    }


def analyze_delta_by_task(dataset_dir: Path, max_per_category: int = 200) -> dict[str, Any]:
    """Analyze delta vectors grouped by task category.

    Checks whether numeracy and spatial deltas occupy different regions
    of latent space.
    """
    pt_files = sorted(dataset_dir.rglob("*.pt"))
    if not pt_files:
        return {"error": "no .pt files"}

    cat_deltas: dict[str, list[torch.Tensor]] = {}
    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        cat = str(data["category"])
        if cat not in cat_deltas:
            cat_deltas[cat] = []
        if len(cat_deltas[cat]) < max_per_category:
            cat_deltas[cat].append(data["delta_noise"])

    result: dict[str, Any] = {}
    cat_centroids: dict[str, torch.Tensor] = {}

    for cat, deltas in cat_deltas.items():
        stack = torch.stack(deltas).flatten(1)  # (N, D)
        centroid = stack.mean(dim=0)
        cat_centroids[cat] = centroid

        norms = stack.norm(dim=1)
        result[f"{cat}_count"] = len(deltas)
        result[f"{cat}_mean_norm"] = norms.mean().item()
        result[f"{cat}_std_norm"] = norms.std().item()

    # Cross-category cosine similarity of centroids
    cats = list(cat_centroids.keys())
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1 :]:
            cos = F.cosine_similarity(
                cat_centroids[c1].unsqueeze(0),
                cat_centroids[c2].unsqueeze(0),
            ).item()
            result[f"centroid_cosine_{c1}_vs_{c2}"] = cos

    return result


def run_diagnostics(config: DeltaDiagnosticsConfig) -> None:
    """Run all diagnostics and write report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading dataset from %s ...", config.dataset_dir)
    records = _load_all_pt_metadata(config.dataset_dir)
    logger.info("Loaded %d records", len(records))

    if not records:
        logger.error("No .pt files found in %s", config.dataset_dir)
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Run metadata analyses
    norm_dist = analyze_delta_norm_distribution(records)
    coverage = analyze_good_seed_coverage(records)
    prompts = analyze_prompt_coverage(records)
    balance = analyze_category_balance(records)
    pairs = analyze_unique_pairs(records)

    # Run tensor-level analyses
    delta_stats = analyze_delta_statistics(config.dataset_dir)
    task_analysis = analyze_delta_by_task(config.dataset_dir)

    report = {
        "delta_norm_distribution": norm_dist,
        "good_seed_coverage": coverage,
        "prompt_coverage": prompts,
        "category_balance": balance,
        "unique_pairs": pairs,
        "delta_statistics": delta_stats,
        "delta_by_task": task_analysis,
    }

    # Save JSON report
    report_path = config.output_dir / "diagnostics_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info("Report saved to %s", report_path)

    # Print summary
    logger.info("=== Delta-Noise Dataset Diagnostics ===")
    logger.info("Total pairs: %d", len(records))
    logger.info("Unique bad seeds: %d", pairs["unique_bad_seeds"])
    logger.info("Unique good seeds: %d", pairs["unique_good_seeds"])
    logger.info("Category balance: %s", balance["category_counts"])
    logger.info("Prompts per split: %s", prompts["splits"])
    logger.info("Prompt leaks: %s", prompts["prompt_leaks"])
    logger.info(
        "Delta norm: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        norm_dist.get("mean", 0),
        norm_dist.get("std", 0),
        norm_dist.get("min", 0),
        norm_dist.get("max", 0),
    )
    logger.info(
        "Delta/source ratio: mean=%.3f, std=%.3f",
        delta_stats.get("delta_source_ratio_mean", 0),
        delta_stats.get("delta_source_ratio_std", 0),
    )
    logger.info(
        "Pairwise cosine similarity: mean=%.4f, std=%.4f",
        delta_stats.get("pairwise_cosine_mean", 0),
        delta_stats.get("pairwise_cosine_std", 0),
    )
