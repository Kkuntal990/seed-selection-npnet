"""Diagnostics and analysis for nearest-good datasets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from npnet.config import DiagnosticsConfig

logger = logging.getLogger("npnet.diagnostics")


def _load_all_pt_metadata(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load metadata from all .pt files across train/val/test splits."""
    records: list[dict[str, Any]] = []
    for pt_path in sorted(dataset_dir.rglob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        records.append(
            {
                "path": str(pt_path.relative_to(dataset_dir)),
                "source_seed": int(data["source_seed"]),
                "target_seed": int(data["target_seed"]),
                "prompt_id": int(data["prompt_id"]),
                "category": str(data["category"]),
                "l2_distance": float(data["l2_distance"]),
                "split": pt_path.relative_to(dataset_dir).parts[0],
            }
        )
    return records


def analyze_golden_seed_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze how many unique golden seeds are targeted."""
    target_counts = Counter(r["target_seed"] for r in records)
    total = len(records)

    # Per-category
    cat_counts: dict[str, Counter[int]] = {}
    for r in records:
        cat = r["category"]
        if cat not in cat_counts:
            cat_counts[cat] = Counter()
        cat_counts[cat][r["target_seed"]] += 1

    top_seed, top_count = target_counts.most_common(1)[0] if target_counts else (-1, 0)
    concentration = top_count / total if total > 0 else 0.0

    result: dict[str, Any] = {
        "unique_target_seeds": len(target_counts),
        "total_pairs": total,
        "most_common_target_seed": top_seed,
        "most_common_target_count": top_count,
        "most_common_target_fraction": concentration,
        "WARNING_high_concentration": concentration > 0.40,
        "target_seed_counts": dict(target_counts.most_common()),
    }

    for cat, counts in cat_counts.items():
        top_s, top_c = counts.most_common(1)[0] if counts else (-1, 0)
        cat_total = sum(counts.values())
        result[f"{cat}_unique_targets"] = len(counts)
        result[f"{cat}_most_common_target"] = top_s
        result[f"{cat}_most_common_fraction"] = top_c / cat_total if cat_total > 0 else 0.0

    return result


def analyze_distance_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze L2 distance distribution across all pairs."""
    distances = [r["l2_distance"] for r in records]
    if not distances:
        return {"error": "no distances"}

    t = torch.tensor(distances)
    return {
        "count": len(distances),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p25": t.quantile(0.25).item(),
        "p50": t.quantile(0.50).item(),
        "p75": t.quantile(0.75).item(),
        "p95": t.quantile(0.95).item(),
    }


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
        "category_fractions": {
            c: cnt / total for c, cnt in cat_counts.items()
        }
        if total > 0
        else {},
    }


def analyze_unique_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze unique (source_seed, target_seed) pair diversity."""
    pairs = set((r["source_seed"], r["target_seed"]) for r in records)
    src_seeds = set(r["source_seed"] for r in records)
    tgt_seeds = set(r["target_seed"] for r in records)
    return {
        "unique_src_tgt_pairs": len(pairs),
        "unique_source_seeds": len(src_seeds),
        "unique_target_seeds": len(tgt_seeds),
        "total_records": len(records),
    }


def run_diagnostics(config: DiagnosticsConfig) -> None:
    """Run all diagnostics and write report."""
    logger.info("Loading dataset from %s ...", config.dataset_dir)
    records = _load_all_pt_metadata(config.dataset_dir)
    logger.info("Loaded %d records", len(records))

    if not records:
        logger.error("No .pt files found in %s", config.dataset_dir)
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Run analyses
    coverage = analyze_golden_seed_coverage(records)
    distances = analyze_distance_distribution(records)
    prompts = analyze_prompt_coverage(records)
    balance = analyze_category_balance(records)
    pairs = analyze_unique_pairs(records)

    report = {
        "golden_seed_coverage": coverage,
        "distance_distribution": distances,
        "prompt_coverage": prompts,
        "category_balance": balance,
        "unique_pairs": pairs,
    }

    # Save JSON report
    report_path = config.output_dir / "diagnostics_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info("Report saved to %s", report_path)

    # Print human-readable summary
    logger.info("=== Dataset Diagnostics ===")
    logger.info("Total pairs: %d", len(records))
    logger.info("Unique source seeds: %d", pairs["unique_source_seeds"])
    logger.info("Unique target seeds: %d", pairs["unique_target_seeds"])
    logger.info("Unique (src, tgt) pairs: %d", pairs["unique_src_tgt_pairs"])
    logger.info("Category balance: %s", balance["category_counts"])
    logger.info("Prompts per split: %s", prompts["splits"])
    logger.info("Prompt leaks: %s", prompts["prompt_leaks"])
    logger.info(
        "L2 distance: mean=%.2f, std=%.2f, min=%.2f, max=%.2f",
        distances.get("mean", 0),
        distances.get("std", 0),
        distances.get("min", 0),
        distances.get("max", 0),
    )

    if coverage.get("WARNING_high_concentration"):
        logger.warning(
            "HIGH CONCENTRATION: %.1f%% of targets come from seed %d",
            coverage["most_common_target_fraction"] * 100,
            coverage["most_common_target_seed"],
        )
