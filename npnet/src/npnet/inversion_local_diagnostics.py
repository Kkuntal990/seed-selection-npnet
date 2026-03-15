"""Diagnostics and analysis for inversion-local datasets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from npnet.config import InversionLocalDiagnosticsConfig

logger = logging.getLogger("npnet.inversion_local_diagnostics")


def _load_all_pt_metadata(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load metadata from all .pt files across train/val/test splits."""
    records: list[dict[str, Any]] = []
    for pt_path in sorted(dataset_dir.rglob("*.pt")):
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        records.append(
            {
                "path": str(pt_path.relative_to(dataset_dir)),
                "seed": int(data["seed"]),
                "prompt_id": int(data["prompt_id"]),
                "category": str(data["category"]),
                "perturbation_idx": int(data["perturbation_idx"]),
                "epsilon_norm": float(data["epsilon_norm"]),
                "sigma": float(data["sigma"]),
                "split": pt_path.relative_to(dataset_dir).parts[0],
            }
        )
    return records


def analyze_epsilon_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze ε norm distribution."""
    norms = [r["epsilon_norm"] for r in records]
    if not norms:
        return {"error": "no norms"}
    t = torch.tensor(norms)
    return {
        "count": len(norms),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "p50": t.quantile(0.50).item(),
        "p95": t.quantile(0.95).item(),
        "sigma_used": records[0]["sigma"] if records else 0.0,
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

    splits = list(split_prompts.keys())
    leaks: list[str] = []
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = split_prompts[s1] & split_prompts[s2]
            if overlap:
                leaks.append(f"{s1} & {s2}: {len(overlap)} shared prompts")

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


def analyze_perturbation_statistics(dataset_dir: Path, max_samples: int = 500) -> dict[str, Any]:
    """Load actual tensors and verify perturbation properties.

    Checks:
    - ||source - target|| / ||target|| ratio (should be ~sigma)
    - cosine(source, target) (should be ~1.0 for small perturbations)
    """
    pt_files = sorted(dataset_dir.rglob("*.pt"))
    if not pt_files:
        return {"error": "no .pt files"}

    sample_files = pt_files[:max_samples]
    ratios: list[float] = []
    cosines: list[float] = []
    target_norms: list[float] = []
    source_norms: list[float] = []

    for pt_path in sample_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
        source = data["source_noise"].flatten()
        target = data["target_noise"].flatten()

        target_norm = target.norm().item()
        source_norm = source.norm().item()
        diff_norm = (source - target).norm().item()

        ratios.append(diff_norm / target_norm if target_norm > 0 else 0.0)
        cos = F.cosine_similarity(source.unsqueeze(0), target.unsqueeze(0)).item()
        cosines.append(cos)
        target_norms.append(target_norm)
        source_norms.append(source_norm)

    def stats(vals: list[float]) -> dict[str, float]:
        t = torch.tensor(vals)
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
        }

    return {
        "num_samples": len(sample_files),
        "diff_over_target_ratio": stats(ratios),
        "cosine_source_target": stats(cosines),
        "target_norm": stats(target_norms),
        "source_norm": stats(source_norms),
    }


def run_diagnostics(config: InversionLocalDiagnosticsConfig) -> None:
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

    eps_dist = analyze_epsilon_distribution(records)
    prompts = analyze_prompt_coverage(records)
    balance = analyze_category_balance(records)
    pert_stats = analyze_perturbation_statistics(config.dataset_dir)

    report = {
        "epsilon_distribution": eps_dist,
        "prompt_coverage": prompts,
        "category_balance": balance,
        "perturbation_statistics": pert_stats,
    }

    report_path = config.output_dir / "diagnostics_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    logger.info("Report saved to %s", report_path)

    logger.info("=== Inversion-Local Dataset Diagnostics ===")
    logger.info("Total pairs: %d", len(records))
    logger.info("Category balance: %s", balance["category_counts"])
    logger.info("Prompts per split: %s", prompts["splits"])
    logger.info("Prompt leaks: %s", prompts["prompt_leaks"])
    logger.info(
        "Epsilon norm: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        eps_dist.get("mean", 0),
        eps_dist.get("std", 0),
        eps_dist.get("min", 0),
        eps_dist.get("max", 0),
    )

    diff_ratio = pert_stats.get("diff_over_target_ratio", {})
    cos_st = pert_stats.get("cosine_source_target", {})
    logger.info(
        "||source-target||/||target||: mean=%.6f, std=%.6f",
        diff_ratio.get("mean", 0),
        diff_ratio.get("std", 0),
    )
    logger.info(
        "cosine(source, target): mean=%.6f, std=%.6f",
        cos_st.get("mean", 0),
        cos_st.get("std", 0),
    )
