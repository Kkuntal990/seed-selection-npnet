"""Seed ranking: compute per-seed accuracy and chi-squared test."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seed_mining.logging_utils import setup_logging
from seed_mining.response_parser import extract_spatial_answer, get_number_from_response

if TYPE_CHECKING:
    from seed_mining.eval_config import RankingConfig

logger = logging.getLogger("seed_mining.ranking")


@dataclass
class SeedScore:
    """Accuracy score for one seed."""

    seed: int
    correct: int
    total: int
    accuracy: float


def load_eval_responses(path: Path) -> list[dict[str, Any]]:
    """Load evaluation responses from a JSONL file."""
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def compute_seed_accuracy(
    responses: list[dict[str, Any]],
    category: str,
) -> dict[int, SeedScore]:
    """Compute per-seed accuracy from evaluation responses."""
    seed_correct: dict[int, int] = defaultdict(int)
    seed_total: dict[int, int] = defaultdict(int)

    for rec in responses:
        seed = rec["seed"]
        seed_total[seed] += 1

        if category == "numeracy":
            detected = get_number_from_response(rec["vlm_response"])
            if detected is not None and detected == rec["count_target"]:
                seed_correct[seed] += 1
        else:
            answer = extract_spatial_answer(rec.get("vlm_response_2", ""))
            if answer is True:
                seed_correct[seed] += 1

    scores: dict[int, SeedScore] = {}
    for seed in sorted(seed_total):
        c = seed_correct[seed]
        t = seed_total[seed]
        scores[seed] = SeedScore(seed=seed, correct=c, total=t, accuracy=c / t if t > 0 else 0.0)
    return scores


def chi_square_test(seed_scores: dict[int, SeedScore]) -> tuple[float, float]:
    """Run chi-squared contingency test on seed correct/incorrect counts.

    Returns ``(chi2_statistic, p_value)``.
    """
    import numpy as np
    from scipy.stats import chi2_contingency

    # Build 2 x N contingency table: [correct, incorrect] per seed
    table = np.array(
        [[s.correct, s.total - s.correct] for s in seed_scores.values()],
        dtype=np.int64,
    )

    # chi2_contingency expects (n_categories, n_outcomes) — transpose to (2, n_seeds)
    chi2, p, _dof, _expected = chi2_contingency(table.T)
    return float(chi2), float(p)


def run_ranking(config: RankingConfig) -> None:
    """Analyze evaluation results and rank seeds."""
    setup_logging(config.output_dir / "logs", rank=0)

    for category in ("numeracy", "spatial"):
        responses_path = config.eval_results_dir / "responses" / f"{category}_responses.jsonl"
        if not responses_path.exists():
            logger.info("No %s responses found, skipping", category)
            continue

        responses = load_eval_responses(responses_path)
        if not responses:
            logger.info("Empty %s responses, skipping", category)
            continue

        logger.info("[%s] Loaded %d responses", category, len(responses))

        scores = compute_seed_accuracy(responses, category)
        ranked = sorted(scores.values(), key=lambda s: s.accuracy, reverse=True)

        # Chi-squared test
        chi2, p_value = chi_square_test(scores)
        logger.info("[%s] Chi-squared: %.2f, p-value: %.2e", category, chi2, p_value)

        # Write outputs
        out_dir = config.output_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        # ranked_seeds.json
        result = {
            "category": category,
            "num_seeds": len(ranked),
            "chi2_statistic": chi2,
            "chi2_p_value": p_value,
            "seeds_ranked": [asdict(s) for s in ranked],
        }
        (out_dir / "ranked_seeds.json").write_text(json.dumps(result, indent=2) + "\n")

        # ranking_report.txt
        lines = [
            f"=== {category.upper()} Seed Ranking ===",
            f"Seeds evaluated: {len(ranked)}",
            f"Chi-squared: {chi2:.2f} (p={p_value:.2e})",
            "",
            f"Top {config.top_k}:",
        ]
        for s in ranked[: config.top_k]:
            lines.append(f"  seed={s.seed:3d}  accuracy={s.accuracy:.3f}  ({s.correct}/{s.total})")
        lines.append("")
        lines.append(f"Bottom {config.top_k}:")
        for s in ranked[-config.top_k :]:
            lines.append(f"  seed={s.seed:3d}  accuracy={s.accuracy:.3f}  ({s.correct}/{s.total})")
        lines.append("")

        report = "\n".join(lines)
        (out_dir / "ranking_report.txt").write_text(report)
        logger.info("\n%s", report)
