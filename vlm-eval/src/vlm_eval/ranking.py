"""Seed ranking: per-seed, per-prompt, and per-subtask accuracy with chi-squared test."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seed_mining.logging_utils import setup_logging

from vlm_eval.response_parser import extract_spatial_answer, get_number_from_response

if TYPE_CHECKING:
    from vlm_eval.eval_config import RankingConfig

logger = logging.getLogger("vlm_eval.ranking")


@dataclass
class SeedScore:
    """Accuracy score for one seed."""

    seed: int
    correct: int
    total: int
    accuracy: float


@dataclass
class PromptScore:
    """Accuracy score for one prompt across all seeds."""

    prompt_id: int
    prompt_text: str
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


def _is_correct(rec: dict[str, Any], category: str) -> bool:
    """Determine if a single evaluation record is correct."""
    if category == "numeracy":
        detected = get_number_from_response(rec["vlm_response"])
        return detected is not None and detected == rec["count_target"]
    # spatial
    answer = extract_spatial_answer(rec.get("vlm_response_2", ""))
    return answer is True


# ---------------------------------------------------------------------------
# Per-seed accuracy (overall)
# ---------------------------------------------------------------------------


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
        if _is_correct(rec, category):
            seed_correct[seed] += 1

    scores: dict[int, SeedScore] = {}
    for seed in sorted(seed_total):
        c = seed_correct[seed]
        t = seed_total[seed]
        scores[seed] = SeedScore(seed=seed, correct=c, total=t, accuracy=c / t if t > 0 else 0.0)
    return scores


# ---------------------------------------------------------------------------
# Per-subtask seed accuracy (per-count / per-relation)
# ---------------------------------------------------------------------------


def _subtask_key(rec: dict[str, Any], category: str) -> str:
    """Extract the subtask grouping key from a record.

    Numeracy: count_target (2, 3, 4, 5, 6)
    Spatial:  expected_relation ("on the left of", etc.)
    """
    if category == "numeracy":
        return str(rec["count_target"])
    return str(rec.get("expected_relation", "unknown"))


def compute_subtask_seed_accuracy(
    responses: list[dict[str, Any]],
    category: str,
) -> dict[str, dict[int, SeedScore]]:
    """Compute per-seed accuracy grouped by subtask.

    Returns ``{subtask_key: {seed: SeedScore}}``.
    For numeracy subtask keys are "2", "3", "4", "5", "6".
    For spatial subtask keys are relation strings.
    """
    # {subtask: {seed: [correct, total]}}
    buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for rec in responses:
        key = _subtask_key(rec, category)
        seed = rec["seed"]
        buckets[key][seed][1] += 1
        if _is_correct(rec, category):
            buckets[key][seed][0] += 1

    result: dict[str, dict[int, SeedScore]] = {}
    for subtask in sorted(buckets):
        scores: dict[int, SeedScore] = {}
        for seed in sorted(buckets[subtask]):
            c, t = buckets[subtask][seed]
            scores[seed] = SeedScore(
                seed=seed, correct=c, total=t, accuracy=c / t if t > 0 else 0.0
            )
        result[subtask] = scores
    return result


# ---------------------------------------------------------------------------
# Per-prompt accuracy
# ---------------------------------------------------------------------------


def compute_prompt_accuracy(
    responses: list[dict[str, Any]],
    category: str,
) -> dict[int, PromptScore]:
    """Compute per-prompt accuracy across all seeds."""
    prompt_correct: dict[int, int] = defaultdict(int)
    prompt_total: dict[int, int] = defaultdict(int)
    prompt_text_map: dict[int, str] = {}

    for rec in responses:
        pid = rec["prompt_id"]
        prompt_total[pid] += 1
        prompt_text_map[pid] = rec.get("prompt_text", "")
        if _is_correct(rec, category):
            prompt_correct[pid] += 1

    scores: dict[int, PromptScore] = {}
    for pid in sorted(prompt_total):
        c = prompt_correct[pid]
        t = prompt_total[pid]
        scores[pid] = PromptScore(
            prompt_id=pid,
            prompt_text=prompt_text_map.get(pid, ""),
            correct=c,
            total=t,
            accuracy=c / t if t > 0 else 0.0,
        )
    return scores


# ---------------------------------------------------------------------------
# Seed × Prompt matrix
# ---------------------------------------------------------------------------


def compute_seed_prompt_matrix(
    responses: list[dict[str, Any]],
    category: str,
) -> dict[tuple[int, int], bool]:
    """Build a (seed, prompt_id) → correct mapping."""
    matrix: dict[tuple[int, int], bool] = {}
    for rec in responses:
        key = (rec["seed"], rec["prompt_id"])
        matrix[key] = _is_correct(rec, category)
    return matrix


def write_seed_prompt_csv(
    matrix: dict[tuple[int, int], bool],
    seeds: list[int],
    prompt_ids: list[int],
    path: Path,
) -> None:
    """Write seed × prompt correctness matrix as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [f"p{pid}" for pid in prompt_ids])
        for seed in seeds:
            row = [seed] + [int(matrix.get((seed, pid), False)) for pid in prompt_ids]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _write_subtask_ranking(
    subtask_scores: dict[str, dict[int, SeedScore]],
    category: str,
    top_k: int,
    out_dir: Path,
) -> None:
    """Write per-subtask seed ranking files.

    Produces:
    - subtask_seeds.json   — full ranking per subtask
    - subtask_report.txt   — human-readable top/bottom per subtask
    """
    subtask_label = "count" if category == "numeracy" else "relation"

    all_subtask_data: dict[str, Any] = {"category": category, "subtasks": {}}

    report_lines = [f"=== {category.upper()} Per-{subtask_label.title()} Seed Ranking ===", ""]

    for subtask, scores in sorted(subtask_scores.items()):
        ranked = sorted(scores.values(), key=lambda s: s.accuracy, reverse=True)

        # JSON data
        all_subtask_data["subtasks"][subtask] = {
            "num_seeds": len(ranked),
            "seeds_ranked": [asdict(s) for s in ranked],
        }

        # Report section
        report_lines.append(f"--- {subtask_label}={subtask} ({len(ranked)} seeds) ---")
        report_lines.append(f"  Top {top_k}:")
        for s in ranked[:top_k]:
            report_lines.append(
                f"    seed={s.seed:3d}  accuracy={s.accuracy:.3f}  ({s.correct}/{s.total})"
            )
        report_lines.append(f"  Bottom {top_k}:")
        for s in ranked[-top_k:]:
            report_lines.append(
                f"    seed={s.seed:3d}  accuracy={s.accuracy:.3f}  ({s.correct}/{s.total})"
            )
        report_lines.append("")

    (out_dir / "subtask_seeds.json").write_text(json.dumps(all_subtask_data, indent=2) + "\n")
    report = "\n".join(report_lines)
    (out_dir / "subtask_report.txt").write_text(report)
    logger.info("\n%s", report)


# ---------------------------------------------------------------------------
# Main ranking entry point
# ---------------------------------------------------------------------------


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

        out_dir = config.output_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Per-seed ranking (overall) ---
        scores = compute_seed_accuracy(responses, category)
        ranked = sorted(scores.values(), key=lambda s: s.accuracy, reverse=True)

        chi2, p_value = chi_square_test(scores)
        logger.info("[%s] Chi-squared: %.2f, p-value: %.2e", category, chi2, p_value)

        result = {
            "category": category,
            "num_seeds": len(ranked),
            "chi2_statistic": chi2,
            "chi2_p_value": p_value,
            "seeds_ranked": [asdict(s) for s in ranked],
        }
        (out_dir / "ranked_seeds.json").write_text(json.dumps(result, indent=2) + "\n")

        lines = [
            f"=== {category.upper()} Seed Ranking (Overall) ===",
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

        # --- 2. Per-subtask seed ranking (per-count / per-relation) ---
        subtask_scores = compute_subtask_seed_accuracy(responses, category)
        _write_subtask_ranking(subtask_scores, category, config.top_k, out_dir)

        # --- 3. Per-prompt ranking ---
        prompt_scores = compute_prompt_accuracy(responses, category)
        prompt_ranked = sorted(prompt_scores.values(), key=lambda s: s.accuracy, reverse=True)

        prompt_result = {
            "category": category,
            "num_prompts": len(prompt_ranked),
            "prompts_ranked": [asdict(s) for s in prompt_ranked],
        }
        (out_dir / "prompt_accuracy.json").write_text(json.dumps(prompt_result, indent=2) + "\n")

        p_lines = [
            f"=== {category.upper()} Per-Prompt Accuracy ===",
            f"Prompts evaluated: {len(prompt_ranked)}",
            "",
            f"Easiest {config.top_k} prompts (highest accuracy):",
        ]
        for ps in prompt_ranked[: config.top_k]:
            p_lines.append(
                f"  prompt_id={ps.prompt_id:4d}  accuracy={ps.accuracy:.3f}"
                f"  ({ps.correct}/{ps.total})  {ps.prompt_text[:60]}"
            )
        p_lines.append("")
        p_lines.append(f"Hardest {config.top_k} prompts (lowest accuracy):")
        for ps in prompt_ranked[-config.top_k :]:
            p_lines.append(
                f"  prompt_id={ps.prompt_id:4d}  accuracy={ps.accuracy:.3f}"
                f"  ({ps.correct}/{ps.total})  {ps.prompt_text[:60]}"
            )
        p_lines.append("")
        p_report = "\n".join(p_lines)
        (out_dir / "prompt_ranking_report.txt").write_text(p_report)
        logger.info("\n%s", p_report)

        # --- 4. Seed × Prompt matrix ---
        matrix = compute_seed_prompt_matrix(responses, category)
        all_seeds = sorted({rec["seed"] for rec in responses})
        all_prompt_ids = sorted({rec["prompt_id"] for rec in responses})
        write_seed_prompt_csv(matrix, all_seeds, all_prompt_ids, out_dir / "seed_prompt_matrix.csv")
        logger.info(
            "[%s] Seed×Prompt matrix written (%d seeds × %d prompts)",
            category,
            len(all_seeds),
            len(all_prompt_ids),
        )
