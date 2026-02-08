"""CLI entrypoints for VLM evaluation and seed ranking."""

from __future__ import annotations


def eval_run() -> None:
    """Run VLM evaluation on generated images."""
    from vlm_eval.eval_config import EvalConfig
    from vlm_eval.evaluator import run_evaluation

    config = EvalConfig()  # type: ignore[call-arg]
    run_evaluation(config)


def eval_rank() -> None:
    """Analyze evaluation results and rank seeds."""
    from vlm_eval.eval_config import RankingConfig
    from vlm_eval.ranking import run_ranking

    config = RankingConfig()  # type: ignore[call-arg]
    run_ranking(config)
