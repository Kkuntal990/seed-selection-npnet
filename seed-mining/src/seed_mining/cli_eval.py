"""CLI entrypoints for VLM evaluation and seed ranking."""

from __future__ import annotations


def eval_run() -> None:
    """Run VLM evaluation on generated images."""
    from seed_mining.eval_config import EvalConfig
    from seed_mining.evaluator import run_evaluation

    config = EvalConfig()  # type: ignore[call-arg]
    run_evaluation(config)


def eval_rank() -> None:
    """Analyze evaluation results and rank seeds."""
    from seed_mining.eval_config import RankingConfig
    from seed_mining.ranking import run_ranking

    config = RankingConfig()  # type: ignore[call-arg]
    run_ranking(config)
