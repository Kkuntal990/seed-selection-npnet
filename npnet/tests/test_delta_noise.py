"""Tests for delta-noise pair building."""

from __future__ import annotations

import torch

from npnet.delta_noise import (
    build_delta_pairs_for_prompt,
    filter_prompts_by_difficulty,
    generate_latent,
    get_bad_seeds_for_prompt,
    get_good_seeds_for_prompt,
    split_prompts_deterministic,
)


def _make_matrix() -> dict[int, dict[int, bool]]:
    """Create a small correctness matrix for testing.

    Seeds 0-9, prompts 0-2.
    Prompt 0: seeds 0,1,2 correct, rest incorrect.
    Prompt 1: seeds 3,4,5 correct, rest incorrect.
    Prompt 2: all correct (too easy).
    """
    matrix: dict[int, dict[int, bool]] = {}
    for seed in range(10):
        matrix[seed] = {
            0: seed in (0, 1, 2),
            1: seed in (3, 4, 5),
            2: True,  # all seeds correct
        }
    return matrix


def _make_seed_acc() -> dict[int, float]:
    """Per-seed overall accuracy (higher seed = higher accuracy for sorting)."""
    return {s: s / 10.0 for s in range(10)}


class TestGetGoodSeeds:
    def test_returns_correct_seeds(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        good = get_good_seeds_for_prompt(matrix, 0, top_k=3, seed_overall_acc=seed_acc)
        assert set(good) == {0, 1, 2}

    def test_ranked_by_accuracy_descending(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        good = get_good_seeds_for_prompt(matrix, 0, top_k=3, seed_overall_acc=seed_acc)
        # Seed 2 has highest accuracy (0.2), then 1 (0.1), then 0 (0.0)
        assert good == [2, 1, 0]

    def test_top_k_limits(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        good = get_good_seeds_for_prompt(matrix, 0, top_k=2, seed_overall_acc=seed_acc)
        assert len(good) == 2

    def test_no_correct_seeds(self) -> None:
        matrix: dict[int, dict[int, bool]] = {0: {0: False}, 1: {0: False}}
        good = get_good_seeds_for_prompt(matrix, 0, top_k=3, seed_overall_acc={})
        assert good == []


class TestGetBadSeeds:
    def test_returns_incorrect_seeds(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        bad = get_bad_seeds_for_prompt(matrix, 0, bottom_k=3, seed_overall_acc=seed_acc)
        # Seeds 3-9 are incorrect for prompt 0, worst first
        assert all(s not in (0, 1, 2) for s in bad)
        assert len(bad) == 3

    def test_ranked_by_accuracy_ascending(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        bad = get_bad_seeds_for_prompt(matrix, 0, bottom_k=3, seed_overall_acc=seed_acc)
        # Worst seeds first: 3 (0.3), 4 (0.4), 5 (0.5)
        assert bad == [3, 4, 5]


class TestBuildDeltaPairs:
    def test_pair_count(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        pairs = build_delta_pairs_for_prompt(
            prompt_id=0,
            category="numeracy",
            prompt_text="test",
            matrix=matrix,
            seed_overall_acc=seed_acc,
            num_good_seeds=3,
            num_bad_seeds=3,
        )
        # 3 good × 3 bad = 9 pairs
        assert len(pairs) == 9

    def test_delta_correctness(self) -> None:
        matrix = _make_matrix()
        seed_acc = _make_seed_acc()
        pairs = build_delta_pairs_for_prompt(
            prompt_id=0,
            category="numeracy",
            prompt_text="test",
            matrix=matrix,
            seed_overall_acc=seed_acc,
            num_good_seeds=1,
            num_bad_seeds=1,
        )
        pair, source_noise, delta_noise = pairs[0]
        # delta = z_good - z_bad
        z_bad = generate_latent(pair.bad_seed)
        z_good = generate_latent(pair.good_seed)
        expected_delta = z_good - z_bad
        assert torch.allclose(delta_noise, expected_delta, atol=1e-6)
        assert torch.allclose(source_noise, z_bad, atol=1e-6)

    def test_no_pairs_when_no_correct_seeds(self) -> None:
        matrix: dict[int, dict[int, bool]] = {0: {0: False}, 1: {0: False}}
        pairs = build_delta_pairs_for_prompt(
            prompt_id=0,
            category="numeracy",
            prompt_text="test",
            matrix=matrix,
            seed_overall_acc={},
            num_good_seeds=3,
            num_bad_seeds=3,
        )
        assert pairs == []


class TestFilterPrompts:
    def test_filters_by_range(self) -> None:
        accs = {0: 0.05, 1: 0.50, 2: 0.95, 3: 0.10, 4: 0.90}
        filtered = filter_prompts_by_difficulty(accs, 0.10, 0.90)
        assert set(filtered) == {1, 3, 4}

    def test_empty_when_none_pass(self) -> None:
        accs = {0: 0.01, 1: 0.99}
        filtered = filter_prompts_by_difficulty(accs, 0.10, 0.90)
        assert filtered == []


class TestSplitPrompts:
    def test_no_overlap(self) -> None:
        pids = list(range(100))
        train, val, test = split_prompts_deterministic(pids, 0.8, 0.1, seed=42)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_covers_all(self) -> None:
        pids = list(range(100))
        train, val, test = split_prompts_deterministic(pids, 0.8, 0.1, seed=42)
        assert set(train) | set(val) | set(test) == set(pids)

    def test_deterministic(self) -> None:
        pids = list(range(50))
        r1 = split_prompts_deterministic(pids, 0.8, 0.1, seed=42)
        r2 = split_prompts_deterministic(pids, 0.8, 0.1, seed=42)
        assert r1 == r2


class TestGenerateLatent:
    def test_shape(self) -> None:
        z = generate_latent(42)
        assert z.shape == (4, 128, 128)

    def test_deterministic(self) -> None:
        z1 = generate_latent(42)
        z2 = generate_latent(42)
        assert torch.equal(z1, z2)

    def test_different_seeds(self) -> None:
        z1 = generate_latent(0)
        z2 = generate_latent(1)
        assert not torch.equal(z1, z2)
