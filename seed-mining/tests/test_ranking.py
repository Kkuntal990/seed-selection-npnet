"""Tests for seed ranking and accuracy computation."""

from __future__ import annotations

from seed_mining.ranking import SeedScore, chi_square_test, compute_seed_accuracy


class TestComputeSeedAccuracy:
    def test_perfect_seed(self) -> None:
        responses = [
            {"seed": 0, "prompt_id": i, "vlm_response": f"There are {c} items.", "count_target": c}
            for i, c in enumerate([2, 3, 4])
        ]
        scores = compute_seed_accuracy(responses, "numeracy")
        assert scores[0].accuracy == 1.0
        assert scores[0].correct == 3

    def test_zero_accuracy(self) -> None:
        responses = [
            {"seed": 0, "prompt_id": 0, "vlm_response": "I see 5 items.", "count_target": 3},
            {"seed": 0, "prompt_id": 1, "vlm_response": "I see 2 items.", "count_target": 4},
        ]
        scores = compute_seed_accuracy(responses, "numeracy")
        assert scores[0].accuracy == 0.0

    def test_multiple_seeds(self) -> None:
        responses = [
            {"seed": 0, "prompt_id": 0, "vlm_response": "3 items", "count_target": 3},
            {"seed": 0, "prompt_id": 1, "vlm_response": "5 items", "count_target": 3},
            {"seed": 1, "prompt_id": 0, "vlm_response": "3 items", "count_target": 3},
            {"seed": 1, "prompt_id": 1, "vlm_response": "3 items", "count_target": 3},
        ]
        scores = compute_seed_accuracy(responses, "numeracy")
        assert scores[0].accuracy == 0.5
        assert scores[1].accuracy == 1.0

    def test_spatial_yes_no(self) -> None:
        responses = [
            {"seed": 0, "prompt_id": 0, "vlm_response_2": "Yes, the apple is on the left."},
            {"seed": 0, "prompt_id": 1, "vlm_response_2": "No, the apple is on the right."},
            {"seed": 1, "prompt_id": 0, "vlm_response_2": "Yes"},
        ]
        scores = compute_seed_accuracy(responses, "spatial")
        assert scores[0].correct == 1
        assert scores[0].total == 2
        assert scores[1].correct == 1
        assert scores[1].total == 1


class TestChiSquaredTest:
    def test_different_seeds(self) -> None:
        """Seeds with very different accuracies should give low p-value."""
        scores = {
            0: SeedScore(seed=0, correct=90, total=100, accuracy=0.9),
            1: SeedScore(seed=1, correct=10, total=100, accuracy=0.1),
        }
        chi2, p = chi_square_test(scores)
        assert chi2 > 0
        assert p < 0.01

    def test_similar_seeds(self) -> None:
        """Seeds with equal accuracies should give high p-value."""
        scores = {
            i: SeedScore(seed=i, correct=50, total=100, accuracy=0.5)
            for i in range(10)
        }
        chi2, p = chi_square_test(scores)
        assert p > 0.9
