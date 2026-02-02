"""Tests for prompt generation: counts, ordering, stability, format."""

from __future__ import annotations

from pathlib import Path

import pytest

from seed_mining.prompts import (
    NB_TO_WORD,
    build_all_prompts,
    build_numeracy_prompts,
    build_spatial_prompts,
    load_backgrounds,
    load_objects,
    load_spatial_prompts,
)

DATASET_DIR = Path(__file__).resolve().parent.parent / "prompt_dataset"


# ---------------------------------------------------------------------------
# Skip if dataset files not present (e.g. CI without data)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not (DATASET_DIR / "objects_train.txt").exists(),
    reason="prompt_dataset/ files not found",
)

# Actual counts from the official Reliable-Random-Seeds repo
N_OBJECTS_TRAIN = 60
N_OBJECTS_EVAL = 30
N_BACKGROUNDS_TRAIN = 8
N_BACKGROUNDS_EVAL = 4
N_SPATIAL_TRAIN = 320
N_SPATIAL_EVAL = 160


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


class TestObjectLoading:
    def test_train_count(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        assert len(objs) == N_OBJECTS_TRAIN

    def test_eval_count(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_eval.txt")
        assert len(objs) == N_OBJECTS_EVAL

    def test_format_three_parts(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        for singular, article, plural in objs:
            assert singular, "singular should not be empty"
            assert article, "article form should not be empty"
            assert plural, "plural should not be empty"


class TestBackgroundLoading:
    def test_train_count(self) -> None:
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        assert len(bgs) == N_BACKGROUNDS_TRAIN

    def test_eval_count(self) -> None:
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_eval.txt")
        assert len(bgs) == N_BACKGROUNDS_EVAL


class TestSpatialLoading:
    def test_train_count(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_train.txt")
        assert len(lines) == N_SPATIAL_TRAIN

    def test_eval_count(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_eval.txt")
        assert len(lines) == N_SPATIAL_EVAL


# ---------------------------------------------------------------------------
# Numeracy prompts
# ---------------------------------------------------------------------------


class TestNumeracyPrompts:
    def test_train_count(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        assert len(prompts) == 5 * N_OBJECTS_TRAIN * N_BACKGROUNDS_TRAIN

    def test_eval_count(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_eval.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_eval.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        assert len(prompts) == 5 * N_OBJECTS_EVAL * N_BACKGROUNDS_EVAL

    def test_with_limits(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs, num_objects=3, num_settings=2)
        assert len(prompts) == 5 * 3 * 2

    def test_ids_sequential(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        ids = [p.prompt_id for p in prompts]
        assert ids == list(range(len(prompts)))

    def test_template_format(self) -> None:
        """Each numeracy prompt should match '{word} {plural}, {background}'."""
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        count_words = set(NB_TO_WORD.values())
        for p in prompts:
            first_word = p.text.split()[0]
            assert first_word in count_words, f"Unexpected first word: {first_word!r}"
            assert ", " in p.text, f"Missing ', ' separator: {p.text!r}"

    def test_no_duplicate_texts(self) -> None:
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        texts = [p.text for p in prompts]
        assert len(texts) == len(set(texts)), "Duplicate numeracy prompt texts"

    def test_ordering_count_first(self) -> None:
        """Prompts should be ordered: count asc -> object -> background."""
        objs = load_objects(DATASET_DIR / "objects_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_numeracy_prompts(objs, bgs)
        counts = [p.metadata["count_target"] for p in prompts]
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], f"Count ordering broken at index {i}"


# ---------------------------------------------------------------------------
# Spatial prompts
# ---------------------------------------------------------------------------


class TestSpatialPrompts:
    def test_train_with_background(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_spatial_prompts(lines, bgs, append_background=True)
        assert len(prompts) == N_SPATIAL_TRAIN * N_BACKGROUNDS_TRAIN

    def test_eval_with_background(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_eval.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_eval.txt")
        prompts = build_spatial_prompts(lines, bgs, append_background=True)
        assert len(prompts) == N_SPATIAL_EVAL * N_BACKGROUNDS_EVAL

    def test_without_background(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_spatial_prompts(lines, bgs, append_background=False)
        assert len(prompts) == N_SPATIAL_TRAIN

    def test_ids_sequential(self) -> None:
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_spatial_prompts(lines, bgs, append_background=True)
        ids = [p.prompt_id for p in prompts]
        assert ids == list(range(len(prompts)))

    def test_no_duplicate_prompt_ids(self) -> None:
        """Each prompt_id must be unique (texts may have duplicates from source data)."""
        lines = load_spatial_prompts(DATASET_DIR / "spatial_prompts_train.txt")
        bgs = load_backgrounds(DATASET_DIR / "backgrounds_train.txt")
        prompts = build_spatial_prompts(lines, bgs, append_background=True)
        ids = [p.prompt_id for p in prompts]
        assert len(ids) == len(set(ids)), "Duplicate spatial prompt IDs"


# ---------------------------------------------------------------------------
# build_all_prompts integration
# ---------------------------------------------------------------------------


class TestBuildAllPrompts:
    def test_train_split(self) -> None:
        numeracy, spatial = build_all_prompts(DATASET_DIR, "train")
        assert len(numeracy) == 5 * N_OBJECTS_TRAIN * N_BACKGROUNDS_TRAIN
        assert len(spatial) == N_SPATIAL_TRAIN * N_BACKGROUNDS_TRAIN

    def test_eval_split(self) -> None:
        numeracy, spatial = build_all_prompts(DATASET_DIR, "eval")
        assert len(numeracy) == 5 * N_OBJECTS_EVAL * N_BACKGROUNDS_EVAL
        assert len(spatial) == N_SPATIAL_EVAL * N_BACKGROUNDS_EVAL

    def test_all_split(self) -> None:
        numeracy, spatial = build_all_prompts(DATASET_DIR, "all")
        n_obj = N_OBJECTS_TRAIN + N_OBJECTS_EVAL
        n_bg = N_BACKGROUNDS_TRAIN + N_BACKGROUNDS_EVAL
        n_sp = N_SPATIAL_TRAIN + N_SPATIAL_EVAL
        assert len(numeracy) == 5 * n_obj * n_bg
        assert len(spatial) == n_sp * n_bg

    def test_stability(self) -> None:
        """Two calls yield identical prompts."""
        a_num, a_sp = build_all_prompts(DATASET_DIR, "all")
        b_num, b_sp = build_all_prompts(DATASET_DIR, "all")
        for pa, pb in zip(a_num, b_num, strict=True):
            assert pa.prompt_id == pb.prompt_id
            assert pa.text == pb.text
        for pa, pb in zip(a_sp, b_sp, strict=True):
            assert pa.prompt_id == pb.prompt_id
            assert pa.text == pb.text

    def test_no_background_mode(self) -> None:
        numeracy, spatial = build_all_prompts(
            DATASET_DIR, "train", append_background_to_spatial=False
        )
        assert len(spatial) == N_SPATIAL_TRAIN

    def test_paper_defaults_62k(self) -> None:
        """Paper configuration: 15 objects, 4 backgrounds, no bg on spatial, train split.

        Numeracy: 5 counts × 15 objects × 4 backgrounds = 300 per seed
        Spatial: 320 prompts (no backgrounds appended) per seed
        Total: 620 per seed × 100 seeds = 62,000 images
        """
        numeracy, spatial = build_all_prompts(
            DATASET_DIR,
            "train",
            num_objects=15,
            num_settings=4,
            append_background_to_spatial=False,
        )
        assert len(numeracy) == 300  # 5 × 15 × 4
        assert len(spatial) == 320
        assert len(numeracy) + len(spatial) == 620
