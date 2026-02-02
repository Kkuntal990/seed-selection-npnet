"""Tests for path generation, atomic writes, and resume detection."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from seed_mining.io_utils import (
    find_existing_images,
    image_path,
    save_image_atomic,
)
from seed_mining.prompts import build_all_prompts

DATASET_DIR = Path(__file__).resolve().parent.parent / "prompt_dataset"

pytestmark = pytest.mark.skipif(
    not (DATASET_DIR / "objects_train.txt").exists(),
    reason="prompt_dataset/ files not found",
)


class TestImagePath:
    def test_format(self) -> None:
        p = image_path(Path("/out"), "numeracy", 42, 301, "jpg")
        assert p == Path("/out/images/numeracy/seed=042/prompt=0301.jpg")

    def test_png_format(self) -> None:
        p = image_path(Path("/out"), "spatial", 0, 0, "png")
        assert p == Path("/out/images/spatial/seed=000/prompt=0000.png")

    def test_collision_free_small(self) -> None:
        """Verify uniqueness for a small subset (2 seeds × train prompts)."""
        numeracy, spatial = build_all_prompts(DATASET_DIR, "train", num_objects=5, num_settings=2)
        paths: set[Path] = set()
        for seed in range(2):
            for p in numeracy:
                path = image_path(Path("/out"), "numeracy", seed, p.prompt_id)
                assert path not in paths, f"Collision: {path}"
                paths.add(path)
            for p in spatial:
                path = image_path(Path("/out"), "spatial", seed, p.prompt_id)
                assert path not in paths, f"Collision: {path}"
                paths.add(path)
        expected = 2 * (len(numeracy) + len(spatial))
        assert len(paths) == expected


class TestAtomicWrite:
    def test_writes_valid_jpg(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (64, 64), color="red")
        dest = tmp_path / "test.jpg"
        save_image_atomic(img, dest)
        assert dest.exists()
        reopened = Image.open(dest)
        assert reopened.size == (64, 64)

    def test_writes_valid_png(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (64, 64), color="blue")
        dest = tmp_path / "test.png"
        save_image_atomic(img, dest)
        assert dest.exists()
        reopened = Image.open(dest)
        assert reopened.size == (64, 64)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (32, 32), color="green")
        dest = tmp_path / "a" / "b" / "c" / "image.jpg"
        save_image_atomic(img, dest)
        assert dest.exists()

    def test_no_temp_files_left(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (32, 32))
        dest = tmp_path / "clean.jpg"
        save_image_atomic(img, dest)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "clean.jpg"


class TestResumeDetection:
    def test_finds_existing_images(self, tmp_path: Path) -> None:
        """Create fake images and verify find_existing_images detects them."""
        out_dir = tmp_path
        for seed in [0, 1]:
            for pid in [0, 5, 10]:
                dest = image_path(out_dir, "numeracy", seed, pid, "jpg")
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8)).save(dest)

        existing = find_existing_images(out_dir, "numeracy")
        assert len(existing) == 6
        assert (0, 0) in existing
        assert (1, 10) in existing
        assert (0, 99) not in existing

    def test_empty_dir(self, tmp_path: Path) -> None:
        existing = find_existing_images(tmp_path, "numeracy")
        assert len(existing) == 0

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        out_dir = tmp_path
        d = out_dir / "images" / "numeracy" / "seed=000"
        d.mkdir(parents=True)
        (d / "prompt=0000.txt").write_text("not an image")
        (d / "prompt=0001.jpg").write_text("")  # empty but correct extension
        existing = find_existing_images(out_dir, "numeracy")
        assert (0, 1) in existing
        assert (0, 0) not in existing  # .txt ignored
