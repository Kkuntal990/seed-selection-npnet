"""Tests for NoiseDataset: loading, manifest parsing, item retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from npnet.dataset import NoiseDataset


def _create_test_data(root: Path, manifest_path: Path) -> int:
    """Create a small set of test .npz files and a manifest.

    Returns the number of samples created.
    """
    records = []
    count = 0
    for category in ("numeracy", "spatial"):
        for seed in (0, 1):
            for prompt_id in (0, 1):
                npz_dir = root / category / f"seed={seed:03d}"
                npz_dir.mkdir(parents=True, exist_ok=True)
                npz_path = npz_dir / f"prompt={prompt_id:04d}.npz"

                source = np.random.randn(1, 4, 128, 128).astype(np.float32)
                target = np.random.randn(1, 4, 128, 128).astype(np.float32)
                np.savez(npz_path, source_noise=source, target_noise=target, prompt_id=prompt_id)

                records.append(
                    {
                        "category": category,
                        "prompt_id": prompt_id,
                        "text": f"Test prompt {category} {prompt_id}",
                    }
                )
                count += 1

    # Deduplicate manifest (same prompt_id can appear for different seeds)
    seen = set()
    with manifest_path.open("w") as f:
        for rec in records:
            key = (rec["category"], rec["prompt_id"])
            if key not in seen:
                f.write(json.dumps(rec) + "\n")
                seen.add(key)

    return count


class TestNoiseDataset:
    def test_len(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        n = _create_test_data(tmp_path / "pairs", manifest)
        ds = NoiseDataset(tmp_path / "pairs", manifest)
        assert len(ds) == n

    def test_getitem_types(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _create_test_data(tmp_path / "pairs", manifest)
        ds = NoiseDataset(tmp_path / "pairs", manifest)
        source, target, text = ds[0]
        assert isinstance(source, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(text, str)

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _create_test_data(tmp_path / "pairs", manifest)
        ds = NoiseDataset(tmp_path / "pairs", manifest)
        source, target, text = ds[0]
        assert source.shape == (4, 128, 128)
        assert target.shape == (4, 128, 128)

    def test_prompt_text_lookup(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _create_test_data(tmp_path / "pairs", manifest)
        ds = NoiseDataset(tmp_path / "pairs", manifest)
        # All items should have non-empty prompt text
        for i in range(len(ds)):
            _, _, text = ds[i]
            assert len(text) > 0, f"Empty prompt text at index {i}"

    def test_manifest_parsing(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        _create_test_data(tmp_path / "pairs", manifest)
        ds = NoiseDataset(tmp_path / "pairs", manifest)
        # Should have entries for both categories
        categories = {cat for (cat, _) in ds.prompts}
        assert "numeracy" in categories
        assert "spatial" in categories

    def test_empty_dataset(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("")
        pairs_dir = tmp_path / "pairs"
        pairs_dir.mkdir()
        ds = NoiseDataset(pairs_dir, manifest)
        assert len(ds) == 0
