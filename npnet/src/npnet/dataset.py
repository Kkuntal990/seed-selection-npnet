"""PyTorch Dataset for loading (source_noise, target_noise, prompt) triples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NoiseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Load noise pairs from .npz files collected by data_collection.py."""

    def __init__(
        self,
        noise_pairs_dir: Path,
        prompt_manifest_path: Path,
    ) -> None:
        self.prompts = self._load_manifest(prompt_manifest_path)
        self.samples = self._scan_npz_files(noise_pairs_dir)

    @staticmethod
    def _load_manifest(path: Path) -> dict[tuple[str, int], str]:
        """Load prompt manifest: (category, prompt_id) → prompt text."""
        mapping: dict[tuple[str, int], str] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            mapping[(rec["category"], rec["prompt_id"])] = rec["text"]
        return mapping

    @staticmethod
    def _scan_npz_files(root: Path) -> list[tuple[Path, str]]:
        """Find all .npz files and extract their category from directory structure."""
        samples: list[tuple[Path, str]] = []
        for npz_path in sorted(root.rglob("*.npz")):
            # Path: root/{category}/seed={N}/prompt={M}.npz
            parts = npz_path.relative_to(root).parts
            if len(parts) >= 2:
                category = parts[0]
                samples.append((npz_path, category))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        npz_path, category = self.samples[idx]
        data = np.load(npz_path)
        source = torch.from_numpy(data["source_noise"].squeeze(0))
        target = torch.from_numpy(data["target_noise"].squeeze(0))
        prompt_id = int(data["prompt_id"])
        prompt_text = self.prompts.get((category, prompt_id), "")
        return source, target, prompt_text
