"""File I/O utilities: atomic writes, metadata, resume scanning."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from seed_mining.prompts import Prompt

logger = logging.getLogger("seed_mining.io")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def image_path(out_dir: Path, category: str, seed: int, prompt_id: int, ext: str = "jpg") -> Path:
    """Canonical image path: ``images/{category}/seed={seed:03d}/prompt={prompt_id:04d}.{ext}``."""
    return out_dir / "images" / category / f"seed={seed:03d}" / f"prompt={prompt_id:04d}.{ext}"


# ---------------------------------------------------------------------------
# Atomic image writes
# ---------------------------------------------------------------------------


def save_image_atomic(img: Image.Image, dest: Path, *, quality: int = 95) -> None:
    """Write *img* to *dest* atomically (temp file + rename).

    The temp file is created in the same directory as *dest* so that
    ``os.rename`` is guaranteed to be atomic on POSIX.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = "JPEG" if dest.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    fd, tmp_path = tempfile.mkstemp(suffix=f".tmp{dest.suffix}", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            if fmt == "JPEG":
                img.save(f, format=fmt, quality=quality)
            else:
                img.save(f, format=fmt)
        os.rename(tmp_path, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Metadata JSONL
# ---------------------------------------------------------------------------


def append_metadata_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append *records* to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def load_existing_keys(path: Path) -> set[tuple[int, int]]:
    """Load ``(seed, prompt_id)`` pairs from an existing metadata JSONL file."""
    keys: set[tuple[int, int]] = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            keys.add((rec["seed"], rec["prompt_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return keys


# ---------------------------------------------------------------------------
# Resume scanning
# ---------------------------------------------------------------------------


def find_existing_images(out_dir: Path, category: str) -> set[tuple[int, int]]:
    """Scan output dir and return ``(seed, prompt_id)`` pairs that already exist."""
    existing: set[tuple[int, int]] = set()
    images_dir = out_dir / "images" / category
    if not images_dir.exists():
        return existing
    for img_file in images_dir.rglob("*"):
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            seed_part = img_file.parent.name  # "seed=042"
            prompt_part = img_file.stem  # "prompt=0301"
            seed = int(seed_part.split("=", 1)[1])
            pid = int(prompt_part.split("=", 1)[1])
            existing.add((seed, pid))
        except (IndexError, ValueError):
            continue
    return existing


# ---------------------------------------------------------------------------
# Config / prompts persistence
# ---------------------------------------------------------------------------


def save_prompts_jsonl(out_dir: Path, prompts: list[Prompt], category: str) -> None:
    """Write prompts manifest to ``prompts/{category}_prompts.jsonl``."""
    path = out_dir / "prompts" / f"{category}_prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            rec = {
                "prompt_id": p.prompt_id,
                "category": p.category,
                "text": p.text,
                **p.metadata,
            }
            f.write(json.dumps(rec, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Completion marker
# ---------------------------------------------------------------------------


def write_done_marker(out_dir: Path) -> None:
    """Write ``_DONE.txt`` with a UTC timestamp."""
    (out_dir / "_DONE.txt").write_text(
        f"Completed at {datetime.now(UTC).isoformat()}\n"
    )


def verify_completeness(
    out_dir: Path,
    seeds: list[int],
    numeracy: list[Prompt],
    spatial: list[Prompt],
    ext: str = "jpg",
) -> dict[str, Any]:
    """Verify that all expected images exist. Return summary dict."""
    missing_numeracy: list[tuple[int, int]] = []
    missing_spatial: list[tuple[int, int]] = []

    for seed in seeds:
        for p in numeracy:
            if not image_path(out_dir, "numeracy", seed, p.prompt_id, ext).exists():
                missing_numeracy.append((seed, p.prompt_id))
        for p in spatial:
            if not image_path(out_dir, "spatial", seed, p.prompt_id, ext).exists():
                missing_spatial.append((seed, p.prompt_id))

    expected_numeracy = len(seeds) * len(numeracy)
    expected_spatial = len(seeds) * len(spatial)
    actual_numeracy = expected_numeracy - len(missing_numeracy)
    actual_spatial = expected_spatial - len(missing_spatial)

    summary = {
        "expected_numeracy": expected_numeracy,
        "actual_numeracy": actual_numeracy,
        "missing_numeracy": len(missing_numeracy),
        "expected_spatial": expected_spatial,
        "actual_spatial": actual_spatial,
        "missing_spatial": len(missing_spatial),
        "total_expected": expected_numeracy + expected_spatial,
        "total_actual": actual_numeracy + actual_spatial,
    }

    if missing_numeracy:
        logger.warning("Missing %d numeracy images", len(missing_numeracy))
        for seed, pid in missing_numeracy[:10]:
            logger.warning("  missing: seed=%d prompt_id=%d", seed, pid)
        if len(missing_numeracy) > 10:
            logger.warning("  ... and %d more", len(missing_numeracy) - 10)

    if missing_spatial:
        logger.warning("Missing %d spatial images", len(missing_spatial))
        for seed, pid in missing_spatial[:10]:
            logger.warning("  missing: seed=%d prompt_id=%d", seed, pid)
        if len(missing_spatial) > 10:
            logger.warning("  ... and %d more", len(missing_spatial) - 10)

    return summary
