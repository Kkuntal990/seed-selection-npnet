"""Deterministic prompt generation from Comp90 dataset files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NB_TO_WORD: dict[int, str] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
COUNTS: list[int] = [2, 3, 4, 5, 6]


@dataclass(frozen=True)
class Prompt:
    """A single generation prompt with metadata."""

    prompt_id: int
    category: str  # "numeracy" | "spatial"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def load_objects(path: Path) -> list[tuple[str, str, str]]:
    """Load objects from a Comp90 objects file.

    Each line has format: ``singular, article singular, plural``
    Returns list of ``(singular, article_form, plural)`` tuples.
    """
    objects: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected 3 comma-separated values, got {len(parts)}: {line!r}")
        objects.append((parts[0], parts[1], parts[2]))
    return objects


def load_backgrounds(path: Path) -> list[str]:
    """Load background settings, one per line."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_spatial_prompts(path: Path) -> list[str]:
    """Load pre-constructed spatial prompts, one per line."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_numeracy_prompts(
    objects: list[tuple[str, str, str]],
    backgrounds: list[str],
    *,
    num_objects: int | None = None,
    num_settings: int | None = None,
) -> list[Prompt]:
    """Build numeracy prompts: count × object × background.

    Template: ``"{count_word} {plural}, {background}"``
    Order: count ascending → object list order → background list order.
    """
    objs = objects[:num_objects] if num_objects is not None else objects
    bgs = backgrounds[:num_settings] if num_settings is not None else backgrounds

    prompts: list[Prompt] = []
    pid = 0
    for count in COUNTS:
        word = NB_TO_WORD[count]
        for singular, _article, plural in objs:
            for bg in bgs:
                text = f"{word} {plural}, {bg}"
                prompts.append(
                    Prompt(
                        prompt_id=pid,
                        category="numeracy",
                        text=text,
                        metadata={
                            "count_target": count,
                            "count_word": word,
                            "object": singular,
                            "object_plural": plural,
                            "background": bg,
                        },
                    )
                )
                pid += 1
    return prompts


def build_spatial_prompts(
    spatial_lines: list[str],
    backgrounds: list[str],
    *,
    append_background: bool = True,
    num_settings: int | None = None,
) -> list[Prompt]:
    """Build spatial prompts from pre-constructed lines.

    If *append_background* is True, each base prompt is combined with
    each background: ``"{prompt}, {background}"``.
    """
    bgs = backgrounds[:num_settings] if num_settings is not None else backgrounds

    prompts: list[Prompt] = []
    pid = 0

    if append_background:
        for base_prompt in spatial_lines:
            for bg in bgs:
                text = f"{base_prompt}, {bg}"
                prompts.append(
                    Prompt(
                        prompt_id=pid,
                        category="spatial",
                        text=text,
                        metadata={
                            "spatial_prompt_raw": base_prompt,
                            "background": bg,
                        },
                    )
                )
                pid += 1
    else:
        for base_prompt in spatial_lines:
            prompts.append(
                Prompt(
                    prompt_id=pid,
                    category="spatial",
                    text=base_prompt,
                    metadata={"spatial_prompt_raw": base_prompt},
                )
            )
            pid += 1

    return prompts


def build_all_prompts(
    prompt_dataset_dir: Path,
    split: str = "all",
    *,
    num_objects: int | None = None,
    num_settings: int | None = None,
    append_background_to_spatial: bool = True,
) -> tuple[list[Prompt], list[Prompt]]:
    """Build numeracy and spatial prompt lists from dataset files.

    Parameters
    ----------
    prompt_dataset_dir:
        Directory containing the Comp90 text files.
    split:
        ``"train"``, ``"eval"``, or ``"all"`` (concatenate train + eval).
    num_objects:
        Limit number of objects per split (``None`` = all).
    num_settings:
        Limit number of backgrounds per split (``None`` = all).
    append_background_to_spatial:
        Whether to append background settings to spatial prompts.

    Returns
    -------
    tuple of (numeracy_prompts, spatial_prompts)
    """
    d = prompt_dataset_dir

    # Collect objects and backgrounds based on split
    objects: list[tuple[str, str, str]] = []
    backgrounds: list[str] = []
    spatial_lines: list[str] = []

    if split in ("train", "all"):
        train_objects = load_objects(d / "objects_train.txt")
        train_bgs = load_backgrounds(d / "backgrounds_train.txt")
        train_spatial = load_spatial_prompts(d / "spatial_prompts_train.txt")

        if split == "train":
            objects = train_objects
            backgrounds = train_bgs
            spatial_lines = train_spatial
        else:
            objects.extend(train_objects)
            backgrounds.extend(train_bgs)
            spatial_lines.extend(train_spatial)

    if split in ("eval", "all"):
        eval_objects = load_objects(d / "objects_eval.txt")
        eval_bgs = load_backgrounds(d / "backgrounds_eval.txt")
        eval_spatial = load_spatial_prompts(d / "spatial_prompts_eval.txt")

        if split == "eval":
            objects = eval_objects
            backgrounds = eval_bgs
            spatial_lines = eval_spatial
        else:
            objects.extend(eval_objects)
            backgrounds.extend(eval_bgs)
            spatial_lines.extend(eval_spatial)

    numeracy = build_numeracy_prompts(
        objects, backgrounds, num_objects=num_objects, num_settings=num_settings
    )
    spatial = build_spatial_prompts(
        spatial_lines,
        backgrounds,
        append_background=append_background_to_spatial,
        num_settings=num_settings,
    )

    return numeracy, spatial
