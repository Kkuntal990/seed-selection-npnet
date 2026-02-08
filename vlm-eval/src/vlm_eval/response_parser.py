"""Parse VLM text responses into structured answers."""

from __future__ import annotations

import re

WORD_TO_NUMBER: dict[str, int] = {
    "zero": 0,
    "no": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def get_number_from_response(response: str) -> int | None:
    """Extract a count from a VLM response string.

    Matches the official repo logic: search for digits first, then number words.
    Returns ``None`` if no number is found.
    """
    # First: look for digit sequences
    match = re.search(r"\d+", response)
    if match:
        return int(match.group())

    # Second: look for number words (case-insensitive)
    lower = response.lower()
    for word, num in WORD_TO_NUMBER.items():
        if word in lower:
            return num

    return None


def extract_spatial_answer(response: str) -> bool | None:
    """Extract yes/no answer from a spatial VLM response.

    Returns ``True`` for yes, ``False`` for no, ``None`` if unclear.
    """
    lower = response.lower().strip()
    if re.search(r"\byes\b", lower):
        return True
    if re.search(r"\bno\b", lower):
        return False
    return None
