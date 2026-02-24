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

NUMBER_TO_WORD: dict[int, str] = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

# Patterns indicating the model thinks there are too many to count
_TOO_MANY_PATTERNS = re.compile(
    r"too many to count|too many|numerous|countless|several dozen|"
    r"a lot of|many|a large number|can'?t count|impossible to count|"
    r"hard to count|difficult to count",
    re.IGNORECASE,
)

NUMEROUS_THRESHOLD = 19


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


def get_numeracy_label(response: str) -> str:
    """Extract a count label from a VLM response, matching the paper methodology.

    Returns one of: "zero", "one", ..., "ten", or "numerous".

    Paper rules:
    - Responses like "there are too many to count" → "numerous"
    - Predicted quantities > 19 → "numerous"
    - Otherwise map to the corresponding number word
    """
    lower = response.lower()

    # Check for "too many" style responses before digit extraction,
    # but only if no specific digit is found (digit takes priority)
    digit_match = re.search(r"\d+", response)
    if digit_match:
        n = int(digit_match.group())
        if n > NUMEROUS_THRESHOLD:
            return "numerous"
        return NUMBER_TO_WORD.get(n, str(n))

    # No digit found — check for "too many" phrasing
    if _TOO_MANY_PATTERNS.search(lower):
        return "numerous"

    # Fall back to number words
    for word, num in WORD_TO_NUMBER.items():
        if word in lower:
            if num > NUMEROUS_THRESHOLD:
                return "numerous"
            return NUMBER_TO_WORD.get(num, str(num))

    return "numerous"  # Unparseable defaults to "numerous"


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
