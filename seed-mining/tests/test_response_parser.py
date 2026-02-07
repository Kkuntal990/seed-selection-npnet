"""Tests for VLM response parsing."""

from __future__ import annotations

from seed_mining.response_parser import extract_spatial_answer, get_number_from_response


class TestGetNumberFromResponse:
    def test_digit(self) -> None:
        assert get_number_from_response("There are 3 apples in the image.") == 3

    def test_word(self) -> None:
        assert get_number_from_response("There are three apples in the image.") == 3

    def test_zero_word(self) -> None:
        assert get_number_from_response("There are no apples in the image.") == 0

    def test_zero_digit(self) -> None:
        assert get_number_from_response("I see 0 apples.") == 0

    def test_large_digit(self) -> None:
        assert get_number_from_response("I count 12 items.") == 12

    def test_digit_takes_precedence(self) -> None:
        # Digits are searched first
        assert get_number_from_response("There are 5 items, five total.") == 5

    def test_no_number(self) -> None:
        assert get_number_from_response("I see some apples.") is None

    def test_empty_string(self) -> None:
        assert get_number_from_response("") is None

    def test_case_insensitive(self) -> None:
        assert get_number_from_response("There are Three apples.") == 3

    def test_one(self) -> None:
        assert get_number_from_response("There is one apple.") == 1

    def test_six(self) -> None:
        assert get_number_from_response("I see six dogs in this image.") == 6


class TestExtractSpatialAnswer:
    def test_yes(self) -> None:
        assert extract_spatial_answer("Yes, the pineapple is on the left.") is True

    def test_no(self) -> None:
        assert extract_spatial_answer("No, the monkey is on the right.") is False

    def test_yes_lowercase(self) -> None:
        assert extract_spatial_answer("yes") is True

    def test_unclear(self) -> None:
        assert extract_spatial_answer("I cannot determine the position.") is None

    def test_empty(self) -> None:
        assert extract_spatial_answer("") is None
