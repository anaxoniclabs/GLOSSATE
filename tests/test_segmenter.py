# SPDX-License-Identifier: MIT
"""Tests for glossate.core.segmenter (and the Cue dataclass)."""

from __future__ import annotations

import dataclasses

import pytest

from glossate import Cue
from glossate.core.segmenter import MAX_CHARS, segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _w(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _result(words: list[dict], lang: str = "en") -> dict:
    return {"language": lang, "segments": [{"words": words}]}


# ---------------------------------------------------------------------------
# Cue dataclass
# ---------------------------------------------------------------------------


def test_cue_is_dataclass() -> None:
    assert dataclasses.is_dataclass(Cue)


def test_cue_importable_from_top_level() -> None:
    from glossate import Cue as TopLevelCue  # noqa: PLC0415

    assert TopLevelCue is Cue


def test_cue_fields() -> None:
    c = Cue(start=0.0, end=1.0, text="hello", lang="en")
    assert c.start == 0.0
    assert c.end == 1.0
    assert c.text == "hello"
    assert c.lang == "en"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_segments_returns_empty_list() -> None:
    assert segment({"language": "en", "segments": []}) == []


# ---------------------------------------------------------------------------
# Short segment passes through unchanged
# ---------------------------------------------------------------------------


def test_short_segment_unchanged() -> None:
    words = [_w("Hello", 0.0, 0.4), _w(" world", 0.4, 1.0)]
    cues = segment(_result(words))

    assert len(cues) == 1
    assert cues[0].text == "Hello world"
    assert cues[0].start == 0.0
    assert cues[0].end == 1.0
    assert cues[0].lang == "en"


def test_exactly_42_chars_unchanged() -> None:
    # 42 'a' chars — must not be split.
    words = [_w("a" * 42, 0.0, 1.0)]
    cues = segment(_result(words))

    assert len(cues) == 1
    assert cues[0].text == "a" * 42


# ---------------------------------------------------------------------------
# Language propagation
# ---------------------------------------------------------------------------


def test_lang_propagated() -> None:
    words = [_w("Merhaba", 0.0, 1.0)]
    cues = segment(_result(words, lang="tr"))

    assert cues[0].lang == "tr"


# ---------------------------------------------------------------------------
# Split at comma (closest punctuation to midpoint)
# ---------------------------------------------------------------------------


def test_comma_split() -> None:
    # "Short text before comma, and some more words after" = 50 chars
    # Comma at position 23; midpoint = 25 → split after word-index 3.
    words = [
        _w("Short", 0.0, 0.5),
        _w(" text", 0.6, 1.0),
        _w(" before", 1.1, 1.8),
        _w(" comma,", 1.9, 4.0),  # long gap: actual end ≠ proportional estimate
        _w(" and", 4.0, 4.2),
        _w(" some", 4.2, 4.5),
        _w(" more", 4.5, 4.7),
        _w(" words", 4.7, 4.9),
        _w(" after", 4.9, 5.0),
    ]
    cues = segment(_result(words))

    assert len(cues) == 2
    assert cues[0].text == "Short text before comma,"
    assert cues[1].text == "and some more words after"


def test_comma_split_uses_actual_word_timestamp_not_proportional() -> None:
    # Comma at char position 23 in a 50-char string → proportional boundary
    # would be 0 + (23/50)*5 = 2.3 s; actual end of " comma," word = 4.0 s.
    words = [
        _w("Short", 0.0, 0.5),
        _w(" text", 0.6, 1.0),
        _w(" before", 1.1, 1.8),
        _w(" comma,", 1.9, 4.0),
        _w(" and", 4.0, 4.2),
        _w(" some", 4.2, 4.5),
        _w(" more", 4.5, 4.7),
        _w(" words", 4.7, 4.9),
        _w(" after", 4.9, 5.0),
    ]
    cues = segment(_result(words))

    proportional = 0.0 + (23 / 50) * 5.0  # 2.3 — wrong estimate
    assert cues[0].end != pytest.approx(proportional), "boundary must not be proportional"
    assert cues[0].end == pytest.approx(4.0)  # actual word end


# ---------------------------------------------------------------------------
# Split at word boundary (no punctuation)
# ---------------------------------------------------------------------------


def test_word_boundary_split_no_punctuation() -> None:
    # "This is a very long line without any punctuation here" = 53 chars
    # Closest word boundary to mid=26 is after " line" (end pos 24, dist=2).
    words = [
        _w("This", 0.0, 0.1),
        _w(" is", 0.1, 0.2),
        _w(" a", 0.2, 0.3),
        _w(" very", 0.3, 0.4),
        _w(" long", 0.4, 0.5),
        _w(" line", 0.5, 1.0),  # actual boundary; proportional at 24/53*10=4.5 s
        _w(" without", 1.0, 1.5),
        _w(" any", 1.5, 1.7),
        _w(" punctuation", 1.7, 1.9),
        _w(" here", 1.9, 2.0),
    ]
    # Give last word end=10.0 to make proportional estimate obviously differ.
    words[-1] = _w(" here", 1.9, 10.0)

    cues = segment(_result(words))

    assert len(cues) == 2
    assert cues[0].text == "This is a very long line"
    assert cues[1].text == "without any punctuation here"


def test_word_boundary_uses_actual_timestamp() -> None:
    words = [
        _w("This", 0.0, 0.1),
        _w(" is", 0.1, 0.2),
        _w(" a", 0.2, 0.3),
        _w(" very", 0.3, 0.4),
        _w(" long", 0.4, 0.5),
        _w(" line", 0.5, 1.0),
        _w(" without", 1.0, 1.5),
        _w(" any", 1.5, 1.7),
        _w(" punctuation", 1.7, 1.9),
        _w(" here", 1.9, 10.0),
    ]
    # Split after " line" (idx 5); actual end = 1.0 s.
    # Proportional: 0 + (24/53)*10 ≈ 4.53 s.
    cues = segment(_result(words))

    proportional = (24 / 53) * 10.0
    assert cues[0].end != pytest.approx(proportional)
    assert cues[0].end == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Single word exceeding 42 chars passes through as-is
# ---------------------------------------------------------------------------


def test_single_oversized_word_passes_as_is() -> None:
    long_word = "a" * (MAX_CHARS + 10)
    words = [_w(long_word, 0.0, 1.0)]
    cues = segment(_result(words))

    assert len(cues) == 1
    assert cues[0].text == long_word
    assert len(cues[0].text) > MAX_CHARS


# ---------------------------------------------------------------------------
# Recursive splitting
# ---------------------------------------------------------------------------


def test_recursive_split_all_cues_within_limit() -> None:
    # Build a long sentence (106 chars) with no punctuation.
    # Both halves after the first split are still > 42 chars → recursion needed.
    words = [
        _w("This", 0.0, 0.3),
        _w(" is", 0.3, 0.5),
        _w(" a", 0.5, 0.6),
        _w(" very", 0.6, 0.9),
        _w(" long", 0.9, 1.2),
        _w(" sentence", 1.2, 1.8),
        _w(" that", 1.8, 2.1),
        _w(" needs", 2.1, 2.5),
        _w(" multiple", 2.5, 3.0),
        _w(" splits", 3.0, 3.4),
        _w(" to", 3.4, 3.6),
        _w(" fit", 3.6, 3.9),
        _w(" within", 3.9, 4.3),
        _w(" the", 4.3, 4.5),
        _w(" subtitle", 4.5, 5.0),
        _w(" character", 5.0, 5.6),
        _w(" limit", 5.6, 6.0),
        _w(" per", 6.0, 6.3),
        _w(" cue", 6.3, 6.5),
    ]
    full = "".join(w["word"] for w in words).strip()
    assert len(full) > MAX_CHARS * 2  # forces at least two levels of recursion

    cues = segment(_result(words))

    assert len(cues) >= 3
    for c in cues:
        assert len(c.text) <= MAX_CHARS, f"Cue too long: {c.text!r}"


# ---------------------------------------------------------------------------
# Trailing punctuation does not produce an empty cue
# ---------------------------------------------------------------------------


def test_trailing_punctuation_not_used_as_split_boundary() -> None:
    # All punctuation is in the final word; segmenter must fall back to word boundary.
    words = [
        _w("This", 0.0, 0.2),
        _w(" is", 0.2, 0.4),
        _w(" a", 0.4, 0.5),
        _w(" fairly", 0.5, 0.8),
        _w(" long", 0.8, 1.1),
        _w(" sentence", 1.1, 1.5),
        _w(" ending.", 1.5, 2.0),
    ]
    cues = segment(_result(words))

    assert len(cues) >= 1
    for c in cues:
        assert c.text  # no empty cues
