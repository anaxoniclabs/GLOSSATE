# SPDX-License-Identifier: MIT
"""Tests for glossate.core.serializer (SRT dump + pure Markdown renderers)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glossate.core.segmenter import Cue
from glossate.core.serializer import (
    _mmss,
    render_dual_stack_md,
    render_prose_md,
    to_srt_string,
    write_srt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_cues() -> list[Cue]:
    return [
        Cue(start=0.0, end=1.5, text="Hello", lang="en"),
        Cue(start=1.5, end=3.0, text="World", lang="en"),
    ]


@pytest.fixture()
def unicode_cues() -> list[Cue]:
    return [
        Cue(start=0.0, end=1.0, text="Merhaba dünya", lang="tr"),
        Cue(start=1.0, end=2.0, text="مرحبا بالعالم", lang="ar"),
    ]


# ---------------------------------------------------------------------------
# SRT format
# ---------------------------------------------------------------------------


def test_empty_srt_is_empty_string() -> None:
    assert to_srt_string([]) == ""


def test_srt_exact_two_cue_output(two_cues: list[Cue]) -> None:
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Hello\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "World\n"
        "\n"
    )
    assert to_srt_string(two_cues) == expected


def test_srt_sequence_starts_at_one(two_cues: list[Cue]) -> None:
    assert to_srt_string(two_cues).splitlines()[0] == "1"


def test_srt_uses_comma_decimal_separator(two_cues: list[Cue]) -> None:
    srt = to_srt_string(two_cues)
    assert "00:00:00,000" in srt
    assert "." not in srt.splitlines()[1]  # timecode line uses comma, not dot


def test_srt_timecode_hours_minutes_seconds() -> None:
    cue = Cue(start=3661.5, end=7323.75, text="t", lang="en")
    assert "01:01:01,500 --> 02:02:03,750" in to_srt_string([cue])


def test_srt_utf8_preserved(unicode_cues: list[Cue]) -> None:
    srt = to_srt_string(unicode_cues)
    assert "Merhaba dünya" in srt
    assert "مرحبا بالعالم" in srt


def test_write_srt_creates_file(tmp_path: Path, two_cues: list[Cue]) -> None:
    out = tmp_path / "sub.srt"
    returned = write_srt(two_cues, out)
    assert returned == out
    assert out.read_text(encoding="utf-8") == to_srt_string(two_cues)


def test_write_srt_utf8_encoding(tmp_path: Path, unicode_cues: list[Cue]) -> None:
    out = tmp_path / "sub.srt"
    write_srt(unicode_cues, out)
    assert "مرحبا بالعالم".encode("utf-8") in out.read_bytes()


# ---------------------------------------------------------------------------
# Short timestamp helper
# ---------------------------------------------------------------------------


def test_mmss_under_an_hour() -> None:
    assert _mmss(0.0) == "0:00"
    assert _mmss(75.0) == "1:15"


def test_mmss_past_an_hour_adds_hours() -> None:
    assert _mmss(3661.0) == "1:01:01"


# ---------------------------------------------------------------------------
# Markdown: prose
# ---------------------------------------------------------------------------


def test_prose_md_with_timestamps_marks_each_block() -> None:
    md = render_prose_md([(0.0, "First para."), (75.0, "Second para.")], timestamps=True)
    assert "**[0:00]**" in md
    assert "**[1:15]**" in md
    assert "First para." in md and "Second para." in md


def test_prose_md_without_timestamps_omits_markers() -> None:
    md = render_prose_md([(0.0, "Body text.")], timestamps=False)
    assert "[0:00]" not in md
    assert "Body text." in md


def test_prose_md_heading_renders_section_title() -> None:
    md = render_prose_md([(0.0, "x")], timestamps=False, heading="Original (English)")
    assert md.startswith("## Original (English)")


def test_prose_md_skips_empty_blocks() -> None:
    md = render_prose_md([(0.0, "  "), (1.0, "kept")], timestamps=True)
    # Only the non-empty block produces a timestamp marker.
    assert md.count("**[") == 1
    assert "kept" in md


# ---------------------------------------------------------------------------
# Markdown: dual-stack (sentence pairs)
# ---------------------------------------------------------------------------


def test_dual_stack_pairs_source_then_translation_quote() -> None:
    md = render_dual_stack_md([(0.0, "Bonjour.", "Hello.")], timestamps=False)
    assert "Bonjour." in md
    assert "> Hello." in md  # translation rendered as a blockquote


def test_dual_stack_timestamp_prefixes_source_line() -> None:
    md = render_dual_stack_md([(75.0, "Bonjour.", "Hello.")], timestamps=True)
    assert "**[1:15]** Bonjour." in md


def test_dual_stack_unicode_preserved() -> None:
    md = render_dual_stack_md([(0.0, "مرحبا", "Merhaba")], timestamps=False)
    assert "مرحبا" in md
    assert "> Merhaba" in md
