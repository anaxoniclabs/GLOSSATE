# SPDX-License-Identifier: MIT
"""Serialize Cue lists to SRT, and render prepared notes data to Markdown.

SRT is a deterministic dump of timed cues. The Markdown renderers here are
*pure*: they take already-prepared text (prose blocks or sentence pairs) plus
flags and emit a string. The LLM "notes" formatting and any translation happen
upstream in :mod:`glossate.core.notes`; this module never touches a model.
"""

from __future__ import annotations

from pathlib import Path

from glossate.core.segmenter import Cue

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ts(seconds: float, sep: str) -> str:
    """Format a timestamp in HH:MM:SS{sep}mmm (SRT/VTT style)."""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _mmss(seconds: float) -> str:
    """Format a short human timestamp: M:SS, or H:MM:SS past an hour."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def to_srt_string(cues: list[Cue]) -> str:
    parts: list[str] = []
    for i, cue in enumerate(cues, 1):
        parts.append(f"{i}\n{_ts(cue.start, ',')} --> {_ts(cue.end, ',')}\n{cue.text}\n\n")
    return "".join(parts)


def write_srt(cues: list[Cue], path: str | Path) -> Path:
    p = Path(path)
    p.write_text(to_srt_string(cues), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Markdown renderers (pure: prepared data in, string out)
# ---------------------------------------------------------------------------


def render_prose_md(
    blocks: list[tuple[float, str]],
    *,
    timestamps: bool,
    heading: str | None = None,
) -> str:
    """Render flowing prose. *blocks* is a list of (start_seconds, prose_text).

    With *timestamps* on, each block is prefixed by a sparse ``**[M:SS]**``
    marker (one per time-window, never per cue). *heading* adds a ``##`` section
    title (used by the two-prose bilingual layout).
    """
    parts: list[str] = []
    if heading:
        parts.append(f"## {heading}\n")
    for start, prose in blocks:
        text = prose.strip()
        if not text:
            continue
        if timestamps:
            parts.append(f"**[{_mmss(start)}]**\n\n{text}\n")
        else:
            parts.append(f"{text}\n")
    return "\n".join(parts).rstrip() + "\n"


def render_dual_stack_md(
    pairs: list[tuple[float, str, str]],
    *,
    timestamps: bool,
) -> str:
    """Render sentence-by-sentence pairs. *pairs* is (start_seconds, source, target).

    Each pair is the source sentence followed by its translation as a blockquote,
    so they read as aligned study cards. With *timestamps* on, a ``**[M:SS]**``
    marker prefixes the source line.
    """
    parts: list[str] = []
    for start, src, tgt in pairs:
        src = src.strip()
        tgt = tgt.strip()
        if not src and not tgt:
            continue
        prefix = f"**[{_mmss(start)}]** " if timestamps else ""
        parts.append(f"{prefix}{src}\n\n> {tgt}\n")
    return "\n".join(parts).rstrip() + "\n"
