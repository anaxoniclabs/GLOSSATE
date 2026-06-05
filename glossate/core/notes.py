# SPDX-License-Identifier: MIT
"""Build readable Markdown "notes" from transcript cues.

This module owns the *orchestration* the deterministic serializer cannot: it
chunks cues into time-windows, sends each window through Gemma to be reflowed
into clean prose, and (for the bilingual dual-stack layout) segments cues into
sentences and translates them at sentence granularity. Timestamps are always
computed here in code from cue start times — never produced by the model.

Layouts (Level 2 = a target language is given):

* monolingual prose .... Level 1, or Level 2 ``scope="translated"``
* two-prose ............. Level 2 ``scope="both"``: original section + translation section
* dual-stack ........... Level 2 ``scope="both"``: sentence-by-sentence pairs
"""

from __future__ import annotations

from typing import Callable, Optional

from glossate.core import serializer as _serializer
from glossate.core import translator as _translator
from glossate.core.segmenter import Cue

# Keep each window's *output* comfortably under the format pass's generation cap.
_CHUNK_MAX_CHARS = 1500
# Sentence terminators (Latin + CJK + Arabic).
_ENDERS = frozenset(".!?…。！？؟")

ProgressFn = Optional[Callable[[int, int], None]]


def build_markdown(
    cues: list[Cue],
    *,
    source_lang: str,
    target_lang: str | None,
    scope: str = "translated",
    layout: str = "two-prose",
    timestamps: bool = True,
    model_state: _translator.TranslationModelState,
    on_progress: ProgressFn = None,
) -> str:
    """Render *source* cues to a Markdown string.

    *cues* are the untranslated transcription. Any translation needed for the
    requested layout is performed here, so the caller only transcribes.
    """
    if not cues:
        return ""

    # Level 1: monolingual original-language notes.
    if target_lang is None:
        blocks = _prose_blocks(
            cues, key=_source_text, lang=source_lang, state=model_state, on_progress=on_progress
        )
        return _serializer.render_prose_md(blocks, timestamps=timestamps)

    # Level 2, translated-only: monolingual target-language notes.
    if scope != "both":
        translated = _translator.translate(
            cues, target_lang, source=source_lang, model_state=model_state
        )
        blocks = _prose_blocks(
            translated, key=_cue_text, lang=target_lang, state=model_state, on_progress=on_progress
        )
        return _serializer.render_prose_md(blocks, timestamps=timestamps)

    # Level 2, both languages.
    if layout == "dual-stack":
        return _dual_stack(cues, source_lang, target_lang, timestamps, model_state, on_progress)

    # two-prose: original section, then translation section.
    translated = _translator.translate(
        cues, target_lang, source=source_lang, model_state=model_state
    )
    orig = _prose_blocks(cues, key=_cue_text, lang=source_lang, state=model_state)
    trans = _prose_blocks(translated, key=_cue_text, lang=target_lang, state=model_state)
    src_name = _translator._lang_name(source_lang)
    tgt_name = _translator._lang_name(target_lang)
    return (
        _serializer.render_prose_md(orig, timestamps=timestamps, heading=f"Original ({src_name})")
        + "\n"
        + _serializer.render_prose_md(
            trans, timestamps=timestamps, heading=f"Translation ({tgt_name})"
        )
    )


# ---------------------------------------------------------------------------
# Prose path: chunk into time-windows, format each with Gemma
# ---------------------------------------------------------------------------


def _prose_blocks(
    cues: list[Cue],
    *,
    key: Callable[[Cue], str],
    lang: str,
    state: _translator.TranslationModelState,
    on_progress: ProgressFn = None,
) -> list[tuple[float, str]]:
    windows = _chunk(cues, key)
    blocks: list[tuple[float, str]] = []
    for i, (start, text) in enumerate(windows, 1):
        prose = _translator.format_prose(text, lang, model_state=state)
        blocks.append((start, prose))
        if on_progress:
            on_progress(i, len(windows))
    return blocks


def _chunk(
    cues: list[Cue], key: Callable[[Cue], str], max_chars: int = _CHUNK_MAX_CHARS
) -> list[tuple[float, str]]:
    """Group cues into (start_seconds, joined_text) windows under *max_chars*."""
    windows: list[tuple[float, str]] = []
    buf: list[str] = []
    chars = 0
    start = 0.0
    for cue in cues:
        t = (key(cue) or "").strip()
        if not t:
            continue
        if buf and chars + len(t) > max_chars:
            windows.append((start, " ".join(buf)))
            buf, chars = [], 0
        if not buf:
            start = cue.start
        buf.append(t)
        chars += len(t)
    if buf:
        windows.append((start, " ".join(buf)))
    return windows


# ---------------------------------------------------------------------------
# Dual-stack path: segment into sentences, translate sentence-by-sentence
# ---------------------------------------------------------------------------


def _dual_stack(
    cues: list[Cue],
    source_lang: str,
    target_lang: str,
    timestamps: bool,
    state: _translator.TranslationModelState,
    on_progress: ProgressFn,
) -> str:
    sentences = _sentences(cues, source_lang)
    translated = _translator.translate(
        sentences, target_lang, source=source_lang, model_state=state, on_progress=on_progress
    )
    pairs = [(c.start, c.source_text or "", c.text) for c in translated]
    return _serializer.render_dual_stack_md(pairs, timestamps=timestamps)


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split *text* into complete sentences plus a trailing incomplete remainder.

    A boundary is a terminator followed by whitespace; a terminator at the very
    end of *text* is left in the remainder (it may continue in the next cue).
    Guards against the common false positives a naive split hits: decimals
    (``3.5``) and intra-token periods (``U.S.``). It does **not** know prose
    abbreviations like "Dr." — a terminator + space + word is treated as a break.
    """
    sentences: list[str] = []
    last = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in _ENDERS:
            i += 1
            continue
        # Decimal like 3.5 — the dot is not a sentence end.
        if text[i] == "." and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in _ENDERS:  # consume runs like "?!" or "..."
            j += 1
        if j >= n:  # terminator at end of buffer → incomplete, defer to next cue
            break
        if text[j].isspace():
            sentences.append(text[last:j].strip())
            last = j
            i = j
            continue
        i = j  # terminator mid-token (e.g. U.S.) — keep scanning
    return sentences, text[last:].strip()


def _sentences(cues: list[Cue], source_lang: str) -> list[Cue]:
    """Merge fragment cues into sentence-level cues, splitting on sentence enders.

    Cues are glued in order, then complete sentences are peeled off the buffer —
    so a sentence can span several cues, and a cue holding two sentences is split.
    Each sentence's start is the cue where it began; sentences peeled later from
    the same buffer inherit the current cue's start (sparse, orientation-only).
    """
    sentences: list[Cue] = []
    buf = ""
    start: float | None = None
    end = 0.0
    for cue in cues:
        piece = _source_text(cue).strip()
        if not piece:
            continue
        if not buf:
            start = cue.start
        buf = f"{buf} {piece}".strip() if buf else piece
        end = cue.end
        complete, buf = _split_sentences(buf)
        for sentence in complete:
            if sentence:
                sentences.append(Cue(start=start or 0.0, end=end, text=sentence, lang=source_lang))
            start = cue.start  # remainder, if any, begins around this cue
        if not buf:
            start = None
    if buf:
        sentences.append(Cue(start=start or 0.0, end=end, text=buf, lang=source_lang))
    return sentences


# ---------------------------------------------------------------------------
# Cue text accessors
# ---------------------------------------------------------------------------


def _cue_text(cue: Cue) -> str:
    return cue.text


def _source_text(cue: Cue) -> str:
    """The original transcription text (source_text once translated, else text)."""
    return cue.source_text or cue.text
