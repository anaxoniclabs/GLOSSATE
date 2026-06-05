# SPDX-License-Identifier: MIT
"""Convert raw Whisper output into normalized Cue objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_CHARS = 42
_PUNCTUATION = frozenset(",.;:!?")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    lang: str
    # Original transcription preserved across translation (None until translated).
    source_text: str | None = None
    source_lang: str | None = None


def segment(result: dict[str, Any]) -> list[Cue]:
    """Return readable subtitle cues from Whisper segments.

    Whisper segments can be too long for subtitle display and translation
    prompts. When word timestamps are available, split at punctuation or word
    boundaries while preserving the actual word timing.
    """
    lang: str = result.get("language", "")
    cues: list[Cue] = []
    for seg in result.get("segments", []):
        words = seg.get("words") or []
        if words:
            cues.extend(_split_words(words, lang))
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        cues.append(Cue(start=seg["start"], end=seg["end"], text=text, lang=lang))
    return cues


def _make_cue(words: list[dict[str, Any]], lang: str) -> Cue:
    return Cue(
        start=float(words[0]["start"]),
        end=float(words[-1]["end"]),
        text=" ".join(w["text"] for w in words),
        lang=lang,
    )


def _split_words(words: list[dict[str, Any]], lang: str) -> list[Cue]:
    normalized = [
        _normalize_word(w)
        for w in words
        if str(w.get("word", w.get("text", ""))).strip()
    ]
    if not normalized:
        return []

    text = " ".join(word["text"] for word in normalized)
    if len(text) <= MAX_CHARS or len(normalized) == 1:
        return [_make_cue(normalized, lang)]

    split_at = _choose_split_index(normalized)
    if split_at is None:
        return [_make_cue(normalized, lang)]

    return _split_words(normalized[: split_at + 1], lang) + _split_words(
        normalized[split_at + 1 :], lang
    )


def _normalize_word(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("word", word.get("text", ""))).strip(),
        "start": word["start"],
        "end": word["end"],
    }


def _word_end_positions(words: list[dict[str, Any]]) -> list[int]:
    positions: list[int] = []
    pos = 0
    for i, word in enumerate(words):
        if i > 0:
            pos += 1
        pos += len(str(word["text"]))
        positions.append(pos)
    return positions


def _choose_split_index(words: list[dict[str, Any]]) -> int | None:
    if len(words) < 2:
        return None

    positions = _word_end_positions(words)
    midpoint = positions[-1] / 2
    usable = range(0, len(words) - 1)

    punctuation = [
        i
        for i in usable
        if str(words[i]["text"]).rstrip()[-1:] in _PUNCTUATION
    ]
    candidates = punctuation or list(usable)
    return min(candidates, key=lambda i: abs(positions[i] - midpoint))
