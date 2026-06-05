# SPDX-License-Identifier: MIT
"""On-disk transcript cache: transcribe once, reuse for translate/format/burn.

ASR is the expensive pipeline stage. Caching the segmented source cues keyed on
(input file identity + ASR settings) lets re-runs — a different ``--target``, a
different output format, a burn pass — skip transcription entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from glossate.core.segmenter import Cue

_log = logging.getLogger("glossate.cache")
_SCHEMA = 1


def cache_enabled(flag: bool = True) -> bool:
    """Caching is on unless the caller disables it or GLOSSATE_NO_CACHE is set."""
    if not flag:
        return False
    return os.environ.get("GLOSSATE_NO_CACHE", "").lower() not in {"1", "true", "yes"}


def cache_dir() -> Path:
    """Transcript cache directory (override with GLOSSATE_CACHE_DIR)."""
    override = os.environ.get("GLOSSATE_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "glossate" / "transcripts"
    base.mkdir(parents=True, exist_ok=True)
    return base


def transcript_key(
    input_path: Path | str,
    *,
    asr_model: str,
    asr_backend: str,
    source: str | None,
) -> str | None:
    """Stable cache key for a transcription, or None if the input can't be stat'd.

    The key includes the file size and mtime, so editing the input invalidates it.
    """
    path = Path(input_path)
    try:
        st = path.stat()
    except OSError:
        return None
    raw = "|".join(
        [
            str(_SCHEMA),
            str(path.resolve()),
            str(st.st_size),
            str(st.st_mtime_ns),
            asr_model,
            asr_backend,
            source or "auto",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def load(key: str | None) -> list[Cue] | None:
    """Return cached cues for *key*, or None on miss / unreadable / stale schema."""
    if not key:
        return None
    path = cache_dir() / f"{key}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("schema") != _SCHEMA:
        return None
    try:
        cues = [_cue_from_dict(d) for d in data["cues"]]
    except (KeyError, TypeError):
        return None
    _log.info("transcript cache hit: %s (%d cues)", key, len(cues))
    return cues


def save(key: str | None, cues: list[Cue]) -> None:
    """Best-effort write of *cues* under *key*; never raises."""
    if not key:
        return
    payload = {"schema": _SCHEMA, "cues": [_cue_to_dict(c) for c in cues]}
    try:
        (cache_dir() / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        _log.info("transcript cached: %s (%d cues)", key, len(cues))
    except OSError:
        pass


def _cue_to_dict(c: Cue) -> dict:
    return {
        "start": c.start,
        "end": c.end,
        "text": c.text,
        "lang": c.lang,
        "source_text": c.source_text,
        "source_lang": c.source_lang,
    }


def _cue_from_dict(d: dict) -> Cue:
    return Cue(
        start=d["start"],
        end=d["end"],
        text=d["text"],
        lang=d["lang"],
        source_text=d.get("source_text"),
        source_lang=d.get("source_lang"),
    )
