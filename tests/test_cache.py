# SPDX-License-Identifier: MIT
"""Tests for glossate.core.cache (transcribe-once transcript cache)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glossate.core import cache
from glossate.core.segmenter import Cue


@pytest.fixture()
def enabled_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Undo the autouse cache-disabling fixture for these tests.
    monkeypatch.delenv("GLOSSATE_NO_CACHE", raising=False)
    monkeypatch.setenv("GLOSSATE_CACHE_DIR", str(tmp_path / "cache"))


def _audio(tmp_path: Path, data: bytes = b"audio") -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(data)
    return p


def test_cache_round_trip_preserves_source(enabled_cache: None, tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    cues = [
        Cue(
            start=0.0,
            end=1.0,
            text="merhaba",
            lang="tr",
            source_text="hello",
            source_lang="en",
        )
    ]
    key = cache.transcript_key(audio, asr_model="turbo", asr_backend="auto", source=None)

    assert key is not None
    assert cache.load(key) is None  # miss before save
    cache.save(key, cues)
    assert cache.load(key) == cues  # full dataclass equality incl. source_*


def test_key_changes_when_input_file_changes(enabled_cache: None, tmp_path: Path) -> None:
    audio = _audio(tmp_path, b"one")
    k1 = cache.transcript_key(audio, asr_model="turbo", asr_backend="auto", source=None)
    audio.write_bytes(b"different-content")
    k2 = cache.transcript_key(audio, asr_model="turbo", asr_backend="auto", source=None)
    assert k1 != k2


def test_key_changes_with_asr_settings(enabled_cache: None, tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    base = cache.transcript_key(audio, asr_model="turbo", asr_backend="auto", source=None)
    assert base != cache.transcript_key(audio, asr_model="medium", asr_backend="auto", source=None)
    assert base != cache.transcript_key(audio, asr_model="turbo", asr_backend="auto", source="en")


def test_key_is_none_for_missing_file(enabled_cache: None, tmp_path: Path) -> None:
    assert cache.transcript_key(
        tmp_path / "absent.wav", asr_model="turbo", asr_backend="auto", source=None
    ) is None


def test_cache_enabled_respects_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLOSSATE_NO_CACHE", raising=False)
    assert cache.cache_enabled(True) is True
    assert cache.cache_enabled(False) is False
    monkeypatch.setenv("GLOSSATE_NO_CACHE", "1")
    assert cache.cache_enabled(True) is False
