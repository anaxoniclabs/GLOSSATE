---
number: 012
title: Public API + Session + exception translation
status: todo
type: AFK
blocked_by: ["003-serializer.md", "004-output-paths.md", "007-ffmpeg-audio-extraction.md", "008-model-download.md", "010-transcriber.md", "011-translator.md"]
---

# 012 — Public API + Session + exception translation

## Parent

`docs/glossate-prd.md`

## What to build

Implement `api.py` — the public surface of GLOSSATE. This is where all the internal modules are composed into a coherent API, and where internal exceptions are translated to public ones.

**Public exception hierarchy** (all subclass `GlossateError`):
- `GlossateError` — base
- `ModelNotInstalledError` — model weights not cached
- `AudioExtractionError` — ffmpeg failed
- `TranscriptionError` — Whisper failed
- `TranslationError` — mlx-lm failed
- `UnsupportedLanguageError` — language pair not covered by the model

**`Session` context manager**:
- `__init__(asr_model="large", mt_model="aya-expanse-8b")` — stores config, does not load models yet
- `__enter__` — loads Whisper and translation models into memory
- `__exit__` — releases model memory
- `session.transcribe(audio_path, source_lang=None) -> list[Cue]`
- `session.translate(cues, target) -> list[Cue]`
- `session.subtitle(audio_path, target, output=None, format="srt") -> Path`

**One-shot convenience functions** (thin wrappers around Session):
- `glossate.transcribe(audio_path, source_lang=None) -> list[Cue]`
- `glossate.translate(cues, target) -> list[Cue]`
- `glossate.subtitle(audio_path, target, output=None, format="srt") -> Path`
- `glossate.write_srt(cues, path)`, `write_vtt`, `write_json`

**Exception translation layer** (`_raise_translated` pattern from ELUATE):
- `FFmpegError` / `FFmpegTimeout` → `AudioExtractionError`
- `WhisperError` → `TranscriptionError`
- `MLXError` → `TranslationError`
- `DownloadError` → `ModelNotInstalledError`
- Stdlib exceptions (`FileNotFoundError`, `PermissionError`) propagate unwrapped

Include integration tests in `tests/test_api.py` — test exception translation and Session lifecycle with mocked internals. Stdlib exceptions must not be wrapped.

## Acceptance criteria

- [ ] `Session.__enter__` loads both models; `Session.__exit__` releases them
- [ ] `session.subtitle()` calls extract → transcribe → segment → translate → serialize in order
- [ ] One-shot `glossate.subtitle()` creates and closes a Session internally
- [ ] `FFmpegError` is translated to `AudioExtractionError` at the API boundary
- [ ] `WhisperError` is translated to `TranscriptionError`
- [ ] `MLXError` is translated to `TranslationError`
- [ ] `DownloadError` is translated to `ModelNotInstalledError`
- [ ] `FileNotFoundError` propagates unwrapped (not caught and re-raised as `GlossateError`)
- [ ] All public exceptions subclass `GlossateError`
- [ ] Output path uses `utils/paths` default when `output=None`
- [ ] All cases covered by tests; `pytest tests/test_api.py` passes
- [ ] `mypy` clean on `api.py`

## Blocked by

- `003-serializer.md`
- `004-output-paths.md`
- `007-ffmpeg-audio-extraction.md`
- `008-model-download.md`
- `010-transcriber.md`
- `011-translator.md`
