---
number: 010
title: Transcriber (mlx-whisper wrapper)
status: done
type: AFK
blocked_by: ["005-device-preflight.md", "002-cue-model-segmenter.md"]
---

# 010 — Transcriber (mlx-whisper wrapper)

## Parent

`docs/glossate-prd.md`

## What to build

Implement `core/transcriber.py` — wraps mlx-whisper to transcribe an audio file and return raw Whisper output (segments with word-level timestamps and detected language).

`transcribe(audio_path: Path, source_lang: str | None = None) -> WhisperResult`:
- Loads the Whisper large-v3 model via mlx-whisper (model size configurable).
- Passes `language=source_lang` to Whisper when provided; otherwise Whisper auto-detects.
- Returns a `WhisperResult` containing: list of segments (each with `start`, `end`, `text`, `words`), `detected_language` string.
- Raises internal `WhisperError(RuntimeError)` on model failure.

`detect_language(audio_path: Path, clip_start: float, clip_duration: float = 10.0) -> tuple[str, float]`:
- Extracts a short clip from the audio at `clip_start` seconds.
- Runs Whisper language detection on that clip.
- Returns `(language_code, confidence)`.
- Used by the CLI for the 20/50/80% sampling flow.

The transcriber does not perform segmentation — it returns raw Whisper output. Segmentation is `core/segmenter`'s job.

Include tests in `tests/test_transcriber.py` with mocked mlx-whisper.

## Acceptance criteria

- [ ] `transcribe()` calls mlx-whisper with the audio path and optional language
- [ ] Explicit `source_lang` is passed to the model; None triggers auto-detection
- [ ] Returned `WhisperResult` includes segments with word-level timestamps
- [ ] `detected_language` field is populated from Whisper's output
- [ ] `WhisperError` raised when mlx-whisper raises an exception
- [ ] `detect_language()` extracts a 10-second clip and returns `(lang, confidence)`
- [ ] `detect_language()` clips are extracted via librosa (not ffmpeg)
- [ ] All cases covered by tests with mocked mlx-whisper; `pytest tests/test_transcriber.py` passes
- [ ] `mypy` clean on `core/transcriber.py`

## Blocked by

- `005-device-preflight.md`
- `002-cue-model-segmenter.md`
