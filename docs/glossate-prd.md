# GLOSSATE PRD

## Problem Statement

Video content in Arabic, English, and other languages is inaccessible to Turkish-speaking viewers who cannot follow the original audio. Existing subtitle tools either require cloud services, lack support for Arabic→Turkish translation, or produce subtitles with poor terminology for Islamic and religious content. There is no self-contained CLI tool that can run efficiently on both Apple Silicon and CUDA cloud/Colab machines, going from a video or audio file to translated subtitle files in a single command.

---

## Solution

GLOSSATE is a Python CLI and library that accepts a video or audio file, transcribes it using Whisper, translates the transcription into a target language, and outputs subtitle files in SRT, VTT, or JSON format. It runs on Apple Silicon through MLX and on CUDA/Colab through faster-whisper plus Gemma 4 via Transformers, requires no hosted cloud API, and is designed to compose with ELUATE (audio cleanup) and OCCLUDE (visual cleanup) as part of a three-tool accessibility suite.

---

## User Stories

### CLI — Core Transcription & Translation

1. As a CLI user, I want to pass a video or audio file and get an SRT subtitle file, so that I can add subtitles to a video without writing any code.
2. As a CLI user, I want to specify a target language with `--target tr`, so that the output subtitles are in Turkish rather than the source language.
3. As a CLI user, I want to specify the source language with `--source ar`, so that I can override auto-detection when I already know the language.
4. As a CLI user, I want the tool to auto-detect the source language from the audio, so that I don't need to specify it for common use cases.
5. As a CLI user, I want language detection to sample three positions in the audio (20%, 50%, 80%), show me a suggestion with confidence, and auto-proceed after a timeout, so that I get a reliable language suggestion without blocking automation.
6. As a CLI user, I want `--source` to bypass the language detection entirely, so that batch scripts run without interactive prompts.
7. As a CLI user, I want to choose the output format with `--format srt|md`, so that I can produce either timed captions or Gemma-formatted readable notes (with `--md-scope`, `--md-layout`, `--md-timestamps` tuning the Markdown).
8. As a CLI user, I want to choose the Whisper model size with `--asr-model tiny|base|small|medium|large`, so that I can trade off speed against accuracy.
9. As a CLI user, I want to specify an explicit output path with `-o`, so that I can control where the subtitle file is written.
10. As a CLI user, I want subtitle files to go to `~/Documents/GLOSSATE/YYYY-MM-DD/` by default, so that my outputs are organized and easy to find.
11. As a CLI user, I want output filenames to auto-increment (`lecture.1.srt`, `lecture.2.srt`) on collision, so that re-running never overwrites previous work.
12. As a CLI user, I want to choose the inference device with `--device auto|mps|cuda|cpu`, so that I can override auto-detection when needed.
13. As a CLI user, I want to choose ASR and translation backends, so that I can run MLX on Apple Silicon and faster-whisper/Gemma (transformers) on CUDA without changing code.

### CLI — Info & Diagnostics

14. As a CLI user, I want to run `glossate info` and see device, model cache status, ffmpeg version, free memory, and the configured output directory, so that I can diagnose issues before running a job.
15. As a CLI user, I want model cache status to show file size when cached and "not downloaded" when missing, so that I know exactly what to download.
16. As a CLI user, I want `glossate info` to tell me when ffmpeg is not on PATH, so that I can install it before attempting video input.
17. As a CLI user, I want `--verbose` to activate structured log output, so that I can debug a failing run without modifying code.

### CLI — Progress Display

18. As a CLI user, I want a live progress display with stage bullets (Extract audio → Transcribe → Translate) and a progress bar for the active stage, so that I can see what the tool is doing during long runs.
19. As a CLI user, I want completed stages to show a checkmark and pending stages to show a hollow bullet, so that I can see the overall pipeline status at a glance.
20. As a CLI user, I want the progress display to show a spinner during model loading (before any measurable progress), so that the tool never appears frozen.
21. As a CLI user, I want the Extract audio stage to be skipped silently when the input is already an audio file, so that the progress display only shows relevant stages.
22. As a CLI user, I want the Translate stage to be skipped silently when no `--target` is given, so that transcription-only runs are clean.

### Python API — One-Shot Functions

23. As a Python developer, I want to call `glossate.transcribe("audio.wav")` and get a list of timed cues, so that I can integrate transcription into a notebook or script.
24. As a Python developer, I want to call `glossate.translate(cues, target="tr")` and get translated cues with the same timestamps, so that I can translate a transcription I already have.
25. As a Python developer, I want to call `glossate.subtitle("audio.wav", target="tr", output="out.srt")` as a one-shot convenience function, so that I can produce a subtitle file in a single line.
26. As a Python developer, I want `glossate.write(cues, "out.srt")` to serialize cues to an SRT file, and `glossate.subtitle(..., format="md")` to produce Gemma-formatted Markdown notes, so that I can derive captions or readable notes from one transcription.
27. As a Python developer, I want one-shot functions to load and unload models automatically, so that I don't need to manage model lifecycle for simple use cases.

### Python API — Session

28. As a Python developer, I want to use `glossate.Session` as a context manager, so that models load once and are reused across multiple files.
29. As a Python developer, I want `session.subtitle(audio_path, target="tr", output=...)` to use already-loaded models, so that batch processing of many files is fast.
30. As a Python developer, I want `Session` to accept `asr_model`, `mt_model`, `device`, backend, and compute-type parameters, so that I can choose model sizes and runtime strategy for the session.
31. As a Python developer, I want `Session` to clean up model memory on `__exit__`, so that resources are released predictably.

### Python API — Composition with ELUATE

32. As a Python developer, I want to pass `eluate.hush()` output directly to `glossate.subtitle()`, so that the two tools compose without intermediate files.
33. As a Python developer, I want GLOSSATE to accept the path to a speech-only WAV (as produced by ELUATE's speech stem) and produce clean subtitles, so that the pipeline produces better transcription quality than raw video audio.

### Cue Quality

34. As a user, I want subtitle cues to respect Whisper's natural sentence segmentation, so that cue breaks align with speech pauses and meaning boundaries.
35. As a user, I want cues longer than 42 characters to be split at punctuation first, then at the nearest word boundary, so that no subtitle line is too long to read.
36. As a user, I want split cues to use actual word timestamps at the split point, so that the timing is accurate rather than proportionally estimated.
37. As a user, I want translated cues to preserve the original timestamps from transcription, so that subtitles stay in sync with the audio.

### Translation Quality & Reliability

38. As a user, I want translation to be batched, so that the model has sentence context across cue boundaries and CUDA jobs avoid per-cue overhead.
39. As a user, I want prompt-based translation to retry the full batch once on parse failure before falling back to per-cue translation, so that a single malformed response doesn't lose an entire batch.
40. As a user, I want per-cue fallback translation to guarantee a 1:1 mapping even when the prompt model is confused, so that no subtitle line is silently dropped.
41. As a user, I want the batched Gemma (transformers) translation path to preserve a 1:1 output count for each input cue batch, so that no subtitle line is silently dropped.
42. As a user, I want unsupported language pairs to produce a clear error message suggesting alternatives, so that I know what to do rather than getting a silent failure.

### Error Handling

43. As a user, I want ffmpeg extraction failures to produce a clear error message suggesting I install ffmpeg, so that I can fix the issue myself.
44. As a user, I want model-not-found errors to tell me to run `glossate info` to download the model, so that the fix is one step away.
45. As a user, I want all GLOSSATE errors to subclass `GlossateError`, so that callers can catch them broadly or specifically.
46. As a user, I want stdlib exceptions (`FileNotFoundError`, `PermissionError`) to propagate unwrapped, so that they have the correct Python name for debugging.

### Packaging & Installation

47. As a developer, I want `pip install glossate` to install the tool and its CLI entry point, so that setup is a single command.
48. As a developer, I want `pip install glossate[apple]` and `pip install glossate[cuda]` to install platform-specific model runtimes, so that local Macs and cloud GPUs do not pull unnecessary dependencies.
49. As a developer, I want `pip install glossate[dev]` to install pytest, ruff, and mypy, so that the dev environment is reproducible.
50. As a developer, I want the package to require Python ≥ 3.10, matching ELUATE's minimum, so that the trio has a consistent runtime requirement.
51. As a developer, I want GitHub Actions OIDC trusted publishing to PyPI, so that releases are automated without storing API keys.

---

## Implementation Decisions

### Module Architecture

**`core/segmenter`** — Deep module. Takes raw Whisper output (segments with word-level timestamps) and returns a normalized list of `Cue` objects. Splits segments exceeding 42 characters at punctuation boundaries first, then at word boundaries, using actual word timestamps at the split point. No model dependencies — pure Python.

**`core/serializer`** — Pure, no model dependencies. `write_srt` dumps `Cue` objects to SRT with enforced timecode formatting; `render_prose_md` / `render_dual_stack_md` render *already-prepared* notes data (prose blocks or sentence pairs) to Markdown strings.

**`core/notes`** — Markdown orchestration. Chunks cues into time-windows and calls Gemma (`translator.format_prose`) to reflow each into clean prose; for the dual-stack layout it segments cues into sentences and translates them sentence-by-sentence. Timestamps are computed in code from cue start times, never produced by the model.

**`core/transcriber`** — Wraps Whisper ASR backends. Uses mlx-whisper on MPS and faster-whisper on CUDA/CPU. Accepts an audio file path, optional source language, backend, device, and compute type. Returns raw Whisper output (segments + word timestamps + detected language). Hides backend API details. Raises `WhisperError` (internal) on failure.

**`core/translator`** — Wraps translation backends. The transformers (Gemma on CUDA/MPS/CPU), MLX, and Ollama backends all use prompt-style translation with XML-tagged cue batches (`<cue id="N">text</cue>`), regex parsing, one batch retry on parse failure, and per-cue fallback on second failure. Raises `TranslationError` (internal) on model failure.

**`utils/ffmpeg`** — Wraps ffmpeg subprocess for audio extraction. Accepts a video or audio path; if video, extracts audio to a temp WAV; if already audio, returns path unchanged. Raises `FFmpegError` / `FFmpegTimeout` (internal). Mirrors ELUATE's ffmpeg utility.

**`utils/paths`** — Output path logic. Constructs `~/Documents/GLOSSATE/YYYY-MM-DD/<stem>.<ext>`, handles auto-increment on collision. Exposes `get_output_path(stem, format, date) → Path`.

**`utils/device`** — Detects MPS > CUDA > CPU. Respects `--device` override with validation and reports CUDA GPU memory, Linux memory, or macOS memory depending on runtime.

**`utils/download`** — Checks model weight cache via huggingface-hub. Downloads on first use. Raises `ModelNotInstalledError` (public) if download fails.

**`utils/preflight`** — Runs pre-flight checks before a job: ffmpeg on PATH, models cached, disk space. Raises appropriate typed exceptions. Mirrors ELUATE's preflight.

**`api.py`** — Public surface. Exports `transcribe`, `translate`, `subtitle`, `Session`, and the public exception hierarchy (`GlossateError`, `ModelNotInstalledError`, `AudioExtractionError`, `TranscriptionError`, `TranslationError`, `UnsupportedLanguageError`). Contains the exception translation layer (internal → public). One-shot functions are thin wrappers around `Session`.

**`cli.py`** — argparse entry point. Handles language detection flow: samples audio at 20/50/80% duration (10s clips each), runs Whisper language detection on each clip, majority-votes the result, prompts user with auto-proceed timeout. Integrates `GlossateProgress` for live display. Handles `glossate info` subcommand.

**`ui/`** — Rich components: `GlossateProgress` (mirrors `EluateProgress` — stage bullets + active progress bar, 20 Hz refresh, bouncingBall spinner), `info_panel` (mirrors ELUATE's), `theme.py` (B&W palette, `glossate.*` namespace), `ascii_art.py` (`GLOSSATE` ANSI Shadow figlet + `GLOSSATE` subtitle line).

### Cue Data Model

A `Cue` is `{start: float, end: float, text: str, lang: str}` where `start`/`end` are seconds. This is the unit passed between all public API functions.

### Translation Prompt Contract

Each batch is sent as:
```
Translate the following subtitle cues from {source} to {target}.
Return only the translated cues in the same XML format. Do not merge or split cues.

<cue id="1">text</cue>
<cue id="2">text</cue>
...
```
Response parsed via `re.findall(r'<cue id="(\d+)">(.*?)</cue>', response, re.DOTALL)`. Count mismatch triggers retry.

### Exception Hierarchy (two-layer)

Internal (stdlib base, in utility modules): `FFmpegError(RuntimeError)`, `FFmpegTimeout(FFmpegError)`, `WhisperError(RuntimeError)`, `MLXError(RuntimeError)`.

Public (GlossateError base, in api.py): `ModelNotInstalledError`, `AudioExtractionError`, `TranscriptionError`, `TranslationError`, `UnsupportedLanguageError`.

`api.py` translates internal → public via a `_raise_translated` pattern (mirrors ELUATE).

### Logging

Every module uses `logging.getLogger("glossate.<module>")`. Silent by default. Activated via `--verbose` CLI flag. Stage start/complete/error events are logged at INFO. Rich console handles all user-facing display separately.

### Output Path

Default: `~/Documents/GLOSSATE/YYYY-MM-DD/<stem>.<ext>`. Auto-increment on collision: `<stem>.1.<ext>`, `<stem>.2.<ext>`, etc. `-o` overrides entirely. Directory created on first write if it does not exist.

### Session is the Real Implementation

`glossate.subtitle()` and other one-shot functions create a `Session`, call the session method, and close it. `Session.__enter__` loads models; `Session.__exit__` releases them. This is the only place models are held in memory.

### Dependencies

Base: `rich`. Optional extras: `[apple]` for MLX runtimes, `[cuda]` for faster-whisper plus torch/Transformers (Gemma) support, `[detect]` for audio sampling helpers, `[mlx-translate]` for the MLX translation runtime, `[all]` for every runtime, and `[dev]` for pytest, coverage, ruff, and mypy.

---

## Testing Decisions

### What makes a good test

Tests verify external behavior through the module's public interface — inputs in, outputs out, exceptions raised. They do not assert on internal state, private methods, or implementation details. A test that breaks when you refactor internals without changing behavior is a bad test.

### Modules to test (all of them)

**`core/segmenter`** — Unit tests, no mocks needed. Test: Whisper segments within 42 chars pass through unchanged; segments over 42 chars split at punctuation; segments with no punctuation split at word boundary; split uses actual word timestamps; back-to-back splits on very long segments; empty segment list; single-word segment.

**`core/serializer`** — Unit tests, no mocks needed. Test: SRT timecode format (`HH:MM:SS,mmm --> HH:MM:SS,mmm`); VTT header and timecode format; JSON roundtrip (parse back and compare); empty cue list produces valid empty file; special characters in text (Arabic, Turkish diacritics) preserved.

**`utils/paths`** — Unit tests, no mocks needed. Test: default path includes today's date folder; stem derived from input filename; format maps to correct extension; first file has no suffix; second file gets `.1`; collision chain increments correctly; `-o` override bypasses default path; directory is created if absent.

**`utils/device`** — Unit tests with mock. Test: MPS preferred when available; CUDA second; CPU fallback; `--device mps` override accepted; `--device cuda` raises when CUDA unavailable; `--device unknown` raises.

**`utils/ffmpeg`** — Unit tests with mocked subprocess. Test: audio file returned unchanged; video file triggers ffmpeg extraction; `FFmpegError` raised on non-zero exit; `FFmpegTimeout` raised on timeout; temp file cleaned up after extraction.

**`core/translator`** — Unit tests with mocked backends. Test: transformers/MLX/Ollama batch prompt contains correct XML structure; response parsed correctly; cue count mismatch triggers one retry; second failure falls back to per-cue; per-cue fallback produces correct 1:1 output; batch boundary behavior; empty cue list returns empty list.

**`core/transcriber`** — Unit tests with mocked Whisper backends. Test: word timestamps passed through to output; detected language included in result; explicit source language passed to model; CUDA auto path selects faster-whisper with CUDA-friendly model names and compute type; `WhisperError` raised on model failure.

**`utils/download`** — Unit tests with mocked huggingface-hub. Test: cached model returns path without downloading; uncached model triggers download; download failure raises `ModelNotInstalledError`.

**`utils/preflight`** — Unit tests with mocked dependencies. Test: all checks pass silently; ffmpeg missing raises `AudioExtractionError`; model missing raises `ModelNotInstalledError`.

**`api.py`** — Integration tests (light). Test: internal `FFmpegError` translated to public `AudioExtractionError`; internal `WhisperError` translated to `TranscriptionError`; internal `MLXError` translated to `TranslationError`; `Session` context manager loads/releases models; one-shot `subtitle()` calls Session internally.

**`ui/components`** — Render tests. Test: `info_panel` produces a Rich `Panel`; `GlossateProgress` stage transitions (pending → active → complete → error); ASCII art renders without exception.

**`cli.py`** — Integration tests with mocked API. Test: language detection samples at 20/50/80% positions; majority vote selects correct language; `--source` bypasses detection; auto-proceed fires after timeout; `glossate info` prints two panels; output path written to correct date folder; collision increments filename.

### Prior art

Mirror ELUATE's test structure: `tests/test_<module>.py` per module, pytest fixtures in `tests/conftest.py`, mock heavy dependencies with `unittest.mock.patch`. Tests run without model weights downloaded.

---

## Out of Scope

- **RAG for Islamic/Quranic content** — Tanzil corpus + Diyanet glossary deferred to v0.2.0.
- **Real GPU benchmarking** — automated tests mock model runtimes; real CUDA/MPS throughput benchmarks remain a manual validation task.
- **Streaming transcription** — full audio loaded into memory for v0.1.0; chunked streaming deferred.
- **Speaker diarization** — no speaker labeling in v0.1.0.
- **Custom glossaries** — user-provided term dictionaries deferred.
- **Web API / REST service** — CLI and Python library only for v0.1.0.
- **Hard subtitle burn-in** — GLOSSATE outputs subtitle files only; ffmpeg muxing is the caller's responsibility.
- **OCCLUDE integration** — the blur+subtitle pipeline is a future composition concern.
- **Windows support** — macOS and Linux/CUDA are the active targets for v0.1.0.

---

## Further Notes

- GLOSSATE is part of a trio: ELUATE (audio cleanup) + OCCLUDE (visual cleanup) + GLOSSATE (language bridge). All three share the same B&W Rich UI style, the same two-layer exception pattern, and the same project structure conventions.
- The primary language pair driving the project is Arabic→Turkish, with strong motivation for Islamic religious content. Standard translation quality is acceptable for v0.1.0; Quranic terminology precision is deferred.
- Model storage depends on selected backends. Whisper large/turbo and Gemma 4 models are multiple-GB downloads; Ollama/MLX model size depends on the chosen model.
- Gemma 4 via HF transformers is the default CUDA/Colab translation path (`gemma-4-e4b`, general multilingual, no server). MLX (Apple Silicon) and Ollama remain available for local LLM translation workflows.
- Tests must run without model weights. All model calls are behind interfaces that can be mocked. CI should pass on any machine without downloading ~10 GB of weights.
