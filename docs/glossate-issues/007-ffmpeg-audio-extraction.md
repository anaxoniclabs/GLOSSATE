---
number: 007
title: ffmpeg audio extraction
status: todo
type: AFK
blocked_by: ["005-device-preflight.md"]
---

# 007 — ffmpeg audio extraction

## Parent

`docs/glossate-prd.md`

## What to build

Implement `utils/ffmpeg.py` — wraps ffmpeg subprocess calls for audio extraction from video files, and exposes `get_ffmpeg_version()` for the info command.

`extract_audio(input_path) -> Path`:
- If the input is already an audio file (WAV, MP3, M4A, FLAC), return the input path unchanged.
- If the input is a video file (MP4, MKV, WebM, etc.), extract the audio track to a temporary WAV file using ffmpeg and return the temp path.
- The caller is responsible for cleaning up the temp file (or use a context manager variant).
- Raises internal `FFmpegError` on non-zero ffmpeg exit; `FFmpegTimeout` if ffmpeg exceeds the timeout.

`get_ffmpeg_version() -> str | None`:
- Runs `ffmpeg -version` and returns the first line (e.g. `ffmpeg version 6.1 ...`).
- Returns `None` if ffmpeg is not on PATH.

Internal exceptions (`FFmpegError(RuntimeError)`, `FFmpegTimeout(FFmpegError)`) live in this module and are translated to public exceptions at the API boundary.

Include tests in `tests/test_ffmpeg.py` with mocked subprocess.

## Acceptance criteria

- [ ] Audio file input returns the input path unchanged (no ffmpeg call)
- [ ] Video file input triggers `ffmpeg -i <input> -vn -acodec pcm_s16le <output.wav>`
- [ ] Returned path for video input is a `.wav` file
- [ ] `FFmpegError` raised when ffmpeg exits non-zero (includes stage and last stderr lines in message)
- [ ] `FFmpegTimeout` raised when ffmpeg exceeds timeout
- [ ] `get_ffmpeg_version()` returns version string when ffmpeg is on PATH
- [ ] `get_ffmpeg_version()` returns `None` when ffmpeg is not found (no exception)
- [ ] All cases covered by tests with mocked subprocess; `pytest tests/test_ffmpeg.py` passes
- [ ] `mypy` clean on `utils/ffmpeg.py`

## Blocked by

- `005-device-preflight.md`
