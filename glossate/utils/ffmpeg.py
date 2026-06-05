# SPDX-License-Identifier: MIT
"""FFmpeg helpers for GLOSSATE."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac"})


class FFmpegError(RuntimeError):
    """ffmpeg exited non-zero; message includes stage and last stderr lines."""


class FFmpegTimeout(FFmpegError):
    """ffmpeg did not finish within the allowed timeout."""


def extract_audio(input_path: Path | str, *, timeout: float = 300.0) -> Path:
    """Return a path to audio suitable for Whisper.

    If *input_path* is already an audio file (WAV/MP3/M4A/FLAC) it is returned
    unchanged.  For video inputs, the audio track is extracted to a temporary
    WAV file; the caller owns that file and must delete it when done.

    Raises:
        FFmpegError: ffmpeg exited non-zero.
        FFmpegTimeout: ffmpeg exceeded *timeout* seconds.
    """
    path = Path(input_path)
    if path.suffix.lower() in _AUDIO_EXTENSIONS:
        return path

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    output = Path(tmp.name)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-y",
        "-i", str(path),
        "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
        str(output),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output.unlink(missing_ok=True)
        raise FFmpegTimeout(f"extract: ffmpeg timed out after {timeout}s") from exc

    if result.returncode != 0:
        output.unlink(missing_ok=True)
        stderr_tail = "\n".join(result.stderr.splitlines()[-5:])
        raise FFmpegError(f"extract: ffmpeg exited {result.returncode}\n{stderr_tail}")

    return output


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for ffmpeg's subtitles filter argument."""
    value = str(path)
    return (
        value.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


def burn_subtitles(
    video_path: Path | str,
    subtitle_path: Path | str,
    output_path: Path | str,
    *,
    timeout: float = 3600.0,
) -> Path:
    """Burn subtitle text into video pixels and return the output path.

    This is a hard burn-in: subtitles become part of the video stream and
    cannot be toggled off in a player.

    Raises:
        FFmpegError: ffmpeg exited non-zero.
        FFmpegTimeout: ffmpeg exceeded *timeout* seconds.
    """
    video = Path(video_path)
    subtitles = Path(subtitle_path)
    output = Path(output_path)

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
        "-y",
        "-i", str(video),
        "-vf", f"subtitles={_escape_filter_path(subtitles)}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "copy",
        str(output),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output.unlink(missing_ok=True)
        raise FFmpegTimeout(f"burn: ffmpeg timed out after {timeout}s") from exc

    if result.returncode != 0:
        output.unlink(missing_ok=True)
        stderr_tail = "\n".join(result.stderr.splitlines()[-5:])
        raise FFmpegError(f"burn: ffmpeg exited {result.returncode}\n{stderr_tail}")

    return output


def get_audio_duration(audio_path: Path | str) -> float:
    """Return duration in seconds using ffprobe.

    Raises:
        FFmpegError: ffprobe failed or returned no output.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError) as exc:
        raise FFmpegError(f"get_audio_duration failed: {exc}") from exc


def get_ffmpeg_version() -> str | None:
    """Run ``ffmpeg -version`` and return the first line, or ``None`` if not found."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.splitlines()[0]
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None
