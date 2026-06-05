# SPDX-License-Identifier: MIT
"""GLOSSATE public API."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Generator, Optional

from glossate.core import cache as _cache
from glossate.core import notes as _notes
from glossate.core import segmenter as _segmenter
from glossate.core import serializer as _serializer
from glossate.core import transcriber as _transcriber
from glossate.core import translator as _translator
from glossate.core.segmenter import Cue
from glossate.core.transcriber import WhisperResult
from glossate.core.translator import TranslationModelState
from glossate.utils import ffmpeg as _ffmpeg
from glossate.utils import paths as _paths

# ---------------------------------------------------------------------------
# Public exception hierarchy
# ---------------------------------------------------------------------------


class GlossateError(Exception):
    """Base exception for all GLOSSATE errors."""


class ModelNotInstalledError(GlossateError):
    """A required model is not installed. Run ``glossate info`` to download."""


class AudioExtractionError(GlossateError):
    """Audio extraction via ffmpeg failed."""


class SubtitleBurnError(GlossateError):
    """Hard subtitle burn-in via ffmpeg failed."""


class TranscriptionError(GlossateError):
    """Whisper transcription failed."""


class TranslationError(GlossateError):
    """Translation via the selected backend failed."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _translate_exceptions() -> Generator[None, None, None]:
    """Translate internal exceptions to public API exceptions at the boundary."""
    from glossate.core.transcriber import WhisperError
    from glossate.core.translator import TranslationError as _MTError
    from glossate.utils.download import DownloadError
    from glossate.utils.ffmpeg import FFmpegError

    try:
        yield
    except FFmpegError as exc:
        raise AudioExtractionError(str(exc)) from exc
    except WhisperError as exc:
        raise TranscriptionError(str(exc)) from exc
    except _MTError as exc:
        raise TranslationError(str(exc)) from exc
    except DownloadError as exc:
        raise ModelNotInstalledError(str(exc)) from exc


def _result_to_dict(result: WhisperResult) -> dict[str, Any]:
    return {
        "language": result.detected_language,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in seg.words
                ],
            }
            for seg in result.segments
        ],
    }


_VALID_FORMATS = frozenset({"srt", "md"})
_BURNABLE_FORMATS = frozenset({"srt"})
_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac"})


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Context manager that loads models once and processes multiple files.

    Usage::

        with glossate.Session() as s:
            for path in paths:
                s.subtitle(path, target="tr", output=f"{path}.srt")
    """

    def __init__(
        self,
        *,
        asr_model: str = "turbo",
        mt_model: str = "gemma-4-e4b",
        device: Optional[str] = None,
        asr_backend: str = "auto",
        mt_backend: str = "auto",
        compute_type: Optional[str] = None,
        cache: bool = True,
    ) -> None:
        self.asr_model = asr_model
        self.mt_model = mt_model
        self.device = device
        self.asr_backend = asr_backend
        self.mt_backend = mt_backend
        self.compute_type = compute_type
        self.cache = cache
        self._mt_state: Optional[TranslationModelState] = None
        self._asr_session: Optional[_transcriber.WhisperSession] = None
        self._active = False

    def __enter__(self) -> "Session":
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._mt_state is not None:
            _translator.unload_model(self._mt_state)
            self._mt_state = None
        if self._asr_session is not None:
            self._asr_session.close()
            self._asr_session = None
        self._active = False

    def _get_mt_state(self) -> TranslationModelState:
        if not self._active:
            raise RuntimeError("Session is not active. Use 'with Session() as s:' before calling translate().")
        if self._mt_state is None:
            with _translate_exceptions():
                self._mt_state = _translator.load_model(
                    self.mt_model,
                    device=self.device,
                    backend=self.mt_backend,
                    compute_type=self.compute_type,
                )
        return self._mt_state

    def _get_asr_session(self) -> _transcriber.WhisperSession:
        if not self._active:
            raise RuntimeError("Session is not active. Use 'with Session() as s:' before calling transcribe().")
        if self._asr_session is None:
            self._asr_session = _transcriber.WhisperSession(
                backend=self.asr_backend,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._asr_session

    def _transcribe_cached(self, input_path: Path, source: Optional[str]) -> list[Cue]:
        """Return segmented source cues, using the on-disk transcript cache."""
        key = (
            _cache.transcript_key(
                input_path, asr_model=self.asr_model, asr_backend=self.asr_backend, source=source
            )
            if _cache.cache_enabled(self.cache)
            else None
        )
        cached = _cache.load(key)
        if cached is not None:
            return cached

        audio = _ffmpeg.extract_audio(input_path)
        is_temp = audio != input_path
        try:
            result = self._get_asr_session().transcribe(
                audio, source_lang=source, model=self.asr_model
            )
            cues = _segmenter.segment(_result_to_dict(result))
        finally:
            if is_temp:
                audio.unlink(missing_ok=True)
        _cache.save(key, cues)
        return cues

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        source: Optional[str] = None,
    ) -> list[Cue]:
        """Extract audio, transcribe, and return timed cues in the source language."""
        input_path = Path(audio_path)
        with _translate_exceptions():
            return self._transcribe_cached(input_path, source)

    def translate(
        self,
        cues: list[Cue],
        *,
        target: str,
        source: Optional[str] = None,
    ) -> list[Cue]:
        """Translate cues to target language, preserving timestamps."""
        mt_state = self._get_mt_state()
        with _translate_exceptions():
            return _translator.translate(cues, target, source=source, model_state=mt_state)

    def subtitle(
        self,
        audio_path: str | Path,
        *,
        target: Optional[str] = None,
        source: Optional[str] = None,
        format: str = "srt",
        output: Optional[str | Path] = None,
        md_scope: str = "translated",
        md_layout: str = "two-prose",
        md_timestamps: bool = True,
    ) -> Path:
        """Extract, transcribe, optionally translate, and write the output file.

        ``format="srt"`` writes timed cues (translated in place when *target* is
        given). ``format="md"`` writes Gemma-formatted readable notes and always
        needs the MT model — even with no *target* — to reflow the transcript.
        The ``md_*`` arguments only apply to Markdown.
        """
        if format not in _VALID_FORMATS:
            raise ValueError(f"Unknown format {format!r}; expected one of {sorted(_VALID_FORMATS)}")
        needs_model = format == "md" or target is not None
        if needs_model and not self._active:
            raise RuntimeError("Session is not active. Use 'with Session() as s:' before calling subtitle().")

        input_path = Path(audio_path)
        with _translate_exceptions():
            cues = self._transcribe_cached(input_path, source)
            source_lang = source or (cues[0].lang if cues else "")
            out = (
                Path(output)
                if output is not None
                else _paths.default_output_path(input_path, format)  # type: ignore[arg-type]
            )

            if format == "md":
                md = _notes.build_markdown(
                    cues,
                    source_lang=source_lang,
                    target_lang=target,
                    scope=md_scope,
                    layout=md_layout,
                    timestamps=md_timestamps,
                    model_state=self._get_mt_state(),
                )
                out.write_text(md, encoding="utf-8")
                return out

            if target is not None:
                cues = _translator.translate(
                    cues, target, source=source_lang, model_state=self._get_mt_state()
                )
            return _serializer.write_srt(cues, out)

    def subtitle_video(
        self,
        video_path: str | Path,
        *,
        target: Optional[str] = None,
        source: Optional[str] = None,
        format: str = "srt",
        subtitle_output: Optional[str | Path] = None,
        output: Optional[str | Path] = None,
    ) -> Path:
        """Create subtitles and burn them into a video.

        Returns the path to the hard-subtitled video. The intermediate subtitle
        file is kept at *subtitle_output* or the normal default subtitle path.
        """
        if format not in _BURNABLE_FORMATS:
            raise ValueError(
                f"Cannot burn format {format!r}; expected one of {sorted(_BURNABLE_FORMATS)}"
            )
        subtitle_path = self.subtitle(
            video_path,
            target=target,
            source=source,
            format=format,
            output=subtitle_output,
        )
        return burn_subtitles(video_path, subtitle_path, output=output)


# ---------------------------------------------------------------------------
# One-shot convenience functions
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    *,
    model: str = "turbo",
    source: Optional[str] = None,
    device: Optional[str] = None,
    backend: str = "auto",
    compute_type: Optional[str] = None,
    cache: bool = True,
) -> list[Cue]:
    """Transcribe audio to timed cues in the source language.

    ``backend`` is the ASR backend (maps to ``Session(asr_backend=...)``).
    """
    with Session(asr_model=model, device=device, asr_backend=backend, compute_type=compute_type, cache=cache) as s:
        return s.transcribe(audio_path, source=source)


def translate(
    cues: list[Cue],
    *,
    target: str,
    source: Optional[str] = None,
    model: str = "gemma-4-e4b",
    device: Optional[str] = None,
    backend: str = "auto",
    compute_type: Optional[str] = None,
) -> list[Cue]:
    """Translate a list of cues to the target language.

    ``backend`` is the translation backend (maps to ``Session(mt_backend=...)``).
    """
    with Session(mt_model=model, device=device, mt_backend=backend, compute_type=compute_type) as s:
        return s.translate(cues, target=target, source=source)


def subtitle(
    audio_path: str | Path,
    *,
    target: Optional[str] = None,
    source: Optional[str] = None,
    asr_model: str = "turbo",
    mt_model: str = "gemma-4-e4b",
    format: str = "srt",
    output: Optional[str | Path] = None,
    md_scope: str = "translated",
    md_layout: str = "two-prose",
    md_timestamps: bool = True,
    device: Optional[str] = None,
    asr_backend: str = "auto",
    mt_backend: str = "auto",
    compute_type: Optional[str] = None,
    cache: bool = True,
) -> Path:
    """Convenience one-shot: transcribe + (optionally) translate + write.

    Returns the path to the written output file. The ``md_*`` arguments apply
    only when ``format="md"``.
    """
    with Session(
        asr_model=asr_model,
        mt_model=mt_model,
        device=device,
        asr_backend=asr_backend,
        mt_backend=mt_backend,
        compute_type=compute_type,
        cache=cache,
    ) as s:
        return s.subtitle(
            audio_path,
            target=target,
            source=source,
            format=format,
            output=output,
            md_scope=md_scope,
            md_layout=md_layout,
            md_timestamps=md_timestamps,
        )


def burn_subtitles(
    video_path: str | Path,
    subtitle_path: str | Path,
    *,
    output: Optional[str | Path] = None,
) -> Path:
    """Hard-burn an existing subtitle file into a video and return the video path."""
    video = Path(video_path)
    if video.suffix.lower() in _AUDIO_EXTENSIONS:
        raise ValueError("burn_subtitles requires a video input, not an audio file")
    out = Path(output) if output is not None else _paths.default_burn_output_path(video)
    try:
        return _ffmpeg.burn_subtitles(video, Path(subtitle_path), out)
    except _ffmpeg.FFmpegError as exc:
        raise SubtitleBurnError(str(exc)) from exc


def subtitle_video(
    video_path: str | Path,
    *,
    target: Optional[str] = None,
    source: Optional[str] = None,
    asr_model: str = "turbo",
    mt_model: str = "gemma-4-e4b",
    format: str = "srt",
    subtitle_output: Optional[str | Path] = None,
    output: Optional[str | Path] = None,
    device: Optional[str] = None,
    asr_backend: str = "auto",
    mt_backend: str = "auto",
    compute_type: Optional[str] = None,
    cache: bool = True,
) -> Path:
    """Convenience one-shot: create subtitles and burn them into a video.

    Returns the path to the hard-subtitled video.
    """
    with Session(
        asr_model=asr_model,
        mt_model=mt_model,
        device=device,
        asr_backend=asr_backend,
        mt_backend=mt_backend,
        compute_type=compute_type,
        cache=cache,
    ) as s:
        return s.subtitle_video(
            video_path,
            target=target,
            source=source,
            format=format,
            subtitle_output=subtitle_output,
            output=output,
        )


# ---------------------------------------------------------------------------
# Write helpers (thin wrappers over serializer)
# ---------------------------------------------------------------------------


def write(cues: list[Cue], output: str | Path) -> Path:
    """Write cues to an SRT file (inferred from a ``.srt`` extension).

    Markdown notes are not a plain cue dump (they need the Gemma formatting
    pass), so they are produced via :func:`subtitle` / ``Session.subtitle`` with
    ``format="md"``, not here.
    """
    out = Path(output)
    if out.suffix.lower() == ".srt":
        return _serializer.write_srt(cues, out)
    raise ValueError(
        f"Cannot infer format from {out.name!r}; use a .srt extension "
        f"(Markdown notes are written via subtitle(..., format='md'))."
    )
