# SPDX-License-Identifier: MIT
"""Whisper transcription wrapper for GLOSSATE."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty as _QueueEmpty
from typing import Any, Callable, Optional

_log = logging.getLogger("glossate.transcriber")

_MLX_MODEL_MAP: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

_FASTER_WHISPER_MODEL_MAP: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large-v3",
    "turbo": "large-v3-turbo",
}

_MODEL_MAP = _MLX_MODEL_MAP
_DEFAULT_MODEL = "turbo"

# Conservative real-time multipliers measured on Apple Silicon (MPS/Metal).
# Intentionally low so the progress bar moves honestly rather than freezing at
# ~48% when the estimate overshoots.  Turbo is empirically ~2–3× on M-series.
_SPEED_ESTIMATE: dict[str, float] = {
    "tiny":   8.0,
    "base":   5.0,
    "small":  3.5,
    "medium": 2.0,
    "large":  1.5,
    "turbo":  2.5,
}


class WhisperError(RuntimeError):
    """Whisper backend raised an exception."""


@dataclass
class WhisperWord:
    word: str
    start: float
    end: float


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str
    words: list[WhisperWord]


@dataclass
class WhisperResult:
    segments: list[WhisperSegment]
    detected_language: str


def _resolve_backend(backend: str = "auto", device: str | None = None) -> tuple[str, str]:
    if backend not in {"auto", "mlx", "faster-whisper"}:
        raise WhisperError(f"unknown ASR backend: {backend!r}")

    from glossate.utils.device import _cuda_available, get_optimal_device

    if backend == "faster-whisper" and device in (None, "auto"):
        return backend, "cuda" if _cuda_available() else "cpu"
    resolved_device = get_optimal_device(device)
    if backend == "auto":
        return ("mlx", resolved_device) if resolved_device == "mps" else ("faster-whisper", resolved_device)
    if backend == "mlx" and resolved_device == "cuda":
        raise WhisperError("MLX ASR cannot run on CUDA; use backend='faster-whisper'.")
    if backend == "faster-whisper" and resolved_device == "mps":
        raise WhisperError("faster-whisper does not support MPS; use device='cuda' or device='cpu'.")
    return backend, resolved_device


def _default_compute_type(device: str, compute_type: str | None) -> str:
    if compute_type and compute_type != "auto":
        return compute_type
    if device == "cuda":
        return "float16"
    return "int8"


def _run_mlx_transcribe(
    audio_str: str,
    repo: str,
    source_lang: str | None,
) -> dict[str, Any]:
    """Load mlx_whisper lazily and run transcription."""
    import mlx_whisper

    with open(os.devnull, "w") as _null, contextlib.redirect_stderr(_null):
        return mlx_whisper.transcribe(
            audio_str,
            path_or_hf_repo=repo,
            word_timestamps=True,
            language=source_lang,
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
        )


def _run_mlx_detect_language(clip_str: str, repo: str) -> dict[str, Any]:
    """Load mlx_whisper lazily and run language detection."""
    import mlx_whisper

    with open(os.devnull, "w") as _null, contextlib.redirect_stderr(_null):
        return mlx_whisper.transcribe(
            clip_str,
            path_or_hf_repo=repo,
            word_timestamps=False,
            verbose=False,
        )


def _faster_word(word: Any) -> dict[str, Any]:
    start = 0.0 if word.start is None else float(word.start)
    end = start if word.end is None else float(word.end)
    return {"word": word.word, "start": start, "end": end}


def _run_faster_whisper_transcribe(
    whisper: Any,
    audio_path: Path | str,
    source_lang: str | None,
    *,
    word_timestamps: bool,
    batch_size: int = 0,
    total_seconds: float = 0.0,
    on_progress: Optional[Callable[[float, float], None]] = None,
) -> dict[str, Any]:
    # On CUDA we run through BatchedInferencePipeline (batch_size > 0), which
    # transcribes audio chunks in parallel for a large GPU speedup. Its
    # transcribe() accepts only the widely-supported options, so the advanced
    # decoding thresholds are applied on the sequential (CPU) path only.
    common: dict[str, Any] = {
        "language": source_lang,
        "word_timestamps": word_timestamps,
        "vad_filter": True,
        "beam_size": 5,
    }
    if batch_size > 0:
        segments_iter, info = whisper.transcribe(
            str(audio_path), batch_size=batch_size, **common
        )
    else:
        segments_iter, info = whisper.transcribe(
            str(audio_path),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            **common,
        )

    segments: list[dict[str, Any]] = []
    for seg in segments_iter:
        words = [_faster_word(w) for w in (seg.words or [])] if word_timestamps else []
        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
                "words": words,
            }
        )
        if on_progress is not None and total_seconds > 0:
            on_progress(min(float(seg.end), total_seconds), total_seconds)

    if on_progress is not None and total_seconds > 0:
        on_progress(total_seconds, total_seconds)

    raw: dict[str, Any] = {
        "language": info.language or source_lang or "",
        "segments": segments,
    }
    if getattr(info, "language_probability", None) is not None:
        raw["language_probs"] = {raw["language"]: float(info.language_probability)}
    return raw


def _sp_transcribe(
    queue: mp.Queue,
    audio_str: str,
    repo: str,
    source_lang: str | None,
) -> None:
    """Subprocess entry point for isolated one-shot transcription."""
    try:
        queue.put(("ok", _run_mlx_transcribe(audio_str, repo, source_lang)))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("err", repr(exc)))


def _sp_detect_language(
    queue: mp.Queue,
    clip_str: str,
    repo: str,
) -> None:
    """Subprocess entry point for isolated one-shot language detection."""
    try:
        queue.put(("ok", _run_mlx_detect_language(clip_str, repo)))
    except BaseException as exc:  # noqa: BLE001
        queue.put(("err", repr(exc)))


def _worker_loop(request_q: mp.Queue, response_q: mp.Queue) -> None:
    """Persistent worker entry point used by Session for ASR batching."""
    while True:
        item = request_q.get()
        if item == "stop":
            return

        request_id, op, payload = item
        try:
            if op == "transcribe":
                audio_str, repo, source_lang = payload
                response = _run_mlx_transcribe(audio_str, repo, source_lang)
            else:
                raise WhisperError(f"unknown worker op: {op}")
            response_q.put((request_id, "ok", response))
        except BaseException as exc:  # noqa: BLE001
            response_q.put((request_id, "err", repr(exc)))


class WhisperSession:
    """Persistent Whisper session for either MLX or faster-whisper."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.backend, self.device = _resolve_backend(backend, device)
        self.compute_type = _default_compute_type(self.device, compute_type)
        # CUDA uses BatchedInferencePipeline for parallel-chunk transcription.
        self._faster_batch_size = 16 if self.device == "cuda" else 0
        self._models: dict[str, Any] = {}
        self._ctx = mp.get_context("spawn")
        self._request_q: mp.Queue | None = None
        self._response_q: mp.Queue | None = None
        self._worker: Any | None = None
        self._next_request_id = 0

    def close(self) -> None:
        self._models.clear()
        if self._worker is None:
            return
        if self._worker.is_alive() and self._request_q is not None:
            self._request_q.put("stop")
            self._worker.join(timeout=5)
        if self._worker.is_alive():
            self._worker.terminate()
            self._worker.join(timeout=5)
        self._worker = None
        self._request_q = None
        self._response_q = None

    def transcribe(
        self,
        audio_path: Path,
        source_lang: str | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        total_seconds: float = 0.0,
        on_progress: Optional[Callable[[float, float], None]] = None,
    ) -> WhisperResult:
        if self.backend == "faster-whisper":
            raw = _run_faster_whisper_transcribe(
                self._get_faster_model(model),
                audio_path,
                source_lang,
                word_timestamps=True,
                batch_size=self._faster_batch_size,
                total_seconds=total_seconds,
                on_progress=on_progress,
            )
            _dump_raw(raw, audio_path)
            result = _parse_result(raw)
            _log.info("transcribe: done backend=%s lang=%s segments=%d", self.backend, result.detected_language, len(result.segments))
            return result

        repo = _MLX_MODEL_MAP.get(model, model)
        raw = self._request(
            "transcribe",
            (str(audio_path), repo, source_lang),
            model=model,
            total_seconds=total_seconds,
            on_progress=on_progress,
        )
        _dump_raw(raw, audio_path)
        result = _parse_result(raw)
        _log.info("transcribe: done lang=%s segments=%d", result.detected_language, len(result.segments))
        return result

    def _get_faster_model(self, model: str) -> Any:
        model_id = _FASTER_WHISPER_MODEL_MAP.get(model, model)
        if model_id not in self._models:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise WhisperError(
                    "CUDA/CPU ASR requires faster-whisper. Install with: pip install 'glossate[cuda]'"
                ) from exc
            base = WhisperModel(
                model_id,
                device=self.device,
                compute_type=self.compute_type,
            )
            batched = False
            if self._faster_batch_size > 0:
                try:
                    from faster_whisper import BatchedInferencePipeline

                    self._models[model_id] = BatchedInferencePipeline(model=base)
                    batched = True
                except ImportError:
                    # Older faster-whisper without batched inference: fall back.
                    self._faster_batch_size = 0
                    self._models[model_id] = base
            else:
                self._models[model_id] = base
            _log.info("faster-whisper loaded model=%s device=%s compute_type=%s batched=%s", model_id, self.device, self.compute_type, batched)
        return self._models[model_id]

    def _ensure_started(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._request_q = self._ctx.Queue()
        self._response_q = self._ctx.Queue()
        worker = self._ctx.Process(
            target=_worker_loop,
            args=(self._request_q, self._response_q),
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def _request(
        self,
        op: str,
        payload: Any,
        *,
        model: str,
        total_seconds: float,
        on_progress: Optional[Callable[[float, float], None]],
    ) -> dict[str, Any]:
        self._ensure_started()
        if self._request_q is None or self._response_q is None or self._worker is None:
            raise WhisperError("transcribe worker did not start")

        request_id = self._next_request_id
        self._next_request_id += 1
        self._request_q.put((request_id, op, payload))

        speed = _SPEED_ESTIMATE.get(model, 2.0)
        start = time.monotonic()
        while True:
            try:
                response_id, status, response = self._response_q.get(timeout=0.15)
                if response_id != request_id:
                    continue
                break
            except _QueueEmpty:
                pass
            if not self._worker.is_alive():
                raise WhisperError("transcribe subprocess exited without a result")
            if on_progress is not None and total_seconds > 0:
                elapsed = time.monotonic() - start
                raw_frac = (elapsed * speed) / total_seconds
                on_progress(total_seconds * (1.0 - math.exp(-1.6 * raw_frac)), total_seconds)

        if status == "err":
            raise WhisperError(f"transcribe failed: {response}")
        if on_progress is not None and total_seconds > 0:
            on_progress(total_seconds, total_seconds)
        return response


def transcribe(
    audio_path: Path,
    source_lang: str | None = None,
    *,
    model: str = _DEFAULT_MODEL,
    total_seconds: float = 0.0,
    on_progress: Optional[Callable[[float, float], None]] = None,
    isolated: bool = True,
    device: str | None = None,
    backend: str = "auto",
    compute_type: str | None = None,
) -> WhisperResult:
    """Transcribe *audio_path* and return raw Whisper output.

    When *on_progress* and *total_seconds* are both provided, the caller
    receives smooth elapsed-time-based progress updates every ~150 ms.
    The MLX backend runs in a subprocess so Metal memory is freed on completion.

    Raises:
        WhisperError: the selected Whisper backend raised an exception.
    """
    resolved_backend, resolved_device = _resolve_backend(backend, device)
    repo = _MLX_MODEL_MAP.get(model, model)
    _log.info(
        "transcribe: start path=%s lang=%s model=%s backend=%s device=%s",
        audio_path,
        source_lang,
        model,
        resolved_backend,
        resolved_device,
    )

    if resolved_backend == "faster-whisper":
        session = WhisperSession(
            backend=resolved_backend,
            device=resolved_device,
            compute_type=compute_type,
        )
        try:
            return session.transcribe(
                audio_path,
                source_lang=source_lang,
                model=model,
                total_seconds=total_seconds,
                on_progress=on_progress,
            )
        finally:
            session.close()

    if not isolated:
        try:
            raw = _run_mlx_transcribe(str(audio_path), repo, source_lang)
        except Exception as exc:
            raise WhisperError(f"transcribe failed: {exc!r}") from exc
        _dump_raw(raw, audio_path)
        result = _parse_result(raw)
        _log.info("transcribe: done lang=%s segments=%d", result.detected_language, len(result.segments))
        return result

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    worker = ctx.Process(
        target=_sp_transcribe,
        args=(q, str(audio_path), repo, source_lang),
        daemon=True,
    )
    worker.start()

    speed = _SPEED_ESTIMATE.get(model, 2.0)
    start = time.monotonic()
    # Drain the queue concurrently with the worker. The child cannot exit while
    # its feeder thread is still flushing a large payload to the OS pipe, so
    # joining before reading would deadlock on results bigger than the pipe
    # buffer (~64 KB on macOS — i.e. anything past a short transcript).
    status: str | None = None
    payload: Any = None
    while True:
        try:
            status, payload = q.get(timeout=0.15)
            break
        except _QueueEmpty:
            pass
        if not worker.is_alive():
            try:
                status, payload = q.get_nowait()
            except _QueueEmpty:
                raise WhisperError("transcribe subprocess exited without a result")
            break
        if on_progress is not None and total_seconds > 0:
            elapsed = time.monotonic() - start
            # Asymptotic curve: always advances, never freezes.
            # At elapsed = total/speed (expected done): shows ~80%.
            # As elapsed → ∞: approaches total_seconds without reaching it.
            raw_frac = (elapsed * speed) / total_seconds
            on_progress(total_seconds * (1.0 - math.exp(-1.6 * raw_frac)), total_seconds)

    worker.join()

    if status == "err":
        raise WhisperError(f"transcribe failed: {payload}")

    raw_payload: dict[str, Any] = payload
    if on_progress is not None and total_seconds > 0:
        on_progress(total_seconds, total_seconds)

    _dump_raw(raw_payload, audio_path)
    result = _parse_result(raw_payload)
    _log.info("transcribe: done lang=%s segments=%d", result.detected_language, len(result.segments))
    return result


def detect_language(
    audio_path: Path,
    clip_start: float,
    clip_duration: float = 10.0,
    *,
    isolated: bool = True,
    device: str | None = None,
    backend: str = "auto",
    compute_type: str | None = None,
) -> tuple[str, float]:
    """Detect the language of a 10-second clip starting at *clip_start* seconds.

    Returns ``(language_code, confidence)``; confidence is 1.0 when the model
    does not return explicit per-language probabilities.
    The MLX backend runs in a subprocess so Metal memory is freed on completion.

    Raises:
        WhisperError: the selected Whisper backend raised an exception.
    """
    import librosa
    import soundfile as sf

    resolved_backend, resolved_device = _resolve_backend(backend, device)
    repo = _MLX_MODEL_MAP[_DEFAULT_MODEL]
    _log.info("detect_language: path=%s start=%.1f dur=%.1f", audio_path, clip_start, clip_duration)

    audio, sr = librosa.load(
        str(audio_path),
        offset=clip_start,
        duration=clip_duration,
        sr=16000,
        mono=True,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    clip_path = Path(tmp.name)

    try:
        sf.write(str(clip_path), audio, sr)

        if resolved_backend == "faster-whisper":
            session = WhisperSession(
                backend=resolved_backend,
                device=resolved_device,
                compute_type=compute_type,
            )
            try:
                raw = _run_faster_whisper_transcribe(
                    session._get_faster_model(_DEFAULT_MODEL),
                    clip_path,
                    None,
                    word_timestamps=False,
                    batch_size=session._faster_batch_size,
                )
            except Exception as exc:
                raise WhisperError(f"detect_language failed: {exc!r}") from exc
            finally:
                session.close()
        elif isolated:
            ctx = mp.get_context("spawn")
            q: mp.Queue = ctx.Queue()
            worker = ctx.Process(
                target=_sp_detect_language,
                args=(q, str(clip_path), repo),
                daemon=True,
            )
            worker.start()

            # Drain the queue concurrently — see note in transcribe().
            status: str | None = None
            payload: Any = None
            while True:
                try:
                    status, payload = q.get(timeout=0.15)
                    break
                except _QueueEmpty:
                    pass
                if not worker.is_alive():
                    try:
                        status, payload = q.get_nowait()
                    except _QueueEmpty:
                        raise WhisperError("detect_language subprocess exited without a result")
                    break

            worker.join()

            if status == "err":
                raise WhisperError(f"detect_language failed: {payload}")

            raw: dict[str, Any] = payload
        else:
            try:
                raw = _run_mlx_detect_language(str(clip_path), repo)
            except Exception as exc:
                raise WhisperError(f"detect_language failed: {exc!r}") from exc
    finally:
        clip_path.unlink(missing_ok=True)

    lang: str = raw.get("language", "")
    probs: dict[str, float] = raw.get("language_probs", {})
    confidence: float = probs.get(lang, 1.0) if probs else 1.0
    _log.info("detect_language: lang=%s conf=%.3f", lang, confidence)
    return (lang, confidence)


def _dump_raw(raw: dict[str, Any], audio_path: Path) -> None:
    if os.environ.get("GLOSSATE_DUMP_WHISPER_RAW", "").lower() not in {"1", "true", "yes"}:
        return
    stem = Path(audio_path).stem
    artifact = Path(tempfile.gettempdir()) / f"glossate_whisper_{stem}.json"
    artifact.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("whisper raw artifact: %s", artifact)


def _parse_result(raw: dict[str, Any]) -> WhisperResult:
    segments: list[WhisperSegment] = []
    for seg in raw.get("segments", []):
        words = [
            WhisperWord(word=w["word"], start=w["start"], end=w["end"])
            for w in seg.get("words", [])
        ]
        segments.append(
            WhisperSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                words=words,
            )
        )
    return WhisperResult(
        segments=segments,
        detected_language=raw.get("language", ""),
    )
