# SPDX-License-Identifier: MIT
"""Preflight validation: ffmpeg availability and model cache checks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class _PreflightError(Exception):
    """Base for internal preflight failures."""


class _FfmpegMissingError(_PreflightError):
    """ffmpeg binary not found on PATH."""


class _ModelMissingError(_PreflightError):
    """A required model weight file is not cached."""


_WHISPER_HF_IDS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}

_MT_HF_IDS: dict[str, str] = {
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
    "gemma-4-26b": "google/gemma-4-26B-A4B-it",
    "gemma-4-31b": "google/gemma-4-31B-it",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "translate-gemma-4b": "mlx-community/translategemma-4b-it-4bit",
}


def _hf_hub_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return hf_home / "hub"


def _is_hf_model_cached(model_id: str) -> bool:
    dir_name = "models--" + model_id.replace("/", "--")
    return (_hf_hub_dir() / dir_name).exists()


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise _FfmpegMissingError(
            "ffmpeg not found on PATH. Install via: brew install ffmpeg"
        )


def _check_whisper_model(model: str, *, backend: str = "auto", device: str | None = None) -> None:
    if backend not in {"auto", "mlx", "faster-whisper"}:
        raise _ModelMissingError(f"Unknown ASR backend: {backend!r}")
    if backend == "faster-whisper":
        return
    if backend == "auto":
        from glossate.utils.device import get_optimal_device

        if get_optimal_device(device) != "mps":
            return
    hf_id = _WHISPER_HF_IDS.get(model, model)
    if not _is_hf_model_cached(hf_id):
        raise _ModelMissingError(
            f"Whisper model '{model}' (HuggingFace: {hf_id!r}) is not cached. "
            "Run `glossate info` to download."
        )


def _check_mt_model(model: str) -> None:
    if model.startswith("ollama/"):
        return
    hf_id = _MT_HF_IDS.get(model, model)
    if not _is_hf_model_cached(hf_id):
        raise _ModelMissingError(
            f"Translation model '{model}' (HuggingFace: {hf_id!r}) is not cached. "
            "Run `glossate info` to download."
        )


def run_preflight_checks(
    asr_model: str = "turbo",
    mt_model: str = "gemma-4-e4b",
    *,
    asr_backend: str = "auto",
    device: str | None = None,
) -> None:
    """Validate runtime prerequisites before inference.

    Raises typed internal exceptions; callers translate these to public
    GlossateError subclasses at the API boundary.

    Raises:
        _FfmpegMissingError: ffmpeg not on PATH.
        _ModelMissingError: a required model weight is not cached.
    """
    _check_ffmpeg()
    _check_whisper_model(asr_model, backend=asr_backend, device=device)
    _check_mt_model(mt_model)
