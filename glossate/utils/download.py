# SPDX-License-Identifier: MIT
"""Model cache inspection and download for GLOSSATE."""

from __future__ import annotations

import os
from pathlib import Path


class DownloadError(RuntimeError):
    """Raised when model download fails."""


def _hf_hub_dir() -> Path:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    return hf_home / "hub"


def _model_cache_dir(model_id: str) -> Path:
    return _hf_hub_dir() / ("models--" + model_id.replace("/", "--"))


def _snapshot_path(model_id: str) -> Path | None:
    """Return the canonical snapshot path for a cached model, or None."""
    model_dir = _model_cache_dir(model_id)
    refs_main = model_dir / "refs" / "main"
    if refs_main.exists():
        commit_hash = refs_main.read_text().strip()
        snap = model_dir / "snapshots" / commit_hash
        if snap.exists() and any(snap.iterdir()):
            return snap
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    for snap in sorted(snapshots_dir.iterdir()):
        if snap.is_dir() and any(snap.iterdir()):
            return snap
    return None


def is_model_cached(model_id: str) -> bool:
    """Return True if model weights are present in the local HF cache."""
    return _snapshot_path(model_id) is not None


def get_model_size_mb(model_id: str) -> int | None:
    """Return cached model size in MB, or None if not cached."""
    blobs_dir = _model_cache_dir(model_id) / "blobs"
    if not blobs_dir.exists():
        return None
    total = sum(f.stat().st_size for f in blobs_dir.iterdir() if f.is_file())
    return total // (1024 * 1024)


def ensure_model(model_id: str) -> Path:
    """Return local snapshot path, downloading from HuggingFace if needed.

    Raises:
        DownloadError: if the download fails (huggingface_hub handles retries internally).
    """
    snapshot = _snapshot_path(model_id)
    if snapshot is not None:
        return snapshot
    try:
        import huggingface_hub

        local_dir = huggingface_hub.snapshot_download(model_id)
        return Path(local_dir)
    except Exception as exc:
        raise DownloadError(f"Failed to download model {model_id!r}: {exc}") from exc
