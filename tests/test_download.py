# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

from glossate.utils.download import (
    DownloadError,
    ensure_model,
    get_model_size_mb,
    is_model_cached,
)

MODEL_ID = "mlx-community/whisper-large-v3-mlx"
AYA_MODEL_ID = "mlx-community/aya-expanse-8b-4bit"


def _make_model_cache(
    hub_dir: Path,
    model_id: str,
    *,
    size_bytes: int = 1024,
    commit: str = "abc123",
) -> Path:
    """Create a minimal HF hub cache structure and return the snapshot dir."""
    dir_name = "models--" + model_id.replace("/", "--")
    model_dir = hub_dir / dir_name
    blobs_dir = model_dir / "blobs"
    snapshot_dir = model_dir / "snapshots" / commit
    refs_dir = model_dir / "refs"

    blobs_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    refs_dir.mkdir(parents=True)

    blob = blobs_dir / "sha256-deadbeef"
    blob.write_bytes(b"x" * size_bytes)
    (snapshot_dir / "model.safetensors").symlink_to(blob)
    (refs_dir / "main").write_text(commit)

    return snapshot_dir


@pytest.fixture
def hub_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    hub = tmp_path / "hub"
    hub.mkdir()
    return hub


# --- is_model_cached ---


def test_is_model_cached_false_when_absent(hub_dir: Path) -> None:
    assert is_model_cached(MODEL_ID) is False


def test_is_model_cached_true_when_present(hub_dir: Path) -> None:
    _make_model_cache(hub_dir, MODEL_ID)
    assert is_model_cached(MODEL_ID) is True


def test_is_model_cached_false_empty_snapshots(hub_dir: Path) -> None:
    dir_name = "models--" + MODEL_ID.replace("/", "--")
    (hub_dir / dir_name / "snapshots").mkdir(parents=True)
    assert is_model_cached(MODEL_ID) is False


def test_is_model_cached_aya_model(hub_dir: Path) -> None:
    _make_model_cache(hub_dir, AYA_MODEL_ID)
    assert is_model_cached(AYA_MODEL_ID) is True


# --- ensure_model ---


def test_ensure_model_returns_snapshot_path_when_cached(hub_dir: Path) -> None:
    snapshot = _make_model_cache(hub_dir, MODEL_ID)
    result = ensure_model(MODEL_ID)
    assert result == snapshot


def test_ensure_model_calls_download_when_not_cached(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_path = hub_dir / "downloaded"
    fake_path.mkdir()
    calls: list[str] = []

    def fake_snapshot_download(model_id: str, **kwargs: object) -> str:
        calls.append(model_id)
        return str(fake_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = ensure_model(MODEL_ID)
    assert calls == [MODEL_ID]
    assert result == fake_path


def test_ensure_model_raises_download_error_on_failure(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    def bad_download(model_id: str, **kwargs: object) -> str:
        raise RuntimeError("network error")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", bad_download)
    with pytest.raises(DownloadError, match="network error"):
        ensure_model(MODEL_ID)


def test_ensure_model_skips_download_when_cached(
    hub_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_model_cache(hub_dir, MODEL_ID)
    calls: list[str] = []

    def should_not_be_called(model_id: str, **kwargs: object) -> str:
        calls.append(model_id)
        return ""

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", should_not_be_called)
    ensure_model(MODEL_ID)
    assert calls == []


# --- get_model_size_mb ---


def test_get_model_size_mb_none_when_not_cached(hub_dir: Path) -> None:
    assert get_model_size_mb(MODEL_ID) is None


def test_get_model_size_mb_correct_for_5mb(hub_dir: Path) -> None:
    _make_model_cache(hub_dir, MODEL_ID, size_bytes=5 * 1024 * 1024)
    assert get_model_size_mb(MODEL_ID) == 5


def test_get_model_size_mb_rounds_down(hub_dir: Path) -> None:
    _make_model_cache(hub_dir, MODEL_ID, size_bytes=1_500_000)
    result = get_model_size_mb(MODEL_ID)
    assert result == 1


def test_get_model_size_mb_returns_int(hub_dir: Path) -> None:
    _make_model_cache(hub_dir, MODEL_ID, size_bytes=2 * 1024 * 1024)
    result = get_model_size_mb(MODEL_ID)
    assert isinstance(result, int)
