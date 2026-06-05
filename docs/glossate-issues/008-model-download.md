---
number: 008
title: Model download + cache check
status: todo
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 008 — Model download + cache check

## Parent

`docs/glossate-prd.md`

## What to build

Implement `utils/download.py` — checks whether model weights are cached and downloads them on first use via `huggingface_hub`.

`ensure_model(model_id: str) -> Path`:
- Checks the huggingface cache for the requested model.
- If cached, returns the local path immediately.
- If not cached, triggers a download with a progress indicator.
- Raises internal `DownloadError(RuntimeError)` if the download fails after retries.

`is_model_cached(model_id: str) -> bool`:
- Returns True if the model weights are present locally, False otherwise.
- Used by `glossate info` to display cache status without triggering a download.

`get_model_size_mb(model_id: str) -> int | None`:
- Returns the size in MB of the cached model, or None if not cached.
- Used by `glossate info` for the "installed (X MB)" display.

Model IDs to support: Whisper large-v3 (via mlx-whisper's expected cache path) and Aya-Expanse-8B (via huggingface-hub).

Include tests in `tests/test_download.py` with mocked `huggingface_hub` calls.

## Acceptance criteria

- [ ] `is_model_cached()` returns True when model files are present at the expected cache path
- [ ] `is_model_cached()` returns False when model files are absent
- [ ] `ensure_model()` returns path immediately when model is cached
- [ ] `ensure_model()` calls huggingface-hub download when model is not cached
- [ ] `ensure_model()` raises `DownloadError` when download fails
- [ ] `get_model_size_mb()` returns correct MB size from cached files
- [ ] `get_model_size_mb()` returns None when model is not cached
- [ ] All cases covered by tests with mocked hub; `pytest tests/test_download.py` passes
- [ ] `mypy` clean on `utils/download.py`

## Blocked by

- `001-project-scaffold.md`
