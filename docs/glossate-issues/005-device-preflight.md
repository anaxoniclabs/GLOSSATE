---
number: 005
title: Device detection + preflight checks
status: done
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 005 — Device detection + preflight checks

## Parent

`docs/glossate-prd.md`

## What to build

Implement `utils/device.py` and `utils/preflight.py`, mirroring ELUATE's equivalents.

`utils/device.py`:
- `get_optimal_device(override=None)` — detects MPS (Apple Silicon) > CUDA > CPU. Validates explicit overrides: `--device cuda` raises `ValueError` when CUDA is unavailable, `--device mps` raises when MPS is unavailable.
- `get_memory_info()` — parses `vm_stat` output on macOS, returns `{free_gb, active_gb, inactive_gb, wired_gb}`. Returns `{"error": ...}` on failure (never raises — used by `info` command).

`utils/preflight.py`:
- `run_preflight_checks()` — validates: ffmpeg is on PATH, required model weights are cached (Whisper + translation model). Raises typed internal exceptions on failure. These are translated to public exceptions at the API boundary.

Include tests in `tests/test_device.py` and `tests/test_preflight.py` using mocks — no real hardware dependency.

## Acceptance criteria

- [x] `get_optimal_device()` returns MPS on Apple Silicon when MPS is available
- [x] `get_optimal_device()` returns CPU when neither CUDA nor MPS is available
- [x] `get_optimal_device("cuda")` raises `ValueError` when CUDA is unavailable
- [x] `get_optimal_device("mps")` raises `ValueError` when MPS is unavailable
- [x] `get_optimal_device("cpu")` always returns CPU device
- [x] `get_memory_info()` returns a dict with `free_gb` key on macOS
- [x] `get_memory_info()` returns `{"error": ...}` rather than raising when `vm_stat` fails
- [x] `run_preflight_checks()` raises when ffmpeg is not on PATH
- [x] `run_preflight_checks()` raises when a required model is not cached
- [x] All cases covered by unit tests with mocked subprocess/filesystem; `pytest tests/test_device.py tests/test_preflight.py` passes
- [x] `mypy` clean on both files

## Blocked by

- `001-project-scaffold.md`
