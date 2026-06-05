---
number: 001
title: Project scaffold with working info stub
status: done
type: AFK
blocked_by: []
---

# 001 — Project scaffold with working info stub

## Parent

`docs/glossate-prd.md`

## What to build

Set up the complete project skeleton so that `pip install -e .` works and `glossate info` runs end-to-end — printing two Rich panels (System + Models) with real device, ffmpeg, and memory data. This is the foundation every other slice builds on, and the info command makes it immediately demoable.

Specifically:
- `pyproject.toml` mirroring ELUATE's: name `glossate`, Python ≥ 3.10, same tool config sections (ruff, mypy, pytest, bandit), `dev` and `cpu` extras, OIDC trusted publishing config
- Full package skeleton: `glossate/__init__.py`, `__main__.py`, `api.py` (stubs), `cli.py` (argparse with `info` subcommand only), `core/__init__.py`, `ui/__init__.py`, `utils/__init__.py`
- `ui/theme.py` — B&W palette and Rich theme (`glossate.*` namespace), `BoxChars`, `Stages` (with the three GLOSSATE stages: extract, transcribe, translate)
- `ui/ascii_art.py` — `GLOSSATE` ANSI Shadow figlet logo + `GLOSSATE` subtitle, gradient styling, `get_header_panel()`
- `ui/components.py` — `info_panel()` (key-value Rich Panel, same as ELUATE)
- `utils/device.py` — `get_optimal_device()` (MPS > CUDA > CPU), `get_memory_info()` (vm_stat on macOS)
- `utils/ffmpeg.py` — `get_ffmpeg_version()` stub (runs `ffmpeg -version`, returns first line or None)
- `utils/paths.py` — `get_app_dir()` returning `~/Documents/GLOSSATE/`
- `cli.py` `cmd_info()` — prints header panel + System panel (device, memory, ffmpeg) + Models panel (Whisper status, translation model status, output dir)
- `.github/workflows/publish.yml` already exists — verify it references the correct package name
- `tests/conftest.py` scaffold

## Acceptance criteria

- [x] `pip install -e ".[dev]"` completes without errors
- [x] `glossate info` prints the header, System panel, and Models panel with real data
- [x] System panel shows correct device (mps on Apple Silicon), free memory, ffmpeg version or "not found"
- [x] Models panel shows "not downloaded" for both models on a clean machine
- [x] `glossate --help` lists available flags without error
- [x] `pytest` collects 0 tests and exits 0 (scaffold ready)
- [x] `ruff check glossate` exits 0
- [x] `mypy glossate` exits 0

## Blocked by

None — can start immediately.
