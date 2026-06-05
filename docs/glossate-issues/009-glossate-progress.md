---
number: 009
title: GlossateProgress (live stage display)
status: todo
type: AFK
blocked_by: ["006-ui-theme-ascii-components.md"]
---

# 009 — GlossateProgress (live stage display)

## Parent

`docs/glossate-prd.md`

## What to build

Implement `ui/progress.py` — the live terminal progress display for GLOSSATE's three-stage pipeline, mirroring ELUATE's `EluateProgress` exactly.

`GlossateProgress`:
- Rich `Live` panel showing stage bullets (pending ○ → active ● → complete ✓ → error ✗) for all three stages (Extract audio, Transcribe, Translate)
- Active stage shows a progress bar with `bouncingBall` spinner, `BarColumn`, `TaskProgressColumn`, `TimeRemainingColumn`
- 20 Hz refresh rate
- Methods: `start()`, `stop()`, `begin_stage(stage_id, total)`, `update_progress(completed, advance, detail, total)`, `complete_stage(stage_id)`, `error_stage(stage_id, message)`
- Stages skipped silently: Extract is skipped when input is already audio; Translate is skipped when no `--target` is given. Skipped stages go directly from PENDING to COMPLETE without ever being ACTIVE.
- `set_file_info(title, duration)` — sets file metadata displayed in the panel header

Include tests in `tests/test_progress.py` — verify stage transitions produce correct status values (no full Rich render required; test the state machine, not the pixels).

## Acceptance criteria

- [ ] `begin_stage("transcribe")` sets transcribe stage to ACTIVE and extract to COMPLETE (if skipped)
- [ ] `complete_stage("transcribe")` sets transcribe stage to COMPLETE
- [ ] `error_stage("translate", "model failed")` sets translate stage to ERROR
- [ ] Skipped stage transitions from PENDING directly to COMPLETE when `skip_stage(stage_id)` is called
- [ ] All three stages visible in the panel at all times (pending/active/complete/error)
- [ ] `start()` / `stop()` do not raise when called in sequence
- [ ] Stage state machine tested without requiring terminal; `pytest tests/test_progress.py` passes
- [ ] `mypy` clean on `ui/progress.py`

## Blocked by

- `006-ui-theme-ascii-components.md`
