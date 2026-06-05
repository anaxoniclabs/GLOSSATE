---
number: 004
title: Output path construction + collision auto-increment
status: todo
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 004 — Output path construction + collision auto-increment

## Parent

`docs/glossate-prd.md`

## What to build

Implement the output path logic in `utils/paths.py`. This module owns two responsibilities: constructing the default output path and handling filename collisions without overwriting existing files.

Default path: `~/Documents/GLOSSATE/YYYY-MM-DD/<stem>.<ext>` where stem is derived from the input filename and ext maps from the chosen format (srt, vtt, json).

Collision policy: if `lecture.srt` exists, try `lecture.1.srt`, then `lecture.2.srt`, and so on until a free slot is found.

The `-o` flag in the CLI bypasses this entirely — when an explicit output path is given, use it as-is (no date folder, no auto-increment).

Create the date directory automatically on first write if it does not exist.

Include comprehensive tests in `tests/test_paths.py` — these must not write to the real `~/Documents/GLOSSATE/` directory; use `tmp_path` pytest fixtures.

## Acceptance criteria

- [ ] Default output path includes today's date folder (`~/Documents/GLOSSATE/YYYY-MM-DD/`)
- [ ] Stem is derived from the input filename (e.g. `lecture.mp4` → `lecture`)
- [ ] Format maps to correct extension (srt → `.srt`, vtt → `.vtt`, json → `.json`)
- [ ] First call produces `<stem>.<ext>` with no numeric suffix
- [ ] Second call on same stem+format produces `<stem>.1.<ext>`
- [ ] Third call produces `<stem>.2.<ext>` (not `<stem>.1.1.<ext>`)
- [ ] Explicit `-o` path bypasses default path and auto-increment entirely
- [ ] Date folder is created if it does not exist
- [ ] All cases covered by unit tests using `tmp_path`; `pytest tests/test_paths.py` passes
- [ ] `mypy` clean on `utils/paths.py`

## Blocked by

- `001-project-scaffold.md`
