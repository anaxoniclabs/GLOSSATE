---
number: 006
title: UI theme + ASCII art + info panel component
status: todo
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 006 — UI theme + ASCII art + info panel component

## Parent

`docs/glossate-prd.md`

## What to build

Complete `ui/theme.py`, `ui/ascii_art.py`, and `ui/components.py` — the full Rich UI layer, mirroring ELUATE's equivalents with GLOSSATE-specific content.

`ui/theme.py`:
- `Colors` — B&W palette identical to ELUATE (`PRIMARY = "#FFFFFF"`, grays, `ERROR = "#FF6B6B"`)
- `GLOSSATE_THEME` — Rich Theme with `glossate.*` namespace (same keys as ELUATE's `eluate.*`)
- `BoxChars` — same Unicode box-drawing characters as ELUATE
- `Stages` — three stage tuples: `EXTRACT = ("extract", "Extracting audio", "🎵")`, `TRANSCRIBE = ("transcribe", "Transcribing", "🎙")`, `TRANSLATE = ("translate", "Translating", "🌐")`, plus `ALL` list

`ui/ascii_art.py`:
- `GLOSSATE_LOGO` — ANSI Shadow figlet for `GLOSSATE`
- `get_styled_logo()` — applies white-to-gray gradient across lines, same pattern as ELUATE
- `get_header_panel(console_width=80)` — centers logo in a Rich Panel, adds `GLOSSATE` subtitle below the figlet in `Colors.TEXT_SECONDARY`
- `GLOSSATE_LOGO_SMALL` — thin box-drawing small variant

`ui/components.py`:
- `info_panel(title, content, style)` — identical interface to ELUATE's: Rich Panel with a `Table.grid` of key-value pairs

Include render tests in `tests/test_ui.py` — verify panels render without exception and contain expected text.

## Acceptance criteria

- [ ] `get_header_panel()` renders without exception
- [ ] Header panel contains `GLOSSATE` (ASCII art) and `GLOSSATE` (subtitle)
- [ ] `info_panel("System", {"Device": "mps"})` returns a Rich `Panel` containing "Device" and "mps"
- [ ] `Colors.PRIMARY` is `"#FFFFFF"` (white)
- [ ] `Stages.ALL` contains exactly three stages with ids `extract`, `transcribe`, `translate`
- [ ] `GLOSSATE_THEME` contains `glossate.primary` style
- [ ] All cases covered by render tests; `pytest tests/test_ui.py` passes
- [ ] `mypy` clean on all three files

## Blocked by

- `001-project-scaffold.md`
