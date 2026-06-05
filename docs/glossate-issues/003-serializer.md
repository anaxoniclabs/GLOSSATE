---
number: 003
title: Serializer (SRT / VTT / JSON)
status: todo
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 003 — Serializer (SRT / VTT / JSON)

## Parent

`docs/glossate-prd.md`

## What to build

Implement `core/serializer.py` — converts a list of `Cue` objects to SRT, VTT, or JSON files. No model dependencies, pure Python.

The three functions:
- `write_srt(cues, path)` — standard SubRip format: sequence number, `HH:MM:SS,mmm --> HH:MM:SS,mmm` timecode, text, blank line between entries.
- `write_vtt(cues, path)` — WebVTT format: `WEBVTT` header, `HH:MM:SS.mmm --> HH:MM:SS.mmm` timecodes (note dot not comma), text blocks.
- `write_json(cues, path)` — list of `{start, end, text, lang}` dicts, UTF-8, indented.

Also expose `to_srt_string`, `to_vtt_string`, `to_json_string` variants that return the content as a string (for testing without touching the filesystem).

Include comprehensive tests in `tests/test_serializer.py`.

## Acceptance criteria

- [ ] `write_srt` produces valid SubRip format (sequence numbers starting at 1, comma as decimal separator in timecodes)
- [ ] `write_vtt` produces valid WebVTT (starts with `WEBVTT`, dot as decimal separator)
- [ ] `write_json` round-trips: parsed JSON matches the original cue list
- [ ] Arabic and Turkish characters (including diacritics) are preserved in all three formats (UTF-8)
- [ ] Empty cue list produces a valid empty SRT/VTT (header only for VTT, empty file for SRT)
- [ ] `to_srt_string` / `to_vtt_string` / `to_json_string` return correct strings without writing files
- [ ] All cases covered by unit tests; `pytest tests/test_serializer.py` passes
- [ ] `mypy` clean on `core/serializer.py`

## Blocked by

- `001-project-scaffold.md`
