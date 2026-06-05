---
number: 002
title: Cue model + segmenter
status: todo
type: AFK
blocked_by: ["001-project-scaffold.md"]
---

# 002 — Cue model + segmenter

## Parent

`docs/glossate-prd.md`

## What to build

Define the `Cue` dataclass (the unit of data flowing through the entire pipeline) and implement `core/segmenter.py` — the module that converts raw Whisper output into a normalized list of `Cue` objects ready for serialization or translation.

Whisper returns segments (phrases/sentences) with word-level timestamps. The segmenter:
1. Accepts each Whisper segment as-is when its text is ≤ 42 characters.
2. Splits overlong segments at the first punctuation boundary (comma, period, semicolon, colon) closest to the midpoint.
3. Falls back to the nearest word boundary when no punctuation is found.
4. Uses the actual word timestamp at the split point as the cue boundary — never proportional estimation.
5. Recurses until all resulting cues are ≤ 42 characters.

The `Cue` type: `{start: float, end: float, text: str, lang: str}` where `start`/`end` are seconds.

Include comprehensive tests in `tests/test_segmenter.py`. No model dependencies — pure Python.

## Acceptance criteria

- [ ] `Cue` dataclass defined and importable from `glossate`
- [ ] Segments ≤ 42 chars pass through unchanged
- [ ] Segment > 42 chars with a comma splits at the comma
- [ ] Segment > 42 chars with no punctuation splits at the nearest word boundary to the midpoint
- [ ] Split boundary uses the actual word timestamp, not a proportional estimate
- [ ] Very long segments split recursively until all cues are ≤ 42 chars
- [ ] Single-word segment that exceeds 42 chars passes through as-is (cannot split)
- [ ] Empty Whisper output returns empty cue list
- [ ] `lang` field is propagated from Whisper's detected language onto every cue
- [ ] All cases covered by unit tests; `pytest tests/test_segmenter.py` passes
- [ ] `mypy` clean on `core/segmenter.py`

## Blocked by

- `001-project-scaffold.md`
