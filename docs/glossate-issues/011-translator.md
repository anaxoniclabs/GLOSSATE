---
number: 011
title: Translator (mlx-lm, XML batch, retry)
status: todo
type: AFK
blocked_by: ["002-cue-model-segmenter.md"]
---

# 011 — Translator (mlx-lm, XML batch, retry)

## Parent

`docs/glossate-prd.md`

## What to build

Implement `core/translator.py` — wraps mlx-lm to translate a list of `Cue` objects from a source language to a target language, using batched XML-tagged prompts with retry and per-cue fallback.

`translate(cues: list[Cue], target: str, source: str | None = None) -> list[Cue]`:
- Splits cues into batches: max 20 cues OR max 2000 characters total per batch, whichever comes first.
- For each batch, builds an XML-tagged prompt: `<cue id="N">text</cue>` lines inside a translate instruction.
- Sends the prompt to Aya-Expanse-8B via mlx-lm.
- Parses the response with `re.findall(r'<cue id="(\d+)">(.*?)</cue>', response, re.DOTALL)`.
- If the parsed cue count does not match the input count: retries the full batch once.
- If the retry also fails: falls back to translating each cue individually (guaranteed 1:1 mapping).
- Returns a new list of `Cue` objects with translated text, original timestamps preserved, `lang` updated to the target language.
- Raises internal `MLXError(RuntimeError)` only when the model itself fails (not on parse failure — that triggers retry/fallback).

Include tests in `tests/test_translator.py` with mocked mlx-lm.

## Acceptance criteria

- [ ] Batch contains at most 20 cues
- [ ] Batch is split before 2000 characters even if fewer than 20 cues
- [ ] Prompt contains `<cue id="1">...</cue>` tags for each cue in the batch
- [ ] Successful response parsed into translated cues with original timestamps
- [ ] `lang` field of output cues is set to the target language
- [ ] Count mismatch triggers exactly one retry of the full batch
- [ ] Second count mismatch falls back to per-cue translation
- [ ] Per-cue fallback calls the model once per cue and produces 1:1 output
- [ ] `MLXError` raised when mlx-lm raises (not on parse failure)
- [ ] All cases covered by tests with mocked mlx-lm; `pytest tests/test_translator.py` passes
- [ ] `mypy` clean on `core/translator.py`

## Blocked by

- `002-cue-model-segmenter.md`
