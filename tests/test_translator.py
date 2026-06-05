# SPDX-License-Identifier: MIT
"""Tests for glossate.core.translator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from glossate.core.segmenter import Cue
from glossate.core.translator import (
    MLXError,
    MLXModelState,
    TranslationModelState,
    _make_batches,
    format_prose,
    load_model,
    translate,
)


def _cue(text: str, start: float = 0.0, end: float = 1.0, lang: str = "en") -> Cue:
    return Cue(start=start, end=end, text=text, lang=lang)


def _ms() -> MLXModelState:
    """Return a dummy MLXModelState for tests."""
    return MLXModelState(model=MagicMock(), tokenizer=MagicMock())


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batch_max_20_cues() -> None:
    cues = [_cue(f"cue {i}") for i in range(21)]
    batches = _make_batches(cues)
    assert len(batches) == 2
    assert len(batches[0]) == 20
    assert len(batches[1]) == 1


def test_batch_splits_before_2000_chars() -> None:
    # 1500 + 600 = 2100 > 2000 → two separate batches
    cues = [_cue("a" * 1500), _cue("b" * 600)]
    batches = _make_batches(cues)
    assert len(batches) == 2
    assert batches[0] == [cues[0]]
    assert batches[1] == [cues[1]]


def test_single_cue_over_2000_chars_in_own_batch() -> None:
    cues = [_cue("x" * 3000)]
    batches = _make_batches(cues)
    assert len(batches) == 1


def test_empty_cues_returns_empty_batches() -> None:
    assert _make_batches([]) == []


def test_load_model_passes_cuda_options_to_transformers_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_load(hf_id: str, **kwargs: object) -> TranslationModelState:
        captured.update({"hf_id": hf_id, **kwargs})
        return TranslationModelState(model=None, tokenizer=None, backend="transformers")

    monkeypatch.setattr("glossate.core.translator._load_transformers", fake_load)

    load_model("gemma-4-e4b", device="cuda", backend="transformers", compute_type="float16")

    assert captured == {
        "hf_id": "google/gemma-4-E4B-it",
        "device": "cuda",
        "compute_type": "float16",
    }


def test_load_model_auto_routes_gemma_to_transformers_off_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA/CPU must use the HF transformers backend with the google/ weights,
    # not the Apple-only MLX path.
    captured: dict[str, object] = {}

    def fake_load(hf_id: str, **kwargs: object) -> TranslationModelState:
        captured["hf_id"] = hf_id
        return TranslationModelState(model=None, tokenizer=None, backend="transformers")

    monkeypatch.setattr("glossate.core.translator._load_transformers", fake_load)
    monkeypatch.setattr("glossate.utils.device.get_optimal_device", lambda d=None: "cuda")

    load_model("gemma-4-e4b", device="cuda")

    assert captured["hf_id"] == "google/gemma-4-E4B-it"


def test_load_model_auto_routes_gemma_to_mlx_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Apple Silicon stays on MLX with the mlx-community weights (MLX optional).
    captured: dict[str, object] = {}

    def fake_load(hf_id: str, **kwargs: object) -> TranslationModelState:
        captured["hf_id"] = hf_id
        return TranslationModelState(model=None, tokenizer=None, backend="mlx")

    monkeypatch.setattr("glossate.core.translator._load_mlx", fake_load)
    monkeypatch.setattr("glossate.utils.device.get_optimal_device", lambda d=None: "mps")

    load_model("gemma-4-e4b", device="mps")

    assert captured["hf_id"] == "mlx-community/gemma-4-e4b-it-8bit"


def test_transformers_translation_parses_batched_xml(monkeypatch: pytest.MonkeyPatch) -> None:
    # The transformers path must parse the XML cue reply and preserve order,
    # timestamps, and relabel the target language.
    state = TranslationModelState(model=MagicMock(), tokenizer=MagicMock(), backend="transformers")

    def fake_call(prompts: list[str], st: TranslationModelState) -> list[str]:
        assert len(prompts) == 1  # one batched prompt, not one call per cue
        return ['<cue id="1">merhaba</cue>\n<cue id="2">dünya</cue>']

    monkeypatch.setattr("glossate.core.translator._call_transformers", fake_call)

    cues = [_cue("hello", start=0.0, end=1.0), _cue("world", start=1.0, end=2.0)]
    out = translate(cues, "tr", source="en", model_state=state)

    assert [c.text for c in out] == ["merhaba", "dünya"]
    assert [c.lang for c in out] == ["tr", "tr"]
    assert [(c.start, c.end) for c in out] == [(0.0, 1.0), (1.0, 2.0)]


def test_translation_preserves_original_as_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # Translating must keep the transcription text/lang so one run can emit bilingual output.
    state = TranslationModelState(model=MagicMock(), tokenizer=MagicMock(), backend="transformers")
    monkeypatch.setattr(
        "glossate.core.translator._call_transformers",
        lambda prompts, st: ['<cue id="1">merhaba</cue>'],
    )

    cues = [_cue("hello", lang="en")]
    out = translate(cues, "tr", source="en", model_state=state)

    assert out[0].text == "merhaba"
    assert out[0].lang == "tr"
    assert out[0].source_text == "hello"
    assert out[0].source_lang == "en"


def test_transformers_translation_falls_back_per_cue_on_parse_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # If the batched reply can't be parsed twice, fall back to one prompt per cue.
    state = TranslationModelState(model=MagicMock(), tokenizer=MagicMock(), backend="transformers")
    calls: list[int] = []

    def fake_call(prompts: list[str], st: TranslationModelState) -> list[str]:
        calls.append(len(prompts))
        if len(prompts) == 1:
            return ["garbage with no cue tags"]
        return [f"translated-{i}" for i in range(len(prompts))]

    monkeypatch.setattr("glossate.core.translator._call_transformers", fake_call)

    cues = [_cue("a"), _cue("b")]
    out = translate(cues, "tr", source="en", model_state=state)

    # two failed single-prompt batch attempts, then one batched per-cue fallback
    assert calls == [1, 1, 2]
    assert [c.text for c in out] == ["translated-0", "translated-1"]


# ---------------------------------------------------------------------------
# Successful translation
# ---------------------------------------------------------------------------


def test_successful_batch_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello"), _cue("World")]

    monkeypatch.setattr(
        "glossate.core.translator._call_mlx",
        lambda p, ms: '<cue id="1">Merhaba</cue>\n<cue id="2">Dünya</cue>',
    )
    result = translate(cues, "tr", model_state=_ms())

    assert [c.text for c in result] == ["Merhaba", "Dünya"]


def test_lang_updated_to_target(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello", lang="en")]

    monkeypatch.setattr(
        "glossate.core.translator._call_mlx",
        lambda p, ms: '<cue id="1">Merhaba</cue>',
    )
    result = translate(cues, "tr", model_state=_ms())

    assert result[0].lang == "tr"


def test_timestamps_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello", start=1.5, end=3.0)]

    monkeypatch.setattr(
        "glossate.core.translator._call_mlx",
        lambda p, ms: '<cue id="1">Merhaba</cue>',
    )
    result = translate(cues, "tr", model_state=_ms())

    assert result[0].start == 1.5
    assert result[0].end == 3.0


# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------


def test_prompt_contains_xml_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello"), _cue("World")]
    captured: list[str] = []

    def fake(prompt: str, ms: object) -> str:
        captured.append(prompt)
        return '<cue id="1">Merhaba</cue>\n<cue id="2">Dünya</cue>'

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    translate(cues, "tr", model_state=_ms())

    assert '<cue id="1">Hello</cue>' in captured[0]
    assert '<cue id="2">World</cue>' in captured[0]


def test_source_hint_included_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello")]
    captured: list[str] = []

    def fake(prompt: str, ms: object) -> str:
        captured.append(prompt)
        return '<cue id="1">Merhaba</cue>'

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    translate(cues, "tr", source="en", model_state=_ms())

    assert "from en" in captured[0]


# ---------------------------------------------------------------------------
# Retry on count mismatch
# ---------------------------------------------------------------------------


def test_count_mismatch_triggers_exactly_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello"), _cue("World")]
    call_count = 0

    def fake(prompt: str, ms: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '<cue id="1">Merhaba</cue>'  # 1 match instead of 2
        return '<cue id="1">Merhaba</cue>\n<cue id="2">Dünya</cue>'

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    result = translate(cues, "tr", model_state=_ms())

    assert call_count == 2
    assert [c.text for c in result] == ["Merhaba", "Dünya"]


# ---------------------------------------------------------------------------
# Fallback to per-cue on second mismatch
# ---------------------------------------------------------------------------


def test_second_mismatch_falls_back_per_cue(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello"), _cue("World")]
    call_count = 0

    def fake(prompt: str, ms: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return '<cue id="1">only one</cue>'  # always mismatched for 2-cue batch
        return "translated"

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    result = translate(cues, "tr", model_state=_ms())

    assert call_count == 4  # 2 batch attempts + 2 per-cue
    assert len(result) == 2
    assert all(c.lang == "tr" for c in result)


def test_per_cue_fallback_calls_model_once_per_cue(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 3
    cues = [_cue(f"cue {i}") for i in range(n)]
    call_count = 0

    def fake(prompt: str, ms: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return ""  # empty → 0 matches → mismatch for any non-empty batch
        return f"translated {call_count - 2}"

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    result = translate(cues, "tr", model_state=_ms())

    assert call_count == 2 + n
    assert len(result) == n


def test_per_cue_fallback_produces_1to1_output(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("A"), _cue("B"), _cue("C")]
    responses = ["X", "Y", "Z"]
    call_count = 0

    def fake(prompt: str, ms: object) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return ""
        return responses[call_count - 3]

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)
    result = translate(cues, "tr", model_state=_ms())

    assert [c.text for c in result] == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# MLXError
# ---------------------------------------------------------------------------


def test_mlx_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    cues = [_cue("Hello")]

    monkeypatch.setattr(
        "glossate.core.translator._call_mlx",
        lambda p, ms: (_ for _ in ()).throw(MLXError("model failed")),
    )
    with pytest.raises(MLXError):
        translate(cues, "tr", model_state=_ms())


def test_mlx_error_not_swallowed_on_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLXError must propagate even on the first call (not trigger fallback)."""
    cues = [_cue("Hello"), _cue("World")]
    call_count = 0

    def fake(prompt: str, ms: object) -> str:
        nonlocal call_count
        call_count += 1
        raise MLXError("GPU unavailable")

    monkeypatch.setattr("glossate.core.translator._call_mlx", fake)

    with pytest.raises(MLXError):
        translate(cues, "tr", model_state=_ms())

    assert call_count == 1  # no retry when model itself fails


# ---------------------------------------------------------------------------
# format_prose — the Markdown notes formatting pass (reformat, not translate)
# ---------------------------------------------------------------------------


def test_format_prose_transformers_reflows_with_larger_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The format pass needs a bigger generation budget than per-cue translation,
    # or Gemma truncates a paragraph mid-sentence (silent content loss).
    captured: dict[str, object] = {}
    state = TranslationModelState(model=MagicMock(), tokenizer=MagicMock(), backend="transformers")

    def fake_call(prompts, st, *, max_new_tokens=1024):  # noqa: ANN001, ANN202
        captured["max_new_tokens"] = max_new_tokens
        captured["prompt"] = prompts[0]
        return ["Clean, readable prose."]

    monkeypatch.setattr("glossate.core.translator._call_transformers", fake_call)

    out = format_prose("hello   world fragment", "en", model_state=state)

    assert out == "Clean, readable prose."
    assert captured["max_new_tokens"] > 1024
    assert "Do NOT translate" in captured["prompt"]  # fidelity instruction is pinned


def test_format_prose_empty_input_skips_model() -> None:
    called: list[int] = []
    state = TranslationModelState(model=MagicMock(), tokenizer=MagicMock(), backend="transformers")
    # No monkeypatch: if it tried to call the model it would blow up on the MagicMock.
    assert format_prose("   ", "en", model_state=state) == ""
    assert called == []
