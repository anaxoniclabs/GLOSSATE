# SPDX-License-Identifier: MIT
"""Tests for glossate.core.notes (Markdown orchestration).

The Gemma calls are stubbed: ``format_prose`` is tagged ``[lang]…`` and
``translate`` is tagged ``T:…`` so assertions can prove *which* pass ran and in
*which* language — the routing and timestamp/sentence logic is what matters here.
"""

from __future__ import annotations

import pytest

from glossate.core import notes
from glossate.core import translator as _translator
from glossate.core.segmenter import Cue


def _cue(text: str, start: float = 0.0, end: float = 1.0, lang: str = "en") -> Cue:
    return Cue(start=start, end=end, text=text, lang=lang)


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _translator.TranslationModelState:
    """Stub the two Gemma passes; record format_prose invocations."""
    calls: list[tuple[str, str]] = []

    def fake_format(text: str, lang: str, *, model_state):  # noqa: ANN001
        calls.append((lang, text))
        return f"[{lang}]{text}"

    def fake_translate(cues, target, source=None, *, model_state, on_progress=None):  # noqa: ANN001
        return [
            Cue(
                start=c.start,
                end=c.end,
                text=f"T:{c.text}",
                lang=target,
                source_text=c.text,
                source_lang=source or c.lang,
            )
            for c in cues
        ]

    monkeypatch.setattr(_translator, "format_prose", fake_format)
    monkeypatch.setattr(_translator, "translate", fake_translate)
    state = _translator.TranslationModelState(model=None, tokenizer=None, backend="transformers")
    state._format_calls = calls  # type: ignore[attr-defined]
    return state


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_groups_under_char_budget() -> None:
    cues = [_cue("a" * 30, start=float(i)) for i in range(4)]
    windows = notes._chunk(cues, notes._cue_text, max_chars=50)
    # 30+30 > 50, so each window holds one cue → 4 windows.
    assert len(windows) == 4
    assert windows[0][0] == 0.0  # start time of first cue in window
    assert windows[1][0] == 1.0


def test_chunk_packs_multiple_cues_when_they_fit() -> None:
    cues = [_cue("hi", start=0.0), _cue("there", start=1.0)]
    windows = notes._chunk(cues, notes._cue_text, max_chars=100)
    assert len(windows) == 1
    assert windows[0] == (0.0, "hi there")


def test_chunk_skips_empty_text() -> None:
    cues = [_cue("  ", start=0.0), _cue("real", start=1.0)]
    windows = notes._chunk(cues, notes._cue_text, max_chars=100)
    assert windows == [(1.0, "real")]


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


def test_sentences_merge_fragments_until_terminator() -> None:
    cues = [_cue("This is", start=0.0, end=1.0), _cue("a sentence.", start=1.0, end=2.0)]
    sents = notes._sentences(cues, "en")
    assert len(sents) == 1
    assert sents[0].text == "This is a sentence."
    assert sents[0].start == 0.0
    assert sents[0].end == 2.0


def test_sentences_split_on_each_terminator() -> None:
    cues = [_cue("One. Two?", start=0.0, end=1.0), _cue("Three!", start=1.0, end=2.0)]
    sents = notes._sentences(cues, "en")
    assert [s.text for s in sents] == ["One.", "Two?", "Three!"]


def test_sentences_flush_trailing_unterminated_text() -> None:
    cues = [_cue("no period here", start=0.0, end=1.0)]
    sents = notes._sentences(cues, "en")
    assert [s.text for s in sents] == ["no period here"]


def test_sentences_do_not_split_decimals() -> None:
    # True sentence pairs were chosen for grammatical quality — a decimal must
    # not be mistaken for a sentence boundary.
    cues = [_cue("It costs 3.5 dollars today.", start=0.0, end=2.0)]
    sents = notes._sentences(cues, "en")
    assert [s.text for s in sents] == ["It costs 3.5 dollars today."]


def test_sentences_do_not_split_period_followed_by_non_space() -> None:
    # A period inside a token (e.g. a domain) is not a boundary; only a
    # terminator followed by whitespace is. (Abbrevs like "Dr. " still over-split.)
    cues = [_cue("Visit google.com now.", start=0.0, end=2.0)]
    sents = notes._sentences(cues, "en")
    assert [s.text for s in sents] == ["Visit google.com now."]


# ---------------------------------------------------------------------------
# build_markdown routing
# ---------------------------------------------------------------------------


def test_level1_monolingual_formats_source_no_translate(fake_model) -> None:  # noqa: ANN001
    cues = [_cue("hello world", lang="en")]
    md = notes.build_markdown(cues, source_lang="en", target_lang=None, model_state=fake_model)
    assert "[en]hello world" in md
    # Source language was formatted; no translation tag present.
    assert "T:" not in md


def test_translated_only_translates_then_formats_target(fake_model) -> None:  # noqa: ANN001
    cues = [_cue("hello", lang="en")]
    md = notes.build_markdown(
        cues, source_lang="en", target_lang="tr", scope="translated", model_state=fake_model
    )
    # Prose is formatted in the TARGET language, over translated text.
    assert "[tr]T:hello" in md


def test_two_prose_has_both_sections(fake_model) -> None:  # noqa: ANN001
    cues = [_cue("hello", lang="en")]
    md = notes.build_markdown(
        cues, source_lang="en", target_lang="tr", scope="both", layout="two-prose",
        model_state=fake_model,
    )
    assert "## Original (English)" in md
    assert "## Translation (Turkish)" in md
    assert "[en]hello" in md       # original formatted in source lang
    assert "[tr]T:hello" in md     # translation formatted in target lang


def test_dual_stack_pairs_sentences_without_prose_format(fake_model) -> None:  # noqa: ANN001
    cues = [_cue("Hello there.", lang="en")]
    md = notes.build_markdown(
        cues, source_lang="en", target_lang="tr", scope="both", layout="dual-stack",
        model_state=fake_model, timestamps=False,
    )
    assert "Hello there." in md          # source sentence
    assert "> T:Hello there." in md      # translated sentence as blockquote
    # dual-stack is deterministic: the prose format pass must NOT run.
    assert fake_model._format_calls == []


def test_empty_cues_returns_empty_string(fake_model) -> None:  # noqa: ANN001
    assert notes.build_markdown([], source_lang="en", target_lang=None, model_state=fake_model) == ""
