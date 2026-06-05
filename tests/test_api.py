# SPDX-License-Identifier: MIT
"""Integration tests for glossate.api — exception translation and Session lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import glossate
from glossate.api import (
    AudioExtractionError,
    GlossateError,
    ModelNotInstalledError,
    Session,
    SubtitleBurnError,
    TranscriptionError,
    TranslationError,
    burn_subtitles,
    subtitle,
    subtitle_video,
    transcribe,
    translate,
)
from glossate.core.segmenter import Cue
from glossate.core.transcriber import WhisperResult, WhisperSegment, WhisperWord
from glossate.core.translator import MLXModelState

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _cue(text: str = "Hello", start: float = 0.0, end: float = 1.0, lang: str = "en") -> Cue:
    return Cue(start=start, end=end, text=text, lang=lang)


def _whisper_result(lang: str = "en") -> WhisperResult:
    return WhisperResult(
        segments=[
            WhisperSegment(
                start=0.0,
                end=1.0,
                text="Hello",
                words=[WhisperWord(word="Hello", start=0.0, end=1.0)],
            )
        ],
        detected_language=lang,
    )


@pytest.fixture()
def no_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent real model calls from load_model."""
    monkeypatch.setattr(
        "glossate.core.translator.load_model",
        lambda *a, **kw: MLXModelState(model=None, tokenizer=None),
    )


@pytest.fixture(autouse=True)
def fake_asr_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route Session's ASR worker through the function-level test seam."""

    class FakeWhisperSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> WhisperResult:
            from glossate.core.transcriber import transcribe as _transcribe

            return _transcribe(*args, **kwargs)

        def close(self) -> None:
            pass

    monkeypatch.setattr("glossate.core.transcriber.WhisperSession", FakeWhisperSession)


@pytest.fixture()
def mock_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None) -> dict[str, Any]:
    """Stub every internal pipeline step; return shared objects for assertions."""
    cues = [_cue("Hello")]
    translated = [_cue("Merhaba", lang="tr")]
    out_path = tmp_path / "output.srt"

    monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
    monkeypatch.setattr("glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result())
    monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: cues)
    monkeypatch.setattr("glossate.core.translator.translate", lambda c, t, **kw: translated)
    monkeypatch.setattr("glossate.core.serializer.write_srt", lambda c, p: out_path)
    monkeypatch.setattr(
        "glossate.utils.paths.default_output_path", lambda *a, **kw: out_path
    )

    return {"cues": cues, "translated": translated, "out_path": out_path}


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_glossate_error_is_exception(self) -> None:
        assert issubclass(GlossateError, Exception)

    def test_all_subclass_glossate_error(self) -> None:
        for cls in (
            ModelNotInstalledError,
            AudioExtractionError,
            SubtitleBurnError,
            TranscriptionError,
            TranslationError,
        ):
            assert issubclass(cls, GlossateError), f"{cls.__name__} must subclass GlossateError"

    def test_all_exported_from_package(self) -> None:
        for name in (
            "AudioExtractionError",
            "SubtitleBurnError",
            "TranscriptionError",
            "TranslationError",
            "ModelNotInstalledError",
            "GlossateError",
        ):
            assert hasattr(glossate, name)


# ---------------------------------------------------------------------------
# Exception translation
# ---------------------------------------------------------------------------


class TestExceptionTranslation:
    def test_ffmpeg_error_becomes_audio_extraction_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        from glossate.utils.ffmpeg import FFmpegError

        def fail(p: Path, **kw: Any) -> Path:
            raise FFmpegError("ffmpeg died")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(AudioExtractionError, match="ffmpeg died"):
                s.transcribe(tmp_path / "video.mp4")

    def test_ffmpeg_timeout_becomes_audio_extraction_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        from glossate.utils.ffmpeg import FFmpegTimeout

        def fail(p: Path, **kw: Any) -> Path:
            raise FFmpegTimeout("timed out")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(AudioExtractionError, match="timed out"):
                s.transcribe(tmp_path / "video.mp4")

    def test_whisper_error_becomes_transcription_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        from glossate.core.transcriber import WhisperError

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: (_ for _ in ()).throw(WhisperError("OOM")),
        )
        with Session() as s:
            with pytest.raises(TranscriptionError, match="OOM"):
                s.transcribe(tmp_path / "audio.wav")

    def test_mlx_error_becomes_translation_error(
        self, monkeypatch: pytest.MonkeyPatch, no_model_load: None
    ) -> None:
        from glossate.core.translator import MLXError

        monkeypatch.setattr(
            "glossate.core.translator.translate",
            lambda *a, **kw: (_ for _ in ()).throw(MLXError("gpu fail")),
        )
        with Session() as s:
            with pytest.raises(TranslationError, match="gpu fail"):
                s.translate([_cue()], target="tr")

    def test_download_error_becomes_model_not_installed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        from glossate.utils.download import DownloadError

        def fail(p: Path, **kw: Any) -> Path:
            raise DownloadError("weights missing")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(ModelNotInstalledError, match="weights missing"):
                s.transcribe(tmp_path / "video.mp4")

    def test_file_not_found_propagates_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        def fail(p: Path, **kw: Any) -> Path:
            raise FileNotFoundError("no such file")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(FileNotFoundError):
                s.transcribe(tmp_path / "video.mp4")

    def test_permission_error_propagates_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        def fail(p: Path, **kw: Any) -> Path:
            raise PermissionError("access denied")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(PermissionError):
                s.transcribe(tmp_path / "video.mp4")

    def test_file_not_found_is_not_wrapped_in_glossate_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        def fail(p: Path, **kw: Any) -> Path:
            raise FileNotFoundError("no such file")

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", fail)
        with Session() as s:
            with pytest.raises(FileNotFoundError) as exc_info:
                s.transcribe(tmp_path / "video.mp4")
        assert not isinstance(exc_info.value, GlossateError)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_enter_returns_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "glossate.core.translator.load_model",
            lambda *a, **kw: MLXModelState(model=None, tokenizer=None),
        )
        s = Session()
        result = s.__enter__()
        s.__exit__(None, None, None)
        assert result is s

    def test_enter_does_not_call_load_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "glossate.core.translator.load_model",
            lambda *a, **kw: (calls.append("load"), MLXModelState(model=None, tokenizer=None))[1],
        )
        with Session():
            pass
        assert calls == []

    def test_exit_clears_model_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "glossate.core.translator.load_model",
            lambda *a, **kw: MLXModelState(model=None, tokenizer=None),
        )
        monkeypatch.setattr("glossate.core.translator.translate", lambda *a, **kw: [])
        s = Session()
        s.__enter__()
        s.translate([_cue()], target="tr")
        assert s._mt_state is not None
        s.__exit__(None, None, None)
        assert s._mt_state is None

    def test_exit_clears_model_state_on_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "glossate.core.translator.load_model",
            lambda *a, **kw: MLXModelState(model=None, tokenizer=None),
        )
        monkeypatch.setattr("glossate.core.translator.translate", lambda *a, **kw: [])
        s = Session()
        with pytest.raises(RuntimeError):
            with s:
                s.translate([_cue()], target="tr")
                raise RuntimeError("boom")
        assert s._mt_state is None

    def test_load_error_translated_to_translation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from glossate.core.translator import MLXError

        def fail(*a: Any, **kw: Any) -> None:
            raise MLXError("load failed")

        monkeypatch.setattr("glossate.core.translator.load_model", fail)
        with pytest.raises(TranslationError):
            with Session() as s:
                s.translate([_cue()], target="tr")


# ---------------------------------------------------------------------------
# Session.transcribe
# ---------------------------------------------------------------------------


class TestSessionTranscribe:
    def test_returns_list_of_cues(
        self, tmp_path: Path, mock_pipeline: dict[str, Any]
    ) -> None:
        with Session() as s:
            result = s.transcribe(tmp_path / "audio.wav")
        assert result == mock_pipeline["cues"]

    def test_calls_extract_then_transcribe_then_segment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        order: list[str] = []
        audio_file = tmp_path / "audio.wav"

        monkeypatch.setattr(
            "glossate.utils.ffmpeg.extract_audio",
            lambda p, **kw: (order.append("extract"), audio_file)[1],
        )
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: (order.append("transcribe"), _whisper_result())[1],
        )
        monkeypatch.setattr(
            "glossate.core.segmenter.segment",
            lambda d: (order.append("segment"), [_cue()])[1],
        )
        with Session() as s:
            s.transcribe(audio_file)

        assert order == ["extract", "transcribe", "segment"]

    def test_second_transcribe_served_from_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The expensive ASR step must run once; the cache serves the re-run.
        monkeypatch.delenv("GLOSSATE_NO_CACHE", raising=False)
        monkeypatch.setenv("GLOSSATE_CACHE_DIR", str(tmp_path / "cache"))
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"data")

        asr_calls: list[int] = []
        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: (asr_calls.append(1), _whisper_result())[1],
        )

        with Session() as s:
            first = s.transcribe(audio)
            second = s.transcribe(audio)

        assert len(asr_calls) == 1
        assert [c.text for c in second] == [c.text for c in first]

    def test_temp_file_cleaned_up_after_transcribe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        temp_audio = tmp_path / "temp.wav"
        temp_audio.touch()
        input_video = tmp_path / "video.mp4"

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: temp_audio)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: _whisper_result(),
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])

        with Session() as s:
            s.transcribe(input_video)

        assert not temp_audio.exists()

    def test_temp_file_cleaned_up_on_transcribe_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        from glossate.core.transcriber import WhisperError

        temp_audio = tmp_path / "temp.wav"
        temp_audio.touch()
        input_video = tmp_path / "video.mp4"

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: temp_audio)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: (_ for _ in ()).throw(WhisperError("crash")),
        )

        with Session() as s:
            with pytest.raises(TranscriptionError):
                s.transcribe(input_video)

        assert not temp_audio.exists()

    def test_audio_input_not_deleted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        audio_file = tmp_path / "audio.wav"
        audio_file.touch()

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: _whisper_result(),
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])

        with Session() as s:
            s.transcribe(audio_file)

        assert audio_file.exists()

    def test_constructs_asr_session_with_cuda_options(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        captured: list[dict[str, Any]] = []

        class CapturingWhisperSession:
            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs)

            def transcribe(self, *args: Any, **kwargs: Any) -> WhisperResult:
                return _whisper_result()

            def close(self) -> None:
                pass

        monkeypatch.setattr("glossate.core.transcriber.WhisperSession", CapturingWhisperSession)
        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])

        with Session(device="cuda", asr_backend="faster-whisper", compute_type="float16") as s:
            s.transcribe(tmp_path / "audio.wav")

        assert captured == [
            {"backend": "faster-whisper", "device": "cuda", "compute_type": "float16"}
        ]


# ---------------------------------------------------------------------------
# Session.translate
# ---------------------------------------------------------------------------


class TestSessionTranslate:
    def test_delegates_to_translator(
        self, monkeypatch: pytest.MonkeyPatch, no_model_load: None
    ) -> None:
        cues = [_cue("Hello")]
        translated = [_cue("Merhaba", lang="tr")]
        monkeypatch.setattr("glossate.core.translator.translate", lambda c, t, **kw: translated)

        with Session() as s:
            result = s.translate(cues, target="tr")

        assert result == translated

    def test_passes_target_language(
        self, monkeypatch: pytest.MonkeyPatch, no_model_load: None
    ) -> None:
        captured: list[str] = []
        monkeypatch.setattr(
            "glossate.core.translator.translate",
            lambda c, t, **kw: (captured.append(t), [_cue()])[1],
        )
        with Session() as s:
            s.translate([_cue()], target="de")

        assert captured == ["de"]

    def test_translate_outside_context_raises(self) -> None:
        s = Session()
        with pytest.raises(RuntimeError, match="Session is not active"):
            s.translate([_cue()], target="tr")

    def test_subtitle_with_target_outside_context_raises(self, tmp_path: Path) -> None:
        s = Session()
        with pytest.raises(RuntimeError, match="Session is not active"):
            s.subtitle(tmp_path / "audio.wav", target="tr")

    def test_loads_translation_model_with_cuda_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[dict[str, Any]] = []

        def fake_load(model: str, **kwargs: Any) -> MLXModelState:
            captured.append({"model": model, **kwargs})
            return MLXModelState(model=None, tokenizer=None)

        monkeypatch.setattr("glossate.core.translator.load_model", fake_load)
        monkeypatch.setattr("glossate.core.translator.translate", lambda *a, **kw: [])

        with Session(
            mt_model="gemma-4-e4b",
            device="cuda",
            mt_backend="transformers",
            compute_type="float16",
        ) as s:
            s.translate([_cue()], target="tr")

        assert captured == [
            {
                "model": "gemma-4-e4b",
                "device": "cuda",
                "backend": "transformers",
                "compute_type": "float16",
            }
        ]


# ---------------------------------------------------------------------------
# Session.subtitle
# ---------------------------------------------------------------------------


class TestSessionSubtitle:
    def test_calls_pipeline_in_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        order: list[str] = []
        audio_file = tmp_path / "audio.wav"
        out_path = tmp_path / "out.srt"

        monkeypatch.setattr(
            "glossate.utils.ffmpeg.extract_audio",
            lambda p, **kw: (order.append("extract"), audio_file)[1],
        )
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe",
            lambda *a, **kw: (order.append("transcribe"), _whisper_result())[1],
        )
        monkeypatch.setattr(
            "glossate.core.segmenter.segment",
            lambda d: (order.append("segment"), [_cue()])[1],
        )
        monkeypatch.setattr(
            "glossate.core.translator.translate",
            lambda c, t, **kw: (order.append("translate"), [_cue()])[1],
        )
        monkeypatch.setattr(
            "glossate.core.serializer.write_srt",
            lambda c, p: (order.append("serialize"), out_path)[1],
        )
        monkeypatch.setattr(
            "glossate.utils.paths.default_output_path", lambda *a, **kw: out_path
        )

        with Session() as s:
            s.subtitle(audio_file, target="tr")

        assert order == ["extract", "transcribe", "segment", "translate", "serialize"]

    def test_skips_translate_when_target_is_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        translate_called = []
        audio_file = tmp_path / "audio.wav"
        out_path = tmp_path / "out.srt"

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: audio_file)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result()
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])
        monkeypatch.setattr(
            "glossate.core.translator.translate",
            lambda c, t, **kw: translate_called.append(t) or [],
        )
        monkeypatch.setattr("glossate.core.serializer.write_srt", lambda c, p: out_path)
        monkeypatch.setattr(
            "glossate.utils.paths.default_output_path", lambda *a, **kw: out_path
        )

        with Session() as s:
            s.subtitle(audio_file, target=None)

        assert translate_called == []

    def test_uses_default_output_path_when_output_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        expected = tmp_path / "default.srt"
        path_calls: list[tuple[Any, ...]] = []

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result()
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])
        monkeypatch.setattr(
            "glossate.utils.paths.default_output_path",
            lambda p, f, **kw: (path_calls.append((p, f)), expected)[1],
        )
        monkeypatch.setattr("glossate.core.serializer.write_srt", lambda c, p: expected)

        audio_file = tmp_path / "audio.wav"
        with Session() as s:
            result = s.subtitle(audio_file)

        assert result == expected
        assert len(path_calls) == 1
        assert path_calls[0][1] == "srt"

    def test_uses_explicit_output_when_provided(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        explicit_out = tmp_path / "my_output.srt"
        written_to: list[Path] = []

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result()
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])
        monkeypatch.setattr(
            "glossate.core.serializer.write_srt",
            lambda c, p: (written_to.append(Path(p)), Path(p))[1],
        )

        audio_file = tmp_path / "audio.wav"
        with Session() as s:
            s.subtitle(audio_file, output=explicit_out)

        assert written_to[0] == explicit_out

    def test_raises_value_error_for_unknown_format(
        self, tmp_path: Path, no_model_load: None
    ) -> None:
        with Session() as s:
            with pytest.raises(ValueError, match="Unknown format"):
                s.subtitle(tmp_path / "audio.wav", format="mkv")

    def test_returns_path(
        self, tmp_path: Path, mock_pipeline: dict[str, Any]
    ) -> None:
        audio_file = tmp_path / "audio.wav"
        with Session() as s:
            result = s.subtitle(audio_file, target="tr")
        assert result == mock_pipeline["out_path"]


class TestSessionSubtitleVideo:
    def test_creates_subtitle_then_burns_video(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        video = tmp_path / "video.mp4"
        subtitle_path = tmp_path / "video.srt"
        burned_path = tmp_path / "video.subbed.mp4"
        calls: list[tuple[Any, ...]] = []

        monkeypatch.setattr(
            "glossate.api.Session.subtitle",
            lambda s, p, **kw: (calls.append(("subtitle", p, kw)), subtitle_path)[1],
        )
        monkeypatch.setattr(
            "glossate.api.burn_subtitles",
            lambda v, sub, **kw: (calls.append(("burn", v, sub, kw)), burned_path)[1],
        )

        with Session() as s:
            result = s.subtitle_video(
                video,
                target="tr",
                subtitle_output=subtitle_path,
                output=burned_path,
            )

        assert result == burned_path
        assert calls == [
            (
                "subtitle",
                video,
                {
                    "target": "tr",
                    "source": None,
                    "format": "srt",
                    "output": subtitle_path,
                },
            ),
            ("burn", video, subtitle_path, {"output": burned_path}),
        ]

    def test_rejects_non_burnable_format(self, tmp_path: Path) -> None:
        with Session() as s:
            with pytest.raises(ValueError, match="Cannot burn format"):
                s.subtitle_video(tmp_path / "video.mp4", format="md")


# ---------------------------------------------------------------------------
# Output format dispatch
# ---------------------------------------------------------------------------


class TestSubtitleFormatDispatch:
    def test_srt_uses_write_srt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        out = tmp_path / "out.srt"
        called: list[str] = []

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result()
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])
        monkeypatch.setattr(
            "glossate.core.serializer.write_srt", lambda c, p: (called.append("srt"), out)[1]
        )
        monkeypatch.setattr(
            "glossate.utils.paths.default_output_path", lambda *a, **kw: out
        )

        with Session() as s:
            s.subtitle(tmp_path / "audio.wav", format="srt")
        assert called == ["srt"]

    def test_md_uses_build_markdown_and_loads_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_model_load: None
    ) -> None:
        # Markdown must route through notes.build_markdown — and do so even with
        # no target, since the Gemma formatting pass needs the loaded model.
        out = tmp_path / "out.md"
        called: list[str] = []

        monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p, **kw: p)
        monkeypatch.setattr(
            "glossate.core.transcriber.transcribe", lambda *a, **kw: _whisper_result()
        )
        monkeypatch.setattr("glossate.core.segmenter.segment", lambda d: [_cue()])
        monkeypatch.setattr(
            "glossate.core.notes.build_markdown",
            lambda *a, **kw: (called.append("md"), "# notes\n")[1],
        )
        monkeypatch.setattr(
            "glossate.utils.paths.default_output_path", lambda *a, **kw: out
        )

        with Session() as s:
            result = s.subtitle(tmp_path / "audio.wav", format="md")  # no target
        assert called == ["md"]
        assert result == out
        assert out.read_text(encoding="utf-8") == "# notes\n"

    def test_md_without_session_raises(self, tmp_path: Path) -> None:
        # md needs the model, so it requires an active session even with no target.
        s = Session()
        with pytest.raises(RuntimeError, match="Session is not active"):
            s.subtitle(tmp_path / "audio.wav", format="md")


# ---------------------------------------------------------------------------
# One-shot convenience functions
# ---------------------------------------------------------------------------


class TestOneShotFunctions:
    def test_oneshot_subtitle_creates_and_closes_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        lifecycle: list[str] = []


        def patched_enter(self: Session) -> Session:
            lifecycle.append("enter")
            return self

        def patched_exit(self: Session, *args: Any) -> None:
            lifecycle.append("exit")

        monkeypatch.setattr(Session, "__enter__", patched_enter)
        monkeypatch.setattr(Session, "__exit__", patched_exit)

        out_path = tmp_path / "out.srt"
        monkeypatch.setattr("glossate.api.Session.subtitle", lambda s, p, **kw: out_path)

        subtitle(tmp_path / "audio.wav", target="tr")

        assert lifecycle == ["enter", "exit"]

    def test_oneshot_transcribe_creates_and_closes_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        lifecycle: list[str] = []
        cues = [_cue()]

        def patched_enter(self: Session) -> Session:
            lifecycle.append("enter")
            return self

        def patched_exit(self: Session, *args: Any) -> None:
            lifecycle.append("exit")

        monkeypatch.setattr(Session, "__enter__", patched_enter)
        monkeypatch.setattr(Session, "__exit__", patched_exit)
        monkeypatch.setattr("glossate.api.Session.transcribe", lambda s, p, **kw: cues)

        transcribe(tmp_path / "audio.wav")

        assert lifecycle == ["enter", "exit"]

    def test_oneshot_translate_creates_and_closes_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lifecycle: list[str] = []
        cues = [_cue()]
        translated = [_cue("Merhaba", lang="tr")]

        def patched_enter(self: Session) -> Session:
            lifecycle.append("enter")
            return self

        def patched_exit(self: Session, *args: Any) -> None:
            lifecycle.append("exit")

        monkeypatch.setattr(Session, "__enter__", patched_enter)
        monkeypatch.setattr(Session, "__exit__", patched_exit)
        monkeypatch.setattr("glossate.api.Session.translate", lambda s, c, **kw: translated)

        translate(cues, target="tr")

        assert lifecycle == ["enter", "exit"]

    def test_oneshot_subtitle_video_creates_and_closes_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        lifecycle: list[str] = []
        out_path = tmp_path / "out.mp4"

        def patched_enter(self: Session) -> Session:
            lifecycle.append("enter")
            return self

        def patched_exit(self: Session, *args: Any) -> None:
            lifecycle.append("exit")

        monkeypatch.setattr(Session, "__enter__", patched_enter)
        monkeypatch.setattr(Session, "__exit__", patched_exit)
        monkeypatch.setattr("glossate.api.Session.subtitle_video", lambda s, p, **kw: out_path)

        result = subtitle_video(tmp_path / "video.mp4", target="tr")

        assert result == out_path
        assert lifecycle == ["enter", "exit"]


class TestBurnSubtitlesApi:
    def test_delegates_to_ffmpeg_helper(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "video.srt"
        out = tmp_path / "video.subbed.mp4"
        calls: list[tuple[Path, Path, Path]] = []

        monkeypatch.setattr(
            "glossate.utils.ffmpeg.burn_subtitles",
            lambda v, s, o: (calls.append((v, s, o)), out)[1],
        )

        result = burn_subtitles(video, subs, output=out)

        assert result == out
        assert calls == [(video, subs, out)]

    def test_uses_default_burn_output_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "video.srt"
        out = tmp_path / "default.mp4"

        monkeypatch.setattr("glossate.utils.paths.default_burn_output_path", lambda p: out)
        monkeypatch.setattr("glossate.utils.ffmpeg.burn_subtitles", lambda v, s, o: o)

        assert burn_subtitles(video, subs) == out

    def test_ffmpeg_error_becomes_subtitle_burn_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from glossate.utils.ffmpeg import FFmpegError

        monkeypatch.setattr(
            "glossate.utils.ffmpeg.burn_subtitles",
            lambda *a, **kw: (_ for _ in ()).throw(FFmpegError("burn failed")),
        )

        with pytest.raises(SubtitleBurnError, match="burn failed"):
            burn_subtitles(tmp_path / "video.mp4", tmp_path / "video.srt")

    def test_rejects_audio_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="video input"):
            burn_subtitles(tmp_path / "audio.wav", tmp_path / "audio.srt")
