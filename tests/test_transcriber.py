# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from glossate.core.transcriber import (
    WhisperError,
    WhisperResult,
    detect_language,
    transcribe,
)

_RAW_RESULT: dict = {
    "text": "Hello world.",
    "language": "en",
    "segments": [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.8, "probability": 0.99},
                {"word": " world.", "start": 0.9, "end": 2.0, "probability": 0.97},
            ],
        }
    ],
}


@pytest.fixture()
def mlx(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx_whisper", mock)
    return mock


@pytest.fixture()
def librosa_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    mock.load.return_value = ([0.0] * 160000, 16000)
    monkeypatch.setitem(sys.modules, "librosa", mock)
    return mock


@pytest.fixture()
def sf_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "soundfile", mock)
    return mock


class TestTranscribe:
    def test_returns_whisper_result(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        result = transcribe(tmp_path / "audio.wav", isolated=False, backend="mlx")
        assert isinstance(result, WhisperResult)

    def test_word_timestamps_passed_through(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        result = transcribe(tmp_path / "audio.wav", isolated=False, backend="mlx")
        seg = result.segments[0]
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"
        assert seg.words[0].start == 0.0
        assert seg.words[1].end == 2.0

    def test_detected_language_populated(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        result = transcribe(tmp_path / "audio.wav", isolated=False, backend="mlx")
        assert result.detected_language == "en"

    def test_explicit_source_lang_passed_to_model(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        transcribe(tmp_path / "audio.wav", source_lang="ar", isolated=False, backend="mlx")
        _, kwargs = mlx.transcribe.call_args
        assert kwargs["language"] == "ar"

    def test_none_source_lang_triggers_autodetect(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        transcribe(tmp_path / "audio.wav", source_lang=None, isolated=False, backend="mlx")
        _, kwargs = mlx.transcribe.call_args
        assert kwargs["language"] is None

    def test_word_timestamps_enabled(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        transcribe(tmp_path / "audio.wav", isolated=False, backend="mlx")
        _, kwargs = mlx.transcribe.call_args
        assert kwargs["word_timestamps"] is True

    def test_whisper_error_on_model_failure(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(WhisperError, match="GPU OOM"):
            transcribe(tmp_path / "audio.wav", isolated=False, backend="mlx")

    def test_model_size_name_resolved_to_repo(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        transcribe(tmp_path / "audio.wav", model="large", isolated=False, backend="mlx")
        _, kwargs = mlx.transcribe.call_args
        assert "whisper-large" in kwargs["path_or_hf_repo"]

    def test_arbitrary_repo_id_passed_through(self, tmp_path: Path, mlx: MagicMock) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        transcribe(tmp_path / "audio.wav", model="org/my-custom-whisper", isolated=False, backend="mlx")
        _, kwargs = mlx.transcribe.call_args
        assert kwargs["path_or_hf_repo"] == "org/my-custom-whisper"

    def test_faster_whisper_backend_uses_cuda_model_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[dict[str, str]] = []

        class FakeWhisperModel:
            def __init__(self, model_id: str, *, device: str, compute_type: str) -> None:
                created.append({"model_id": model_id, "device": device, "compute_type": compute_type})

            def transcribe(self, *args, **kwargs):
                word = SimpleNamespace(word="hello", start=0.0, end=0.5)
                seg = SimpleNamespace(start=0.0, end=0.5, text="hello", words=[word])
                info = SimpleNamespace(language="en", language_probability=0.9)
                return iter([seg]), info

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            SimpleNamespace(WhisperModel=FakeWhisperModel),
        )
        monkeypatch.setattr("glossate.utils.device._cuda_available", lambda: True)

        result = transcribe(tmp_path / "audio.wav", model="turbo", device="cuda")

        assert result.detected_language == "en"
        assert created == [
            {"model_id": "large-v3-turbo", "device": "cuda", "compute_type": "float16"}
        ]

    def test_faster_whisper_uses_batched_pipeline_on_cuda(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On CUDA the model is wrapped in BatchedInferencePipeline and a
        # non-zero batch_size is passed for the parallel-chunk speedup.
        captured: dict[str, object] = {}

        class FakeWhisperModel:
            def __init__(self, model_id: str, *, device: str, compute_type: str) -> None:
                pass

        class FakeBatchedPipeline:
            def __init__(self, *, model: object) -> None:
                captured["wrapped"] = True

            def transcribe(self, audio: str, **kwargs: object):
                captured["batch_size"] = kwargs.get("batch_size")
                seg = SimpleNamespace(start=0.0, end=0.5, text="hi", words=[])
                info = SimpleNamespace(language="en", language_probability=0.9)
                return iter([seg]), info

        monkeypatch.setitem(
            sys.modules,
            "faster_whisper",
            SimpleNamespace(
                WhisperModel=FakeWhisperModel,
                BatchedInferencePipeline=FakeBatchedPipeline,
            ),
        )
        monkeypatch.setattr("glossate.utils.device._cuda_available", lambda: True)

        result = transcribe(tmp_path / "audio.wav", model="turbo", device="cuda")

        assert result.detected_language == "en"
        assert captured["wrapped"] is True
        assert captured["batch_size"] == 16


class TestDetectLanguage:
    def test_returns_language_and_confidence(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = {**_RAW_RESULT, "language_probs": {"en": 0.91}}
        lang, conf = detect_language(tmp_path / "audio.wav", clip_start=5.0, isolated=False, backend="mlx")
        assert lang == "en"
        assert conf == pytest.approx(0.91)

    def test_clip_extracted_at_requested_offset(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        detect_language(tmp_path / "audio.wav", clip_start=30.0, clip_duration=10.0, isolated=False, backend="mlx")
        _, kwargs = librosa_mock.load.call_args
        assert kwargs["offset"] == 30.0
        assert kwargs["duration"] == 10.0

    def test_default_clip_duration_is_10s(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        detect_language(tmp_path / "audio.wav", clip_start=0.0, isolated=False, backend="mlx")
        _, kwargs = librosa_mock.load.call_args
        assert kwargs["duration"] == 10.0

    def test_confidence_defaults_to_1_when_probs_absent(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = {"language": "tr"}
        _, conf = detect_language(tmp_path / "audio.wav", clip_start=0.0, isolated=False, backend="mlx")
        assert conf == 1.0

    def test_confidence_defaults_to_1_when_lang_missing_from_probs(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = {"language": "de", "language_probs": {"en": 0.5}}
        _, conf = detect_language(tmp_path / "audio.wav", clip_start=0.0, isolated=False, backend="mlx")
        assert conf == 1.0

    def test_whisper_error_on_model_failure(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.side_effect = RuntimeError("crash")
        with pytest.raises(WhisperError, match="crash"):
            detect_language(tmp_path / "audio.wav", clip_start=0.0, isolated=False, backend="mlx")

    def test_clip_extracted_via_librosa_not_ffmpeg(
        self, tmp_path: Path, mlx: MagicMock, librosa_mock: MagicMock, sf_mock: MagicMock
    ) -> None:
        mlx.transcribe.return_value = _RAW_RESULT
        detect_language(tmp_path / "audio.wav", clip_start=0.0, isolated=False, backend="mlx")
        librosa_mock.load.assert_called_once()
