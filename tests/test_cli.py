# SPDX-License-Identifier: MIT
"""Integration tests for glossate.cli with mocked API calls."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from glossate.cli import cmd_run, main
from glossate.core.segmenter import Cue
from glossate.core.transcriber import WhisperResult, WhisperSegment, WhisperWord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        input="audio.wav",
        source=None,
        target=None,
        format="srt",
        md_scope="translated",
        md_layout="two-prose",
        md_timestamps=True,
        output=None,
        asr_model="large",
        mt_model="aya-expanse-8b",
        device="auto",
        asr_backend="auto",
        mt_backend="auto",
        compute_type="auto",
        burn=False,
        burn_output=None,
        no_cache=False,
        verbose=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_whisper_result(lang: str = "ar") -> WhisperResult:
    word = WhisperWord(word="hello", start=0.0, end=0.5)
    seg = WhisperSegment(start=0.0, end=0.5, text="hello", words=[word])
    return WhisperResult(segments=[seg], detected_language=lang)


def _make_cues(lang: str = "ar") -> list[Cue]:
    return [Cue(start=0.0, end=0.5, text="hello", lang=lang)]


# ---------------------------------------------------------------------------
# Integration: cmd_run — audio input (extract skipped)
# ---------------------------------------------------------------------------


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=120.0)
def test_transcribe_audio_writes_srt(
    mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(audio))
    result = cmd_run(args, console)

    assert result == 0
    prog.skip_stage.assert_any_call("extract")
    prog.skip_stage.assert_any_call("translate")
    mock_write_srt.assert_called_once()


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.translator.translate", return_value=[Cue(0.0, 0.5, "merhaba", "tr")])
@patch("glossate.core.translator.load_model", return_value=(None, None))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=120.0)
def test_transcribe_and_translate(
    mock_dur, mock_transcribe, mock_load_mt, mock_translate, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(audio), target="tr")
    result = cmd_run(args, console)

    assert result == 0
    mock_translate.assert_called_once()
    _, load_kwargs = mock_load_mt.call_args
    assert load_kwargs["device"] == "auto"
    assert load_kwargs["backend"] == "auto"
    assert load_kwargs["compute_type"] == "auto"
    mock_write_srt.assert_called_once()


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=120.0)
def test_source_flag_passed_to_transcribe(
    mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(audio), source="ar")
    result = cmd_run(args, console)

    assert result == 0
    mock_transcribe.assert_called_once()
    _, kwargs = mock_transcribe.call_args
    assert kwargs["source_lang"] == "ar"
    assert kwargs["model"] == "large"
    assert kwargs["total_seconds"] == 120.0
    assert kwargs["on_progress"] is not None
    assert kwargs["device"] == "auto"
    assert kwargs["backend"] == "auto"
    assert kwargs["compute_type"] == "auto"


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=120.0)
def test_cuda_flags_passed_to_transcribe(
    mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    MockProgress.return_value = MagicMock()

    args = _make_args(
        input=str(audio),
        source="ar",
        device="cuda",
        asr_backend="faster-whisper",
        compute_type="float16",
    )
    result = cmd_run(args, MagicMock())

    assert result == 0
    _, kwargs = mock_transcribe.call_args
    assert kwargs["device"] == "cuda"
    assert kwargs["backend"] == "faster-whisper"
    assert kwargs["compute_type"] == "float16"


# ---------------------------------------------------------------------------
# Integration: cmd_run — video input (extract active)
# ---------------------------------------------------------------------------


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("video.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("video.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=90.0)
@patch("glossate.utils.ffmpeg.extract_audio")
def test_video_input_runs_extract_stage(
    mock_extract, mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    video = tmp_path / "video.mp4"
    video.touch()

    extracted = tmp_path / "extracted.wav"
    extracted.touch()
    mock_extract.return_value = extracted

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(video), source="ar")
    result = cmd_run(args, console)

    assert result == 0
    mock_extract.assert_called_once_with(video)
    prog.begin_stage.assert_any_call("extract")
    prog.complete_stage.assert_any_call("extract")


# ---------------------------------------------------------------------------
# Integration: cmd_run — translate stage skipped without --target
# ---------------------------------------------------------------------------


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=30.0)
def test_translate_stage_skipped_without_target(
    mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path
):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(audio), source="ar", target=None)
    result = cmd_run(args, console)

    assert result == 0
    prog.skip_stage.assert_any_call("translate")


# ---------------------------------------------------------------------------
# Integration: cmd_run — output format flags
# ---------------------------------------------------------------------------


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("audio.srt"))
@patch("glossate.utils.paths.default_output_path", return_value=Path("audio.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=30.0)
def test_output_format_srt(mock_dur, mock_transcribe, mock_default_path, mock_write_srt, MockProgress, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    MockProgress.return_value = MagicMock()

    args = _make_args(input=str(audio), source="ar", format="srt")
    result = cmd_run(args, MagicMock())

    assert result == 0
    mock_write_srt.assert_called_once()


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.notes.build_markdown", return_value="# notes\n")
@patch("glossate.core.translator.load_model")
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=30.0)
def test_output_format_md_builds_notes(
    mock_dur, mock_transcribe, mock_load_mt, mock_build, MockProgress, tmp_path
):
    # md routes through notes.build_markdown (loading the MT model) and writes a file.
    from glossate.core.translator import TranslationModelState

    audio = tmp_path / "audio.wav"
    audio.touch()
    out_md = tmp_path / "audio.md"

    mock_transcribe.return_value = _make_whisper_result("ar")
    mock_load_mt.return_value = TranslationModelState(model=None, tokenizer=None, backend="transformers")
    MockProgress.return_value = MagicMock()

    args = _make_args(input=str(audio), source="ar", format="md")
    with patch("glossate.utils.paths.default_output_path", return_value=out_md):
        result = cmd_run(args, MagicMock())

    assert result == 0
    mock_load_mt.assert_called_once()
    mock_build.assert_called_once()
    assert out_md.read_text(encoding="utf-8") == "# notes\n"


# ---------------------------------------------------------------------------
# Integration: cmd_run — explicit -o output path
# ---------------------------------------------------------------------------


@patch("glossate.cli.GlossateProgress")
@patch("glossate.core.serializer.write_srt", return_value=Path("out.srt"))
@patch("glossate.core.transcriber.transcribe")
@patch("glossate.utils.ffmpeg.get_audio_duration", return_value=30.0)
def test_explicit_output_path(mock_dur, mock_transcribe, mock_write_srt, MockProgress, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()

    mock_transcribe.return_value = _make_whisper_result("ar")
    prog = MagicMock()
    MockProgress.return_value = prog

    console = MagicMock()
    args = _make_args(input=str(audio), source="ar", output=str(tmp_path / "out.srt"))
    result = cmd_run(args, console)

    assert result == 0
    written_path = mock_write_srt.call_args[0][1]
    assert Path(written_path) == tmp_path / "out.srt"


def test_burn_flag_burns_written_subtitles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    extracted = tmp_path / "audio.wav"
    extracted.touch()
    subtitle_path = tmp_path / "video.srt"
    burned_path = tmp_path / "video.subbed.mp4"
    burn_calls: list[tuple[Path, Path, Path]] = []

    prog = MagicMock()
    monkeypatch.setattr("glossate.cli.GlossateProgress", lambda *a, **kw: prog)
    monkeypatch.setattr("glossate.utils.ffmpeg.get_audio_duration", lambda p: 30.0)
    monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p: extracted)
    monkeypatch.setattr("glossate.core.transcriber.transcribe", lambda *a, **kw: _make_whisper_result("ar"))
    monkeypatch.setattr("glossate.core.serializer.write_srt", lambda c, p: subtitle_path)
    monkeypatch.setattr("glossate.utils.paths.default_burn_output_path", lambda p: burned_path)
    monkeypatch.setattr(
        "glossate.utils.ffmpeg.burn_subtitles",
        lambda v, s, o: (burn_calls.append((v, s, o)), burned_path)[1],
    )

    result = cmd_run(_make_args(input=str(video), burn=True), MagicMock())

    assert result == 0
    assert burn_calls == [(video, subtitle_path, burned_path)]


def test_burn_output_flag_overrides_burn_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    extracted = tmp_path / "audio.wav"
    extracted.touch()
    subtitle_path = tmp_path / "video.srt"
    burned_path = tmp_path / "custom.mp4"
    burn_calls: list[tuple[Path, Path, Path]] = []

    monkeypatch.setattr("glossate.cli.GlossateProgress", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("glossate.utils.ffmpeg.get_audio_duration", lambda p: 30.0)
    monkeypatch.setattr("glossate.utils.ffmpeg.extract_audio", lambda p: extracted)
    monkeypatch.setattr("glossate.core.transcriber.transcribe", lambda *a, **kw: _make_whisper_result("ar"))
    monkeypatch.setattr("glossate.core.serializer.write_srt", lambda c, p: subtitle_path)
    monkeypatch.setattr(
        "glossate.utils.ffmpeg.burn_subtitles",
        lambda v, s, o: (burn_calls.append((v, s, o)), burned_path)[1],
    )

    result = cmd_run(
        _make_args(input=str(video), burn=True, burn_output=str(burned_path)),
        MagicMock(),
    )

    assert result == 0
    assert burn_calls == [(video, subtitle_path, burned_path)]


def test_burn_flag_rejects_audio_input(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()

    result = cmd_run(_make_args(input=str(audio), burn=True), MagicMock())

    assert result == 1


def test_burn_flag_rejects_non_subtitle_format(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    result = cmd_run(_make_args(input=str(video), burn=True, format="md"), MagicMock())

    assert result == 1


# ---------------------------------------------------------------------------
# Integration: cmd_run — missing file
# ---------------------------------------------------------------------------


def test_missing_file_returns_1(tmp_path):
    console = MagicMock()
    args = _make_args(input=str(tmp_path / "nonexistent.wav"))
    result = cmd_run(args, console)
    assert result == 1


# ---------------------------------------------------------------------------
# Integration: main — info subcommand
# ---------------------------------------------------------------------------


def test_main_info_exits_0():
    with patch("sys.argv", ["glossate", "info"]):
        with patch("glossate.cli.cmd_info"):
            with patch("glossate.cli.create_console"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Integration: main — --verbose sets logging level
# ---------------------------------------------------------------------------


def test_verbose_flag_sets_debug_logging():
    import logging

    with patch("sys.argv", ["glossate", "--verbose", "info"]):
        with patch("glossate.cli.cmd_info"):
            with patch("glossate.cli.create_console"):
                with patch("logging.basicConfig") as mock_logging:
                    with pytest.raises(SystemExit):
                        main()
    mock_logging.assert_called_once_with(level=logging.DEBUG)
