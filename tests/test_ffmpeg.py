# SPDX-License-Identifier: MIT
import subprocess
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest

from glossate.utils.ffmpeg import (
    FFmpegError,
    FFmpegTimeout,
    burn_subtitles,
    extract_audio,
    get_ffmpeg_version,
)

_SUCCESS = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
_FAILURE = CompletedProcess(args=[], returncode=1, stdout="", stderr="err line 1\nerr line 2")


def _mock_tmp(tmp_path: Path, name: str = "out.wav") -> MagicMock:
    m = MagicMock()
    m.name = str(tmp_path / name)
    return m


class TestExtractAudio:
    def test_wav_returned_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.wav"
        p.touch()
        assert extract_audio(p) == p

    def test_mp3_returned_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.mp3"
        p.touch()
        assert extract_audio(p) == p

    def test_m4a_returned_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.m4a"
        p.touch()
        assert extract_audio(p) == p

    def test_flac_returned_unchanged(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.flac"
        p.touch()
        assert extract_audio(p) == p

    def test_audio_input_makes_no_subprocess_call(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.wav"
        p.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run") as mock_run:
            extract_audio(p)
        mock_run.assert_not_called()

    def test_video_input_calls_ffmpeg(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS) as mock_run:
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                extract_audio(mp4)
        mock_run.assert_called_once()

    def test_ffmpeg_command_shape(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS) as mock_run:
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                extract_audio(mp4)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("stdin") == subprocess.DEVNULL
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-hide_banner" in cmd
        assert "-loglevel" in cmd and cmd[cmd.index("-loglevel") + 1] == "error"
        assert "-nostats" in cmd
        assert "-y" in cmd
        assert "-i" in cmd
        assert str(mp4) in cmd
        assert "-vn" in cmd
        assert "pcm_s16le" in cmd
        assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
        assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"

    def test_returned_path_for_video_is_wav(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS):
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                result = extract_audio(mp4)
        assert result.suffix == ".wav"

    def test_ffmpeg_error_on_nonzero_exit(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_FAILURE):
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                with pytest.raises(FFmpegError) as exc_info:
                    extract_audio(mp4)
        msg = str(exc_info.value)
        assert "extract" in msg
        assert "err line" in msg

    def test_ffmpeg_error_is_exact_type_not_timeout(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_FAILURE):
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                with pytest.raises(FFmpegError) as exc_info:
                    extract_audio(mp4)
        assert type(exc_info.value) is FFmpegError

    def test_ffmpeg_timeout_raised(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", side_effect=TimeoutExpired(cmd="ffmpeg", timeout=1.0)):
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                with pytest.raises(FFmpegTimeout):
                    extract_audio(mp4, timeout=1.0)

    def test_ffmpeg_timeout_is_subclass_of_ffmpeg_error(self, tmp_path: Path) -> None:
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        with patch("glossate.utils.ffmpeg.subprocess.run", side_effect=TimeoutExpired(cmd="ffmpeg", timeout=1.0)):
            with patch("glossate.utils.ffmpeg.tempfile.NamedTemporaryFile", return_value=_mock_tmp(tmp_path)):
                with pytest.raises(FFmpegError):
                    extract_audio(mp4, timeout=1.0)


class TestGetFfmpegVersion:
    def test_returns_first_line_of_version_output(self) -> None:
        mock_result = CompletedProcess(
            args=[],
            returncode=0,
            stdout="ffmpeg version 6.1 Copyright (c) ...\nother stuff",
            stderr="",
        )
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=mock_result):
            assert get_ffmpeg_version() == "ffmpeg version 6.1 Copyright (c) ..."

    def test_returns_none_when_not_on_path(self) -> None:
        with patch("glossate.utils.ffmpeg.subprocess.run", side_effect=FileNotFoundError):
            assert get_ffmpeg_version() is None

    def test_returns_none_when_ffmpeg_exits_nonzero(self) -> None:
        mock_result = CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=mock_result):
            assert get_ffmpeg_version() is None


class TestBurnSubtitles:
    def test_calls_ffmpeg_with_subtitles_filter(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "subs.srt"
        out = tmp_path / "video.subbed.mp4"
        video.touch()
        subs.touch()

        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS) as mock_run:
            result = burn_subtitles(video, subs, out)

        assert result == out
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd and str(video) in cmd
        assert "-vf" in cmd
        assert cmd[cmd.index("-vf") + 1] == f"subtitles={subs}"
        assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
        assert str(out) == cmd[-1]
        assert mock_run.call_args[1]["stdin"] == subprocess.DEVNULL

    def test_escapes_filter_path_delimiters(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "a,b's.srt"
        out = tmp_path / "out.mp4"
        video.touch()
        subs.touch()

        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS) as mock_run:
            burn_subtitles(video, subs, out)

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-vf") + 1].endswith("a\\,b\\'s.srt")

    def test_creates_output_parent_directory(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "subs.srt"
        out = tmp_path / "nested" / "out.mp4"
        video.touch()
        subs.touch()

        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_SUCCESS):
            burn_subtitles(video, subs, out)

        assert out.parent.exists()

    def test_ffmpeg_error_on_nonzero_exit(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "subs.srt"
        out = tmp_path / "out.mp4"
        video.touch()
        subs.touch()

        with patch("glossate.utils.ffmpeg.subprocess.run", return_value=_FAILURE):
            with pytest.raises(FFmpegError, match="burn"):
                burn_subtitles(video, subs, out)

    def test_ffmpeg_timeout_raised(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        subs = tmp_path / "subs.srt"
        out = tmp_path / "out.mp4"
        video.touch()
        subs.touch()

        with patch("glossate.utils.ffmpeg.subprocess.run", side_effect=TimeoutExpired(cmd="ffmpeg", timeout=1.0)):
            with pytest.raises(FFmpegTimeout, match="burn"):
                burn_subtitles(video, subs, out, timeout=1.0)
