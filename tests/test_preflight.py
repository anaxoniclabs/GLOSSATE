# SPDX-License-Identifier: MIT
from unittest.mock import patch

import pytest

from glossate.utils.preflight import (
    _FfmpegMissingError,
    _ModelMissingError,
    run_preflight_checks,
)


class TestRunPreflightChecks:
    def test_raises_ffmpeg_missing_when_not_on_path(self) -> None:
        with patch("glossate.utils.preflight.shutil.which", return_value=None):
            with pytest.raises(_FfmpegMissingError):
                run_preflight_checks()

    def test_passes_when_all_checks_succeed(self) -> None:
        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", return_value=True),
        ):
            run_preflight_checks(asr_backend="mlx")  # must not raise

    def test_raises_model_missing_when_whisper_not_cached(self) -> None:
        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", return_value=False),
        ):
            with pytest.raises(_ModelMissingError):
                run_preflight_checks(asr_backend="mlx")

    def test_raises_model_missing_when_only_mt_model_not_cached(self) -> None:
        def cached_only_whisper(model_id: str) -> bool:
            return "whisper" in model_id

        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", side_effect=cached_only_whisper),
        ):
            with pytest.raises(_ModelMissingError):
                run_preflight_checks(mt_model="gemma-4-e4b", asr_backend="mlx")

    def test_ffmpeg_error_precedes_model_error(self) -> None:
        with (
            patch("glossate.utils.preflight.shutil.which", return_value=None),
            patch("glossate.utils.preflight._is_hf_model_cached", return_value=False),
        ):
            with pytest.raises(_FfmpegMissingError):
                run_preflight_checks()

    def test_custom_asr_model_resolves_to_correct_hf_id(self) -> None:
        calls: list[str] = []

        def track(model_id: str) -> bool:
            calls.append(model_id)
            return False

        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", side_effect=track),
        ):
            with pytest.raises(_ModelMissingError):
                run_preflight_checks(asr_model="medium", asr_backend="mlx")

        assert any("whisper-medium" in c for c in calls)

    def test_custom_mt_model_resolves_to_correct_hf_id(self) -> None:
        calls: list[str] = []

        def track(model_id: str) -> bool:
            calls.append(model_id)
            return True  # whisper passes; we check MT is resolved

        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", side_effect=track),
        ):
            run_preflight_checks(mt_model="qwen2.5-7b", asr_backend="mlx")

        assert any("Qwen" in c for c in calls)

    def test_faster_whisper_backend_skips_mlx_whisper_cache_check(self) -> None:
        calls: list[str] = []

        with (
            patch("glossate.utils.preflight.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("glossate.utils.preflight._is_hf_model_cached", side_effect=lambda m: calls.append(m) or False),
        ):
            # mt_model uses the ollama prefix so only the ASR cache check could fire.
            run_preflight_checks(asr_backend="faster-whisper", mt_model="ollama/x")

        assert calls == []
