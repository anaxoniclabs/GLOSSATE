# SPDX-License-Identifier: MIT
"""State-machine tests for GlossateProgress. No terminal or Rich rendering required."""

import pytest

from glossate.ui.progress import GlossateProgress, StageState


@pytest.fixture()
def prog() -> GlossateProgress:
    return GlossateProgress()


class TestInitialState:
    def test_all_stages_pending(self, prog: GlossateProgress) -> None:
        assert prog.get_state("extract") == StageState.PENDING
        assert prog.get_state("transcribe") == StageState.PENDING
        assert prog.get_state("translate") == StageState.PENDING


class TestBeginStage:
    def test_begin_sets_active(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe")
        assert prog.get_state("transcribe") == StageState.ACTIVE

    def test_other_stages_unchanged(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe")
        assert prog.get_state("extract") == StageState.PENDING
        assert prog.get_state("translate") == StageState.PENDING

    def test_begin_extract_stage(self, prog: GlossateProgress) -> None:
        prog.begin_stage("extract")
        assert prog.get_state("extract") == StageState.ACTIVE

    def test_begin_accepts_total(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe", total=200)
        assert prog.get_state("transcribe") == StageState.ACTIVE


class TestCompleteStage:
    def test_complete_sets_complete(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe")
        prog.complete_stage("transcribe")
        assert prog.get_state("transcribe") == StageState.COMPLETE

    def test_complete_clears_active_stage(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe")
        prog.complete_stage("transcribe")
        assert prog._active_stage is None

    def test_complete_does_not_affect_other_stages(self, prog: GlossateProgress) -> None:
        prog.begin_stage("transcribe")
        prog.complete_stage("transcribe")
        assert prog.get_state("extract") == StageState.PENDING
        assert prog.get_state("translate") == StageState.PENDING


class TestErrorStage:
    def test_error_sets_error_state(self, prog: GlossateProgress) -> None:
        prog.begin_stage("translate")
        prog.error_stage("translate", "model failed")
        assert prog.get_state("translate") == StageState.ERROR

    def test_error_stores_message(self, prog: GlossateProgress) -> None:
        prog.begin_stage("translate")
        prog.error_stage("translate", "model failed")
        assert prog._stages["translate"].error_message == "model failed"

    def test_error_clears_active_stage(self, prog: GlossateProgress) -> None:
        prog.begin_stage("translate")
        prog.error_stage("translate", "model failed")
        assert prog._active_stage is None

    def test_error_without_message(self, prog: GlossateProgress) -> None:
        prog.error_stage("translate")
        assert prog.get_state("translate") == StageState.ERROR


class TestSkipStage:
    def test_skip_goes_from_pending_to_complete(self, prog: GlossateProgress) -> None:
        prog.skip_stage("extract")
        assert prog.get_state("extract") == StageState.COMPLETE

    def test_skip_does_not_pass_through_active(self, prog: GlossateProgress) -> None:
        prog.skip_stage("translate")
        assert prog.get_state("translate") == StageState.COMPLETE
        assert prog._active_stage is None

    def test_skip_extract_then_begin_transcribe(self, prog: GlossateProgress) -> None:
        prog.skip_stage("extract")
        prog.begin_stage("transcribe")
        assert prog.get_state("extract") == StageState.COMPLETE
        assert prog.get_state("transcribe") == StageState.ACTIVE


class TestStartStop:
    def test_start_does_not_raise(self, prog: GlossateProgress) -> None:
        prog.start()
        prog.stop()

    def test_double_start_does_not_raise(self, prog: GlossateProgress) -> None:
        prog.start()
        prog.start()
        prog.stop()

    def test_double_stop_does_not_raise(self, prog: GlossateProgress) -> None:
        prog.start()
        prog.stop()
        prog.stop()

    def test_stop_without_start_does_not_raise(self, prog: GlossateProgress) -> None:
        prog.stop()


class TestSetFileInfo:
    def test_set_file_info_stores_title(self, prog: GlossateProgress) -> None:
        prog.set_file_info("video.mp4", "1:23")
        assert prog._file_title == "video.mp4"

    def test_set_file_info_stores_duration(self, prog: GlossateProgress) -> None:
        prog.set_file_info("video.mp4", "1:23")
        assert prog._file_duration == "1:23"


class TestFullPipelineFlow:
    def test_extract_transcribe_translate(self, prog: GlossateProgress) -> None:
        prog.begin_stage("extract")
        assert prog.get_state("extract") == StageState.ACTIVE
        prog.complete_stage("extract")
        assert prog.get_state("extract") == StageState.COMPLETE

        prog.begin_stage("transcribe")
        assert prog.get_state("transcribe") == StageState.ACTIVE
        prog.complete_stage("transcribe")
        assert prog.get_state("transcribe") == StageState.COMPLETE

        prog.begin_stage("translate")
        assert prog.get_state("translate") == StageState.ACTIVE
        prog.complete_stage("translate")
        assert prog.get_state("translate") == StageState.COMPLETE

    def test_skip_extract_and_translate(self, prog: GlossateProgress) -> None:
        prog.skip_stage("extract")
        prog.begin_stage("transcribe")
        prog.complete_stage("transcribe")
        prog.skip_stage("translate")

        assert prog.get_state("extract") == StageState.COMPLETE
        assert prog.get_state("transcribe") == StageState.COMPLETE
        assert prog.get_state("translate") == StageState.COMPLETE
