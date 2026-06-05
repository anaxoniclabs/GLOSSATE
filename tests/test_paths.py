# SPDX-License-Identifier: MIT
import datetime
from pathlib import Path

from glossate.utils.paths import default_burn_output_path, default_output_path

TODAY = datetime.date(2026, 4, 30)
DATE_STR = "2026-04-30"


def test_default_path_date_folder(tmp_path: Path) -> None:
    result = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert result.parent == tmp_path / DATE_STR


def test_stem_derived_from_input(tmp_path: Path) -> None:
    result = default_output_path("my_lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert result.name == "my_lecture.srt"


def test_extension_srt(tmp_path: Path) -> None:
    result = default_output_path("f.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert result.suffix == ".srt"


def test_extension_md(tmp_path: Path) -> None:
    result = default_output_path("f.mp4", "md", base_dir=tmp_path, today=TODAY)
    assert result.suffix == ".md"


def test_first_call_no_numeric_suffix(tmp_path: Path) -> None:
    result = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert result.name == "lecture.srt"


def test_second_call_suffix_1(tmp_path: Path) -> None:
    first = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    first.touch()
    second = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert second.name == "lecture.1.srt"


def test_third_call_suffix_2(tmp_path: Path) -> None:
    first = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    first.touch()
    second = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    second.touch()
    third = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert third.name == "lecture.2.srt"


def test_no_double_increment(tmp_path: Path) -> None:
    # Verifies lecture.2.srt, not lecture.1.1.srt
    first = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    first.touch()
    second = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    second.touch()
    third = default_output_path("lecture.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert "1.1" not in third.name


def test_date_dir_created(tmp_path: Path) -> None:
    date_dir = tmp_path / DATE_STR
    assert not date_dir.exists()
    default_output_path("f.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert date_dir.is_dir()


def test_different_formats_independent(tmp_path: Path) -> None:
    srt = default_output_path("f.mp4", "srt", base_dir=tmp_path, today=TODAY)
    md = default_output_path("f.mp4", "md", base_dir=tmp_path, today=TODAY)
    assert srt.name == "f.srt"
    assert md.name == "f.md"


def test_input_path_object(tmp_path: Path) -> None:
    result = default_output_path(Path("/some/dir/lecture.mp4"), "srt", base_dir=tmp_path, today=TODAY)
    assert result.name == "lecture.srt"


def test_does_not_touch_real_filesystem(tmp_path: Path) -> None:
    result = default_output_path("f.mp4", "srt", base_dir=tmp_path, today=TODAY)
    assert result.is_relative_to(tmp_path)


def test_default_burn_output_path_uses_subbed_mp4(tmp_path: Path) -> None:
    result = default_burn_output_path("lecture.mkv", base_dir=tmp_path, today=TODAY)
    assert result.parent == tmp_path / DATE_STR
    assert result.name == "lecture.subbed.mp4"


def test_default_burn_output_path_increments(tmp_path: Path) -> None:
    first = default_burn_output_path("lecture.mp4", base_dir=tmp_path, today=TODAY)
    first.touch()
    second = default_burn_output_path("lecture.mp4", base_dir=tmp_path, today=TODAY)
    assert second.name == "lecture.subbed.1.mp4"
