# SPDX-License-Identifier: MIT
"""Path management for GLOSSATE."""

import datetime
from pathlib import Path
from typing import Literal

_FMT_EXT: dict[str, str] = {"srt": "srt", "md": "md"}


def get_app_dir() -> Path:
    """Return ~/Documents/GLOSSATE/ (created if needed)."""
    app_dir = Path.home() / "Documents" / "GLOSSATE"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def default_output_path(
    input_path: "Path | str",
    fmt: Literal["srt", "md"],
    *,
    base_dir: Path | None = None,
    today: datetime.date | None = None,
) -> Path:
    """Return a collision-free output path under base_dir/YYYY-MM-DD/.

    Creates the date directory if it does not exist. Collision policy:
    <stem>.<ext>, then <stem>.1.<ext>, <stem>.2.<ext>, etc.
    """
    if base_dir is None:
        base_dir = get_app_dir()
    if today is None:
        today = datetime.date.today()

    ext = _FMT_EXT[fmt]
    stem = Path(input_path).stem
    date_dir = base_dir / today.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    candidate = date_dir / f"{stem}.{ext}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = date_dir / f"{stem}.{counter}.{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def default_burn_output_path(
    input_path: "Path | str",
    *,
    base_dir: Path | None = None,
    today: datetime.date | None = None,
) -> Path:
    """Return a collision-free hard-subtitled MP4 path under base_dir/YYYY-MM-DD/."""
    if base_dir is None:
        base_dir = get_app_dir()
    if today is None:
        today = datetime.date.today()

    stem = Path(input_path).stem
    date_dir = base_dir / today.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    candidate = date_dir / f"{stem}.subbed.mp4"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = date_dir / f"{stem}.subbed.{counter}.mp4"
        if not candidate.exists():
            return candidate
        counter += 1
