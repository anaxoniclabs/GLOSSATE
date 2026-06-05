# SPDX-License-Identifier: MIT
"""GLOSSATE color theme and styling constants. Black and white palette."""

from rich.style import Style
from rich.theme import Theme


class Colors:
    """Black and white UI palette for GLOSSATE."""

    PRIMARY = "#FFFFFF"
    PRIMARY_LIGHT = "#DDDDDD"
    PRIMARY_DARK = "#AAAAAA"

    SUCCESS = "#FFFFFF"
    ERROR = "#FF6B6B"
    WARNING = "#CCCCCC"
    INFO = "#AAAAAA"

    BG_DARK = "#1A1A1A"
    BG_PANEL = "#2D2D2D"

    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#AAAAAA"
    TEXT_MUTED = "#666666"


GLOSSATE_THEME = Theme(
    {
        "glossate.primary": Style(color=Colors.PRIMARY, bold=True),
        "glossate.primary.light": Style(color=Colors.PRIMARY_LIGHT),
        "glossate.primary.dark": Style(color=Colors.PRIMARY_DARK),
        "glossate.success": Style(color=Colors.SUCCESS, bold=True),
        "glossate.error": Style(color=Colors.ERROR, bold=True),
        "glossate.warning": Style(color=Colors.WARNING),
        "glossate.info": Style(color=Colors.INFO),
        "glossate.text": Style(color=Colors.TEXT_PRIMARY),
        "glossate.text.secondary": Style(color=Colors.TEXT_SECONDARY),
        "glossate.text.muted": Style(color=Colors.TEXT_MUTED),
        "bar.back": Style(color=Colors.BG_PANEL),
        "bar.complete": Style(color=Colors.PRIMARY),
        "bar.finished": Style(color=Colors.PRIMARY_LIGHT),
        "bar.pulse": Style(color=Colors.PRIMARY_LIGHT),
        "progress.description": Style(color=Colors.TEXT_PRIMARY),
        "progress.percentage": Style(color=Colors.PRIMARY),
        "progress.remaining": Style(color=Colors.TEXT_SECONDARY),
        "progress.elapsed": Style(color=Colors.TEXT_SECONDARY),
        "progress.data.speed": Style(color=Colors.TEXT_SECONDARY),
        "progress.download": Style(color=Colors.TEXT_SECONDARY),
        "progress.filesize": Style(color=Colors.TEXT_SECONDARY),
        "progress.spinner": Style(color=Colors.PRIMARY),
        "glossate.panel.border": Style(color=Colors.PRIMARY),
        "glossate.panel.title": Style(color=Colors.PRIMARY, bold=True),
        "prompt": Style(color=Colors.TEXT_PRIMARY),
        "prompt.choices": Style(color=Colors.TEXT_MUTED),
        "prompt.default": Style(color=Colors.TEXT_MUTED),
        "prompt.invalid": Style(color=Colors.ERROR),
        "prompt.invalid.choice": Style(color=Colors.ERROR),
    }
)


class BoxChars:
    """Unicode box drawing characters."""

    TOP_LEFT = "╭"
    TOP_RIGHT = "╮"
    BOTTOM_LEFT = "╰"
    BOTTOM_RIGHT = "╯"
    HORIZONTAL = "─"
    VERTICAL = "│"

    BULLET_ACTIVE = "●"
    BULLET_PENDING = "○"
    BULLET_COMPLETE = "✓"
    BULLET_ERROR = "✗"


class Stages:
    """Pipeline stage definitions for GLOSSATE."""

    EXTRACT = ("extract", "Extracting audio", "", "Extracted audio")
    TRANSCRIBE = ("transcribe", "Transcribing", "", "Transcribed")
    TRANSLATE = ("translate", "Translating", "", "Translated")

    ALL = [EXTRACT, TRANSCRIBE, TRANSLATE]

    @classmethod
    def get_by_id(cls, stage_id: str) -> tuple[str, str, str, str] | None:
        for stage in cls.ALL:
            if stage[0] == stage_id:
                return stage
        return None
