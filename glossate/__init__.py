# SPDX-License-Identifier: MIT
"""
GLOSSATE - Transcribe and translate video content.

Public API: ``transcribe``, ``translate``, ``subtitle``, ``subtitle_video``,
``burn_subtitles``, ``write``,
``Session``, ``GlossateError``, ``main``, ``__version__``.
"""

__version__ = "0.0.2"

from .api import (
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
    write,
)
from .core.segmenter import Cue


def main() -> None:
    """Run the GLOSSATE CLI."""
    from .cli import main as _main

    _main()

__all__ = [
    "AudioExtractionError",
    "SubtitleBurnError",
    "Cue",
    "ModelNotInstalledError",
    "GlossateError",
    "Session",
    "TranscriptionError",
    "TranslationError",
    "__version__",
    "main",
    "burn_subtitles",
    "subtitle",
    "subtitle_video",
    "transcribe",
    "translate",
    "write",
]
