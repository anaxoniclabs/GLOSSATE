# SPDX-License-Identifier: MIT
"""
GLOSSATE CLI.

Transcribe and translate video content. Audio in, subtitles out.
"""

import os
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .ui.ascii_art import get_header_panel
from .ui.components import create_console
from .ui.progress import GlossateProgress

_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac"})


def _clean_dragged_path(raw: str) -> str:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    for ch in (" ", "(", ")", "&", "'", '"', "$", ";", "\\"):
        s = s.replace("\\" + ch, ch)
    return s



def cmd_info(console) -> None:
    """Print system and model diagnostic information."""
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "not installed"

    from .ui.components import info_panel
    from .utils.device import get_memory_info, get_optimal_device
    from .utils.ffmpeg import get_ffmpeg_version
    from .utils.paths import get_app_dir

    device = get_optimal_device()
    ffmpeg_ver = get_ffmpeg_version() or "not found"

    mem = get_memory_info(device)
    memory_str = f"{mem['free_gb']:.1f} GB free" if "error" not in mem else "unavailable"

    console.print(
        info_panel(
            "System",
            {
                "Device": str(device),
                "PyTorch": torch_ver,
                "FFmpeg": ffmpeg_ver,
                "Free memory": memory_str,
            },
        )
    )
    console.print()

    app_dir = get_app_dir()

    import pathlib

    hf_cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    whisper_dirs = list(hf_cache.glob("models--*whisper*")) if hf_cache.exists() else []
    whisper_status = f"installed ({len(whisper_dirs)} variant(s))" if whisper_dirs else "not downloaded"

    gemma_dirs = list(hf_cache.glob("models--google--gemma*")) if hf_cache.exists() else []
    gemma_status = f"installed ({len(gemma_dirs)} variant(s))" if gemma_dirs else "not downloaded"

    console.print(
        info_panel(
            "Models",
            {
                "Whisper (ASR)": whisper_status,
                "Gemma (MT)": gemma_status,
                "Output dir": str(app_dir),
            },
        )
    )


def cmd_interactive(console) -> int:
    """Interactive prompt mode: shown when glossate is run with no arguments."""
    import argparse

    from rich.prompt import Prompt

    from .ui.theme import Colors

    console.clear()
    console.print(get_header_panel())
    console.print()

    try:
        from rich.panel import Panel
        from rich.text import Text

        prompt_text = Text()
        prompt_text.append(
            "Drag a video or an audio file here, or paste its path",
            style=f"bold {Colors.PRIMARY}",
        )
        console.print(Panel(prompt_text, border_style=Colors.PRIMARY, padding=(0, 2)))
        console.print()

        file_input = _clean_dragged_path(
            Prompt.ask(f"[{Colors.TEXT_SECONDARY}]File path[/]", console=console)
        )

        if not file_input:
            return 0

        console.print()
        console.print(f"  [{Colors.TEXT_SECONDARY}]Mode[/{Colors.TEXT_SECONDARY}]")
        console.print("    [dim]1.[/dim] Transcribe + Translate")
        console.print("    [dim]2.[/dim] Only Transcribe")
        mode = Prompt.ask("  Choose", choices=["1", "2"], console=console)

        console.print()
        console.print(f"  [{Colors.TEXT_SECONDARY}]Export[/{Colors.TEXT_SECONDARY}]")
        console.print("    [dim]1.[/dim] Subtitle [dim](.srt)[/dim]")
        console.print(f"       [{Colors.TEXT_MUTED}]Timed captions, ready for video players[/{Colors.TEXT_MUTED}]")
        console.print("    [dim]2.[/dim] Notes [dim](.md)[/dim]")
        console.print(f"       [{Colors.TEXT_MUTED}]Gemma-formatted readable Markdown[/{Colors.TEXT_MUTED}]")
        export = Prompt.ask("  Choose", choices=["1", "2"], console=console)

        md_scope = "translated"
        md_layout = "two-prose"
        md_timestamps = True
        if export == "2":
            fmt = "md"
            md_timestamps = (
                Prompt.ask(
                    f"  [{Colors.TEXT_SECONDARY}]Timestamps[/{Colors.TEXT_SECONDARY}]",
                    choices=["on", "off"],
                    default="on",
                    console=console,
                )
                == "on"
            )
            if mode == "1":
                md_scope = Prompt.ask(
                    f"  [{Colors.TEXT_SECONDARY}]Languages[/{Colors.TEXT_SECONDARY}]",
                    choices=["translated", "both"],
                    default="translated",
                    console=console,
                )
                if md_scope == "both":
                    md_layout = Prompt.ask(
                        f"  [{Colors.TEXT_SECONDARY}]Layout[/{Colors.TEXT_SECONDARY}]",
                        choices=["two-prose", "dual-stack"],
                        default="two-prose",
                        console=console,
                    )
        else:
            fmt = "srt"

        if mode == "1":
            target_raw = Prompt.ask(
                f"  [{Colors.TEXT_SECONDARY}]Target lang[/{Colors.TEXT_SECONDARY}]",
                console=console,
            ).strip()
        else:
            target_raw = ""

        source_raw = Prompt.ask(
            f"  [{Colors.TEXT_SECONDARY}]Source lang[/{Colors.TEXT_SECONDARY}]  [dim](or Enter to auto-detect)[/dim]",
            default="",
            show_default=False,
            console=console,
        ).strip()

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        return 130

    console.print()

    args = argparse.Namespace(
        input=file_input,
        target=target_raw or None,
        source=source_raw or None,
        format=fmt,
        md_scope=md_scope,
        md_layout=md_layout,
        md_timestamps=md_timestamps,
        output=None,
        asr_model="turbo",
        mt_model="gemma-4-e4b",
        device="auto",
        asr_backend="auto",
        mt_backend="auto",
        compute_type="auto",
        burn=False,
        burn_output=None,
        no_cache=False,
    )

    return cmd_run(args, console)


def cmd_run(args, console) -> int:
    """Transcribe and optionally translate an audio/video file."""
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {input_path}")
        return 1

    is_audio = input_path.suffix.lower() in _AUDIO_EXTENSIONS
    fmt: str = args.format
    source_lang: str | None = args.source
    md_text: str | None = None
    burn = bool(getattr(args, "burn", False))
    burn_output = getattr(args, "burn_output", None)

    if burn and is_audio:
        console.print("[red]Error:[/red] --burn requires a video input file")
        return 1
    if burn and fmt != "srt":
        console.print("[red]Error:[/red] --burn requires --format srt")
        return 1

    # --- Pipeline with progress panel ---
    from .utils.ffmpeg import get_audio_duration

    duration_s = 0.0
    try:
        duration_s = get_audio_duration(input_path)
    except Exception:
        pass

    duration_str = ""
    if duration_s > 0:
        m, s = divmod(int(duration_s), 60)
        duration_str = f"{m}:{s:02d}"

    from .core import cache as _cache

    cache_on = _cache.cache_enabled(not getattr(args, "no_cache", False))
    cache_key = (
        _cache.transcript_key(
            input_path, asr_model=args.asr_model, asr_backend=args.asr_backend, source=source_lang
        )
        if cache_on
        else None
    )
    cached_cues = _cache.load(cache_key)

    progress = GlossateProgress(has_translate=args.target is not None or fmt == "md")
    progress.start()
    progress.set_file_info(input_path.name, duration_str)

    try:
        # Stage 1: Extract (skipped on cache hit or for audio input)
        audio_path: Path = input_path
        is_temp = False
        detected_lang = ""

        if cached_cues is None and not is_audio:
            progress.begin_stage("extract")
            from .utils.ffmpeg import FFmpegError, extract_audio

            try:
                audio_path = extract_audio(input_path)
                is_temp = audio_path != input_path
            except FFmpegError as exc:
                progress.error_stage("extract", str(exc))
                progress.stop()
                return 1
            progress.complete_stage("extract")
        else:
            progress.skip_stage("extract")

        try:

            # Stage 2: Transcribe (or reuse cached transcript)
            from .core.segmenter import segment
            from .core.transcriber import WhisperError
            from .core.transcriber import transcribe as _transcribe

            progress.begin_stage("transcribe", total=int(duration_s) or 100)
            if cached_cues is not None:
                cues = cached_cues
                detected_lang = cues[0].lang if cues else ""
                progress.update_progress(
                    completed=int(duration_s) or 100, total=int(duration_s) or 100
                )
            else:
                try:
                    result = _transcribe(
                        audio_path,
                        source_lang=source_lang,
                        model=args.asr_model,
                        total_seconds=duration_s,
                        on_progress=lambda done, total: progress.update_progress(completed=int(done), total=int(total)),
                        device=args.device,
                        backend=args.asr_backend,
                        compute_type=args.compute_type,
                    )
                except WhisperError as exc:
                    progress.error_stage("transcribe", str(exc))
                    progress.stop()
                    return 1

                cues = segment({
                    "language": result.detected_language,
                    "segments": [
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "words": [
                                {"word": w.word, "start": w.start, "end": w.end}
                                for w in seg.words
                            ],
                        }
                        for seg in result.segments
                    ],
                })
                detected_lang = result.detected_language
                _cache.save(cache_key, cues)
            progress.complete_stage("transcribe")

            # Stage 3: Translate (srt) or build Gemma-formatted notes (md).
            # Markdown always needs the MT model — even with no --target — to
            # reflow the transcript, so it owns translation internally.
            if fmt != "md" and args.target is None:
                progress.skip_stage("translate")
            else:
                from .core.translator import TranslationError as MLXError
                from .core.translator import load_model as _load_mt
                from .core.translator import translate as _translate
                from .core.translator import unload_model as _unload_mt

                progress.begin_stage("translate", total=len(cues))
                mt_state = None
                try:
                    mt_state = _load_mt(
                        args.mt_model,
                        device=args.device,
                        backend=args.mt_backend,
                        compute_type=args.compute_type,
                    )
                    if fmt == "md":
                        from .core.notes import build_markdown

                        md_text = build_markdown(
                            cues,
                            source_lang=source_lang or detected_lang,
                            target_lang=args.target,
                            scope=args.md_scope,
                            layout=args.md_layout,
                            timestamps=args.md_timestamps,
                            model_state=mt_state,
                            on_progress=lambda done, total: progress.update_progress(completed=done, total=total),
                        )
                    else:
                        cues = _translate(
                            cues,
                            args.target,
                            source_lang or detected_lang,
                            model_state=mt_state,
                            on_progress=lambda done, total: progress.update_progress(completed=done, total=total),
                        )
                except MLXError as exc:
                    progress.error_stage("translate", str(exc))
                    progress.stop()
                    return 1
                finally:
                    if mt_state is not None and hasattr(mt_state, "backend"):
                        _unload_mt(mt_state)
                progress.complete_stage("translate")

        finally:
            if is_temp:
                audio_path.unlink(missing_ok=True)

    except KeyboardInterrupt:
        progress.stop()
        console.print("\n[dim]Cancelled.[/dim]")
        return 130

    except Exception as exc:
        progress.stop()
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    progress.stop()

    # Write output
    from .utils.paths import default_output_path

    out_path = Path(args.output) if args.output else default_output_path(input_path, fmt)  # type: ignore[arg-type]

    if fmt == "md":
        out_path.write_text(md_text or "", encoding="utf-8")
        written = out_path
    else:
        from .core.serializer import write_srt

        written = write_srt(cues, out_path)

    console.print(f"Written: {written}")

    if burn:
        from .utils.ffmpeg import FFmpegError, burn_subtitles
        from .utils.paths import default_burn_output_path

        video_out = Path(burn_output) if burn_output else default_burn_output_path(input_path)
        try:
            burned = burn_subtitles(input_path, written, video_out)
        except FFmpegError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1
        console.print(f"Burned: {burned}")

    return 0


def main() -> None:
    """Main entry point for GLOSSATE CLI."""
    parser = argparse.ArgumentParser(
        description="GLOSSATE — transcribe and translate video content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  glossate info                          Show device, models, and FFmpeg status
  glossate audio.wav                     Transcribe → audio.srt
  glossate audio.wav --target tr         Transcribe + translate to Turkish
  glossate audio.wav --source ar         Skip language detection
  glossate talk.mp4 --format md          Gemma-formatted notes (original lang)
  glossate talk.mp4 --target en --format md --md-scope both --md-layout dual-stack
  glossate video.mp4 --target en -o out.srt
  glossate video.mp4 --target tr --burn  Write subtitles and hard-burn them

Note: on CUDA (e.g. Colab), ASR uses batched faster-whisper and translation
runs Gemma 4 via transformers. Apple Silicon uses MLX (--mt-backend mlx).

Run 'glossate info' to check your setup.
        """,
    )
    parser.add_argument("--version", action="version", version=f"glossate {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Compute device for ASR and translation (default: auto)",
    )
    parser.add_argument(
        "--asr-backend",
        choices=["auto", "mlx", "faster-whisper"],
        default="auto",
        help="ASR backend (default: auto; CUDA selects faster-whisper)",
    )
    parser.add_argument(
        "--mt-backend",
        choices=["auto", "transformers", "mlx", "ollama"],
        default="auto",
        help="Translation backend (default: auto; CUDA/CPU selects transformers, Apple Silicon selects mlx)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="Model compute type, e.g. float16, bfloat16, int8_float16, int8, or auto",
    )
    parser.add_argument(
        "--asr-model",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default="turbo",
        dest="asr_model",
        help="Whisper model size (default: turbo)",
    )
    parser.add_argument(
        "--mt-model",
        default="gemma-4-e4b",
        dest="mt_model",
        metavar="MODEL",
        help=(
            "Translation model (default: gemma-4-e4b, Gemma 4 via transformers on CUDA/CPU). "
            "Gemma 4: gemma-4-e2b, gemma-4-e4b, gemma-4-26b, gemma-4-31b. "
            "MLX (Apple Silicon, --mt-backend mlx): qwen2.5-7b, translate-gemma-4b. "
            "Ollama: ollama/MODEL (requires Ollama running locally)."
        ),
    )

    parser.add_argument("input", nargs="?", metavar="FILE", help="Input audio or video file (or 'info')")
    parser.add_argument("--source", metavar="LANG", help="Source language code (skips detection)")
    parser.add_argument("--target", metavar="LANG", help="Target language for translation")
    parser.add_argument(
        "--format",
        choices=["srt", "md"],
        default="srt",
        help="Output format: srt (timed captions) or md (Gemma-formatted notes) (default: srt)",
    )
    parser.add_argument(
        "--md-scope",
        choices=["translated", "both"],
        default="translated",
        dest="md_scope",
        help="Markdown + --target: 'translated' (target only) or 'both' (bilingual) (default: translated)",
    )
    parser.add_argument(
        "--md-layout",
        choices=["two-prose", "dual-stack"],
        default="two-prose",
        dest="md_layout",
        help="Bilingual Markdown layout: two-prose (separate sections) or dual-stack (sentence pairs) (default: two-prose)",
    )
    parser.add_argument(
        "--md-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="md_timestamps",
        help="Include sparse timestamps in Markdown (default: on; use --no-md-timestamps to omit)",
    )
    parser.add_argument("-o", "--output", metavar="PATH", help="Output file path")
    parser.add_argument(
        "--burn",
        action="store_true",
        help="Hard-burn generated subtitles into the input video",
    )
    parser.add_argument(
        "--burn-output",
        metavar="PATH",
        help="Output video path for --burn (default: Documents/GLOSSATE/YYYY-MM-DD/<input>.subbed.mp4)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the transcript cache (re-transcribe even if a cached transcript exists)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    console = create_console()

    if args.input == "info":
        console.clear()
        console.print(get_header_panel())
        console.print()
        cmd_info(console)
        sys.exit(0)

    if args.input is None:
        sys.exit(cmd_interactive(console))

    sys.exit(cmd_run(args, console))


if __name__ == "__main__":
    main()
