# GLOSSATE

GLOSSATE turns a video or audio file into what was said: either subtitles to watch along with, or clean notes to read, in the original language, translated, or both stacked line by line. It's a command-line tool (with a small Python API) for anyone working through recorded talks, lectures, or interviews whose captions are missing, wrong, or in a language they only half-speak.

For me, as a non-native English speaker, the more technical the subject gets the harder it is to follow, and subtitles or nicely formatted notes make that easier.

The tool isn't limited to that, though: an accurate transcript if you're hard of hearing, original and translation side by side if you're learning a language, or just a tidy set of notes instead of scrubbing back through an hour of audio.

## Quick start

```bash
pip install "glossate[cuda,detect]"   # NVIDIA / Colab; use [apple,detect] on Apple Silicon
glossate info                          # check device, FFmpeg, and which models are downloaded
glossate talk.mp4 --target en          # → ~/Documents/GLOSSATE/<date>/talk.srt (English subtitles)
```

You need FFmpeg on your PATH (`brew install ffmpeg`, `apt-get install -y ffmpeg`; it's already there on Colab). Translation runs on Google's Gemma, whose weights are gated: accept the license on the [model page](https://huggingface.co/google/gemma-4-E4B-it) and run `huggingface-cli login` once. GLOSSATE is built CUDA-first and is happiest on a GPU; a Colab notebook that runs every feature end to end is at [`colab_outputs_test.ipynb`](colab_outputs_test.ipynb).

It always transcribes. It translates only when you pass `--target`. Run `glossate` with no arguments for an interactive prompt instead of flags.

## Watch it, or read it

That's the one real choice, and it maps to `--format`.

**Subtitles (`--format srt`, the default).** Timed captions for any player. With `--target`, each line is translated where it sits and the timing is left alone. Add `--burn` and it bakes the subtitles into a copy of the video (it keeps the `.srt` sidecar too).

**Notes (`--format md`).** Raw transcripts have no punctuation and break mid-thought, so for notes Gemma reflows them into actual paragraphs. It reformats; it doesn't summarize or invent. You get notes in the original language by default, the target language with `--target`, or both with `--md-scope both`. With both, `--md-layout two-prose` keeps the original and the translation in separate sections, while `--md-layout dual-stack` pairs them sentence by sentence:

```markdown
**[1:15]** Der Wirkungsgrad steigt mit der Temperatur.

> Efficiency increases with temperature.
```

Those `[m:ss]` markers come from the cue timings in code, not from the model, so they stay accurate after the words around them are reflowed. Drop them with `--no-md-timestamps`. Dual-stack rebuilds whole sentences from the transcript fragments and translates them one sentence at a time, so the pairs line up.

## It transcribes once

Transcription is the slow step, so GLOSSATE caches it per file. Run the same video again for a different language, a different format, or a burn-in, and it skips the audio model and goes straight to the cheap part. Turn it off with `--no-cache`, or move the cache with `GLOSSATE_CACHE_DIR` (default `~/.cache/glossate/transcripts`).

## Models and machines

Transcription is Whisper, `turbo` by default, down to `tiny` if you want it lighter (`--asr-model`). Translation and notes default to Gemma 4 (`gemma-4-e4b`); on a bigger GPU, `--mt-model gemma-4-26b` or `gemma-4-31b` buys quality, and `gemma-4-e2b` is lighter. On Apple Silicon the MLX path runs Gemma locally, and `--mt-model ollama/MODEL` serves a model through a local Ollama instead.

On `--device auto` it picks the backend: faster-whisper on CUDA/CPU, MLX on Apple Silicon, and the matching translation backend for each. After the weights download, nothing about your files leaves your machine. Outputs land in `~/Documents/GLOSSATE/`, in a dated folder.

A few common invocations:

```bash
glossate audio.wav                                # transcribe → audio.srt
glossate audio.wav --target tr                    # transcribe + translate to Turkish
glossate talk.mp4 --source ar --target en         # skip detection, translate Arabic → English
glossate talk.mp4 --target en --format md --md-scope both --md-layout dual-stack
glossate movie.mp4 --target tr --burn             # write subtitles and hard-burn them in
```

## From Python

Same engine, for batch jobs or wiring into your own code. Open a `Session` so the models load once:

```python
import glossate

with glossate.Session(device="cuda") as s:
    for path in ["a.mp4", "b.mp4", "c.mp4"]:
        s.subtitle(path, target="en", format="md", md_scope="both", md_layout="dual-stack")
```

`glossate.transcribe()` gives you the timed cues, and `glossate.translate()` translates them while keeping each original in `source_text`. The whole public surface is `transcribe`, `translate`, `subtitle`, `subtitle_video`, `burn_subtitles`, `write`, `Session`, the `Cue` dataclass, and the `GlossateError` exception hierarchy.

## Known limitations

- The notes are only as good as the model. Gemma reflows and translates well, but on a long or messy transcript you'll occasionally hit an awkward seam or a too-literal line.
- Sentence splitting in dual-stack is heuristic. A prose abbreviation like `Dr.` can still end a sentence one word early.
- It runs on CPU, but transcribing a real talk and then running Gemma over it is slow; you'll want a GPU.
- ASR mishears names, jargon, and crosstalk. Pass `--source` when you already know the language, and reach for a larger `--asr-model` when accuracy matters more than speed.

## License and credits

GLOSSATE's code is MIT (see [`LICENSE`](LICENSE)). It leans on [Whisper](https://github.com/openai/whisper) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) for transcription, Google's [Gemma](https://ai.google.dev/gemma) for translation and notes (gated, under the [Gemma Terms](https://ai.google.dev/gemma/terms), with use restrictions), and [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Read the model licenses yourself before any commercial use; this isn't legal advice.

Issues and PRs welcome at <https://github.com/anaxoniclabs/GLOSSATE/issues>. `glossate --verbose <file>` prints a stage-by-stage log that helps with bug reports.
