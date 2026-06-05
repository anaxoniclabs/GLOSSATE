# GLOSSATE

GLOSSATE takes a video or audio file and hands you back what was said — either as subtitles to watch along with, or as clean notes to read. In the original language, translated, or both at once.

I built it because most of the things I wanted to learn from were talks in a language I only half-speak, and the captions were either missing or bad enough to make it worse. Subtitles scrolling past once were never really enough; what I wanted was a page I could read at my own pace and keep. So GLOSSATE does both — the captions for watching, and the page for reading.

It turned out to be useful past my own case, too: an accurate transcript to read if you're hard of hearing, original-and-translation stacked line by line if you're learning the language, or just a tidy set of notes instead of scrubbing back through an hour of audio.

It's a one-person project leaning hard on AI. Fork it and take it further.

## Quick start

```bash
pip install "glossate[cuda,detect]"   # NVIDIA / Colab; use [apple,detect] on Apple Silicon
glossate info                          # confirm device, FFmpeg, models
glossate talk.mp4 --target en          # → English subtitles next to your file
```

You'll need **FFmpeg** on your PATH (already there on Colab; `brew install ffmpeg` or `apt-get install -y ffmpeg` otherwise). Translation runs on **Gemma**, whose weights are gated — accept the license on [its model page](https://huggingface.co/google/gemma-4-E4B-it) and `huggingface-cli login` once. GLOSSATE is built CUDA-first and is happiest on a Colab GPU; there's a notebook that runs every feature end to end at [`colab_outputs_test.ipynb`](colab_outputs_test.ipynb).

It always transcribes. It only translates when you pass `--target`. Everything past that is your call.

## Watch it, or read it

That's really the one decision GLOSSATE asks you to make, and it maps to `--format`.

**Subtitles (`--format srt`, the default).** Timed captions for any player. With `--target` each line is translated where it sits and the timing is left alone. Add `--burn` and it bakes them into a copy of the video instead of leaving a sidecar file.

**Notes (`--format md`).** This is the part other subtitle tools don't do. Raw transcripts read like rubble — no punctuation, broken mid-thought. So for notes, Gemma reflows them into actual paragraphs (it reformats; it doesn't summarize or invent). You get notes in the original language by default, or in the target language with `--target`. Ask for both with `--md-scope both`, and choose how they sit together: `--md-layout two-prose` puts the original and the translation in separate sections, while `--md-layout dual-stack` pairs them sentence by sentence — which is what I use when I'm actually trying to learn:

```markdown
**[1:15]** Bu cümleyi anlamak istiyorum.

> I want to understand this sentence.
```

Those `[m:ss]` markers come from the cue timings in code, not from the model, so they stay accurate even after the words around them have been reflowed. Turn them off with `--no-md-timestamps` for a clean reading copy. (Dual-stack goes a step further than the captions: it rebuilds whole sentences from the fragments and translates them one sentence at a time, so the pairs actually line up grammatically.)

## It transcribes once

Transcription is the slow step, so GLOSSATE caches it per file. Run the same video again for a different language, a different format, or to burn it in, and it skips straight past the audio model to the cheap part. Disable with `--no-cache`, move it with `GLOSSATE_CACHE_DIR` (it lives in `~/.cache/glossate/transcripts` by default).

## Models and machines

Transcription is Whisper — `turbo` by default, down to `tiny` if you want it lighter (`--asr-model`). Translation and notes default to Gemma 4 (`gemma-4-e4b`); on a bigger GPU, `--mt-model gemma-4-26b` or `gemma-4-31b` buys you quality. On Apple Silicon the MLX path runs Gemma locally, and `ollama/MODEL` works if you'd rather serve a model through Ollama.

On `--device auto` it sorts out the backend itself: faster-whisper on CUDA/CPU, MLX on Apple Silicon, and the matching translation backend for each. After the weights download, nothing about your files leaves your machine. Outputs land in `~/Documents/GLOSSATE/`, dated.

## From Python

Same engine, for batch jobs or wiring into your own code. Open a `Session` so the models load once:

```python
import glossate

with glossate.Session(device="cuda") as s:
    for path in ["a.mp4", "b.mp4", "c.mp4"]:
        s.subtitle(path, target="en", format="md", md_scope="both", md_layout="dual-stack")
```

If you want to look at the transcript between steps, `glossate.transcribe()` gives you the timed cues and `glossate.translate()` translates them while keeping each original in `source_text`. `subtitle`, `subtitle_video`, `transcribe`, `translate`, `write`, `burn_subtitles`, `Session`, and the `GlossateError` hierarchy are the whole public surface.

## The edges

Things I'd rather you hear from me:

- The notes are only as good as the model. Gemma reflows and translates well, but on a long or messy transcript you'll occasionally catch an awkward seam or a too-literal line. Skim before you trust it.
- Sentence pairing in dual-stack is heuristic. It won't split a decimal or a `google.com`, but an abbreviation like `Dr.` can still end a sentence one word early.
- It runs on CPU, but transcribing a real talk and then running Gemma on it will test your patience. A GPU is the intended home.
- ASR mishears names, jargon, and crosstalk. Pass `--source` when you already know the language, and reach for a larger `--asr-model` when accuracy matters more than speed.

## License & thanks

GLOSSATE's code is MIT ([`LICENSE`](LICENSE)). It stands on [Whisper](https://github.com/openai/whisper) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) for transcription, [Gemma](https://ai.google.dev/gemma) for translation and notes (Google's [Gemma Terms](https://ai.google.dev/gemma/terms) — gated, with use restrictions), and [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Read the model licenses yourself before anything commercial — this isn't legal advice.

Issues and PRs welcome at <https://github.com/anaxoniclabs/GLOSSATE/issues>; `glossate --verbose <file>` prints a stage-by-stage log that helps with bug reports.
