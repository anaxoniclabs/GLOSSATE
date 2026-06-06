# SPDX-License-Identifier: MIT
"""Translation backend — Gemma/instruction LLMs via HF transformers, MLX-VLM, and Ollama."""

from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from glossate.core.segmenter import Cue

_log = logging.getLogger("glossate.translator")

# Gemma 4 (and other instruction LLMs) on CUDA/CPU/MPS via HF transformers.
# E4B (~16 GB bf16) is the default; bump to gemma-4-26b / gemma-4-31b on a
# large-VRAM A100 for higher quality.
_TRANSFORMERS_MODEL_MAP: dict[str, str] = {
    "gemma-4-e2b": "google/gemma-4-E2B-it",
    "gemma-4-e4b": "google/gemma-4-E4B-it",
    "gemma-4-26b": "google/gemma-4-26B-A4B-it",
    "gemma-4-31b": "google/gemma-4-31B-it",
}

# MLX-quantized variants (Apple Silicon only, opt-in via --mt-backend mlx).
_MLX_MODEL_MAP: dict[str, str] = {
    "qwen2.5-7b":         "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "translate-gemma-4b": "mlx-community/translategemma-4b-it-4bit",
    "gemma-4-e4b":        "mlx-community/gemma-4-e4b-it-8bit",
    "gemma-4-e2b":        "mlx-community/gemma-4-e2b-it-8bit",
}

_DEFAULT_MODEL = "gemma-4-e4b"

_DRAFTER_MAP: dict[str, str] = {
    "mlx-community/gemma-4-e4b-it-8bit": "mlx-community/gemma-4-E4B-it-assistant-bf16",
    "mlx-community/gemma-4-e2b-it-8bit": "mlx-community/gemma-4-E2B-it-assistant-bf16",
}

# ISO 639-1 → human name (used for instruction-LLM prompt construction)
_LANG_NAMES: dict[str, str] = {
    "ar": "Arabic", "en": "English", "tr": "Turkish", "fr": "French",
    "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "cs": "Czech", "hu": "Hungarian", "ro": "Romanian",
    "uk": "Ukrainian", "he": "Hebrew", "fa": "Persian", "hi": "Hindi",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
}


class TranslationError(RuntimeError):
    """Translation backend failed."""


# Back-compat alias
MLXError = TranslationError


@dataclass
class TranslationModelState:
    model: Any
    tokenizer: Any
    backend: str = "mlx"  # "transformers", "mlx", or "ollama"
    draft_model: Any = None
    device: str | None = None
    compute_type: str | None = None


# Back-compat alias
MLXModelState = TranslationModelState


_OLLAMA_PREFIX = "ollama/"
_OLLAMA_BASE_URL = "http://localhost:11434"


def _has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def load_model(
    model_name: str = _DEFAULT_MODEL,
    *,
    device: str | None = None,
    backend: str = "auto",
    compute_type: str | None = None,
) -> TranslationModelState:
    """Load the translation model.

    Dispatches to the transformers (Gemma/other instruction LLMs), MLX-VLM, or
    Ollama backend. In ``auto`` mode the transformers backend is used on
    CUDA/CPU and MLX on Apple Silicon; MLX is otherwise opt-in.
    """
    if backend not in {"auto", "mlx", "ollama", "transformers"}:
        raise TranslationError(f"unknown translation backend: {backend!r}")

    if model_name.startswith(_OLLAMA_PREFIX):
        if backend not in {"auto", "ollama"}:
            raise TranslationError(f"{model_name!r} is an Ollama model; requested backend={backend!r}")
        return _load_ollama(model_name[len(_OLLAMA_PREFIX):])

    try:
        import huggingface_hub.utils as _hf_utils
        _hf_utils.disable_progress_bars()
    except Exception:
        pass

    resolved = _resolve_mt_backend(model_name, backend, device)
    if resolved == "mlx":
        hf_id = _MLX_MODEL_MAP.get(model_name, model_name)
        _log.info("load_model: mlx %s", hf_id)
        return _load_mlx(hf_id, device=device, compute_type=compute_type)
    hf_id = _TRANSFORMERS_MODEL_MAP.get(model_name, model_name)
    _log.info("load_model: transformers %s", hf_id)
    return _load_transformers(hf_id, device=device, compute_type=compute_type)


def _resolve_mt_backend(model_name: str, backend: str, device: str | None) -> str:
    """Resolve the effective translation backend.

    Explicit backends win. In ``auto`` mode Apple Silicon routes to MLX and
    everything else routes to transformers.
    """
    if backend in {"mlx", "transformers"}:
        return backend
    if backend == "ollama":
        raise TranslationError("Ollama models must use the 'ollama/MODEL' name form.")
    from glossate.utils.device import get_optimal_device

    return "mlx" if get_optimal_device(device) == "mps" else "transformers"


def unload_model(state: TranslationModelState) -> None:
    """Release backend resources when the caller is done with a model."""
    if state.backend == "ollama":
        _unload_ollama(state)
        return
    if state.backend in {"transformers", "mlx"}:
        state.model = None
        state.tokenizer = None
        state.draft_model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_ollama(model_name: str) -> TranslationModelState:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise TranslationError(f"Ollama not reachable at {_OLLAMA_BASE_URL}: {exc}") from exc

    available = [m["name"] for m in data.get("models", [])]
    base = model_name.split(":")[0]
    found = any(m == model_name or m.startswith(base + ":") or m == base for m in available)
    if not found:
        raise TranslationError(
            f"Model {model_name!r} not found in Ollama. Run: ollama pull {model_name}"
        )
    _log.info("ollama: model %s ready", model_name)
    return TranslationModelState(model=model_name, tokenizer=None, backend="ollama")


def _unload_ollama(state: TranslationModelState) -> None:
    import json
    import urllib.request

    payload = json.dumps({"model": state.model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        _log.info("ollama: unloaded %s", state.model)
    except Exception:
        pass


def _torch_dtype(torch: Any, device: str, compute_type: str | None) -> Any:
    if compute_type in {None, "auto"}:
        return torch.float32 if device == "cpu" else torch.float16
    if compute_type in {"float16", "fp16"}:
        return torch.float16
    if compute_type in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if compute_type in {"float32", "fp32"}:
        return torch.float32
    raise TranslationError(f"unsupported compute_type: {compute_type!r}")


def _load_mlx(
    hf_id: str,
    *,
    device: str | None = None,
    compute_type: str | None = None,
) -> TranslationModelState:
    try:
        from glossate.utils.device import get_optimal_device

        resolved_device = get_optimal_device(device)
        if resolved_device != "mps":
            raise TranslationError(
                f"{hf_id!r} is an MLX model and requires Apple Silicon/MPS. Use the transformers backend (Gemma) or Ollama for {resolved_device}."
            )
        from mlx_vlm.utils import load
        model, processor = load(hf_id)
        draft_model = None
        drafter_id = _DRAFTER_MAP.get(hf_id)
        if drafter_id:
            from mlx_vlm.speculative.drafters import load_drafter
            draft_model = load_drafter(drafter_id, kind="mtp")
            _log.info("mtp drafter loaded: %s", drafter_id)
        return TranslationModelState(
            model=model,
            tokenizer=processor,
            backend="mlx",
            draft_model=draft_model,
            device=resolved_device,
            compute_type=compute_type or "auto",
        )
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(f"load_mlx failed: {exc}") from exc


def _load_transformers(
    hf_id: str,
    *,
    device: str | None = None,
    compute_type: str | None = None,
) -> TranslationModelState:
    """Load an instruction LLM (e.g. Gemma 4) for translation via HF transformers.

    Runs on CUDA, MPS, or CPU. bfloat16 is the default precision on GPU.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise TranslationError(
            "transformers backend requires torch and transformers>=5.5: pip install 'glossate[cuda]'"
        ) from exc
    try:
        from glossate.utils.device import get_optimal_device

        resolved_device = get_optimal_device(device)
        if compute_type in (None, "auto"):
            dtype = torch.float32 if resolved_device == "cpu" else torch.bfloat16
        else:
            dtype = _torch_dtype(torch, resolved_device, compute_type)

        # Gemma 4 is multimodal: load via AutoProcessor (it wraps the tokenizer).
        processor = AutoProcessor.from_pretrained(hf_id)
        tok = getattr(processor, "tokenizer", processor)
        try:
            tok.padding_side = "left"  # decoder-only batched generation
        except Exception:
            pass

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        use_device_map = resolved_device == "cuda" and _has_package("accelerate")
        if use_device_map:
            load_kwargs["device_map"] = "auto"
            load_kwargs["low_cpu_mem_usage"] = True
        if resolved_device == "cuda" and _has_package("flash_attn"):
            load_kwargs["attn_implementation"] = "flash_attention_2"
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        except TypeError:
            dtype_value = load_kwargs.pop("torch_dtype")
            model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype_value, **load_kwargs)
        if not use_device_map:
            model = model.to(torch.device(resolved_device))
        model.eval()
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None):
            tok.pad_token = tok.eos_token
        _log.info("transformers loaded %s on %s dtype=%s", hf_id, resolved_device, dtype)
        return TranslationModelState(
            model=model,
            tokenizer=processor,
            backend="transformers",
            device=resolved_device,
            compute_type=compute_type or "auto",
        )
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError(f"load_transformers failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Batch helpers (MLX path)
# ---------------------------------------------------------------------------

_BATCH_MAX_CUES = 20
_BATCH_MAX_CHARS = 2000
_OLLAMA_BATCH_MAX_CUES = 10
_OLLAMA_BATCH_MAX_CHARS = 1200
_TRANSFORMERS_BATCH_MAX_CUES = 20
_TRANSFORMERS_BATCH_MAX_CHARS = 2000
# A100-class GPUs have ample headroom over the ~16 GB E4B weights, so push far
# more cues per forward pass — throughput here is dominated by batch size.
_TRANSFORMERS_BATCH_MAX_CUES_CUDA = 96
_TRANSFORMERS_BATCH_MAX_CHARS_CUDA = 8000


def _make_batches(
    cues: list[Cue],
    max_cues: int = _BATCH_MAX_CUES,
    max_chars: int = _BATCH_MAX_CHARS,
) -> list[list[Cue]]:
    batches: list[list[Cue]] = []
    current: list[Cue] = []
    current_chars = 0
    for cue in cues:
        n = len(cue.text)
        if current and (len(current) >= max_cues or current_chars + n > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(cue)
        current_chars += n
    if current:
        batches.append(current)
    return batches


def _build_batch_prompt(cues: list[Cue], src_lang: str | None, tgt_lang: str) -> str:
    src_hint = f" from {src_lang}" if src_lang else ""
    tgt_name = _lang_name(tgt_lang)
    cue_xml = "\n".join(f'<cue id="{i + 1}">{c.text}</cue>' for i, c in enumerate(cues))
    return (
        f"Translate the following subtitle cues{src_hint} to {tgt_name}. "
        f"Return ONLY the translated cues in the same XML format, one per line. "
        f"No other text.\n\n{cue_xml}"
    )


def _parse_batch_response(response: str, expected: int) -> list[str] | None:
    matches = re.findall(r'<cue id="\d+">(.*?)</cue>', response, re.DOTALL)
    if len(matches) != expected:
        return None
    return [m.strip() for m in matches]


def _call_mlx(prompt: str, model_state: TranslationModelState) -> str:
    from mlx_vlm.generate import generate as vlm_generate

    kwargs: dict = dict(max_tokens=512, temperature=0.1, verbose=False)
    if model_state.draft_model is not None:
        kwargs.update(draft_model=model_state.draft_model, draft_kind="mtp", draft_block_size=6)

    result = vlm_generate(model_state.model, model_state.tokenizer, prompt=prompt, **kwargs)
    return result.text


# ---------------------------------------------------------------------------
# Public translate entry point
# ---------------------------------------------------------------------------


def translate(
    cues: list[Cue],
    target: str,
    source: str | None = None,
    *,
    model_state: TranslationModelState,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[Cue]:
    """Translate cues to *target* language, preserving timestamps."""
    total = len(cues)
    if total == 0:
        return []

    _log.info("translate: backend=%s tgt=%s cues=%d", model_state.backend, target, total)

    if model_state.backend == "ollama":
        return _translate_ollama_all(cues, source, target, model_state, on_progress)

    if model_state.backend == "transformers":
        return _translate_transformers_all(cues, source, target, model_state, on_progress)

    return _translate_mlx_all(cues, source, target, model_state, on_progress)


# ---------------------------------------------------------------------------
# Ollama path (one-at-a-time, with context window)
# ---------------------------------------------------------------------------


def _translate_ollama_all(
    cues: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
    on_progress: Optional[Callable[[int, int], None]],
) -> list[Cue]:
    result: list[Cue] = []
    processed = 0
    batches = _make_batches(cues, _OLLAMA_BATCH_MAX_CUES, _OLLAMA_BATCH_MAX_CHARS)
    for batch in batches:
        try:
            translated = _translate_ollama_batch(batch, source, target, state)
        except Exception as exc:
            raise TranslationError(f"translate failed: {exc}") from exc
        result.extend(translated)
        processed += len(batch)
        if on_progress:
            on_progress(processed, len(cues))
    return result


def _translate_ollama_batch(
    batch: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
) -> list[Cue]:
    prompt = _build_ollama_batch_prompt(batch, source, target)
    raw = _call_ollama(prompt, state)
    texts = _parse_batch_response(raw, len(batch))

    if texts is None:
        raw = _call_ollama(prompt, state)
        texts = _parse_batch_response(raw, len(batch))

    if texts is None:
        texts = [
            _translate_one_ollama(cue.text, source, target, state).strip()
            for cue in batch
        ]

    return [_with_translation(cue, text, target) for cue, text in zip(batch, texts)]


def _build_ollama_batch_prompt(cues: list[Cue], src_lang: str | None, tgt_lang: str) -> str:
    src_name = _lang_name(src_lang) if src_lang else None
    tgt_name = _lang_name(tgt_lang)
    src_hint = f" from {src_name}" if src_name else ""
    cue_xml = "\n".join(f'<cue id="{i + 1}">{c.text}</cue>' for i, c in enumerate(cues))
    return (
        f"Translate these subtitle cues{src_hint} to {tgt_name}. "
        f"Return ONLY the translated cues in the same XML format, one cue per line. "
        f"Do not merge, split, renumber, label, or explain anything.\n\n{cue_xml}"
    )


def _call_ollama(prompt: str, state: TranslationModelState) -> str:
    import json
    import urllib.request

    payload = json.dumps({
        "model": state.model,
        "messages": [
            {"role": "system", "content": "You are a subtitle translator. Output only the requested text."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1},
    }).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"]
    except Exception as exc:
        raise TranslationError(f"Ollama request failed: {exc}") from exc


def _translate_one_ollama(
    text: str,
    src_lang: str | None,
    tgt_lang: str,
    state: TranslationModelState,
    prev_source: str | None = None,
    prev_translated: str | None = None,
) -> str:
    src_name = _lang_name(src_lang) if src_lang else None
    tgt_name = _lang_name(tgt_lang)
    src_hint = f" from {src_name}" if src_name else ""

    system_content = (
        f"You are a subtitle translator{src_hint} to {tgt_name}. "
        f"Output only the translated text. No labels, no explanations."
    )
    user_content = text
    if prev_source and prev_translated:
        user_content = (
            f"[Previous line for context — do not translate this]\n{prev_source}\n"
            f"[Previous translation — do not output this]\n{prev_translated}\n"
            f"[Translate this line only]\n{text}"
        )

    raw = _call_ollama(f"{system_content}\n\n{user_content}", state)
    return _clean_response(raw, text)


# ---------------------------------------------------------------------------
# MLX-VLM path (batched, with MTP drafter when available)
# ---------------------------------------------------------------------------


def _translate_mlx_all(
    cues: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
    on_progress: Optional[Callable[[int, int], None]],
) -> list[Cue]:
    result: list[Cue] = []
    batches = _make_batches(cues)
    processed = 0

    for batch in batches:
        translated = _translate_batch(batch, source, target, state)
        result.extend(translated)
        processed += len(batch)
        if on_progress:
            on_progress(processed, len(cues))

    return result


def _translate_batch(
    batch: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
) -> list[Cue]:
    prompt = _build_batch_prompt(batch, source, target)

    response = _call_mlx(prompt, state)
    texts = _parse_batch_response(response, len(batch))

    if texts is None:
        response = _call_mlx(prompt, state)
        texts = _parse_batch_response(response, len(batch))

    if texts is None:
        texts = []
        for cue in batch:
            per_prompt = _build_batch_prompt([cue], source, target)
            raw = _call_mlx(per_prompt, state)
            texts.append(raw.strip() or cue.text)

    return [_with_translation(cue, t, target) for cue, t in zip(batch, texts)]


# ---------------------------------------------------------------------------
# HF transformers path (batched instruction LLM, e.g. Gemma 4 on CUDA)
# ---------------------------------------------------------------------------


def _translate_transformers_all(
    cues: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
    on_progress: Optional[Callable[[int, int], None]],
) -> list[Cue]:
    result: list[Cue] = []
    processed = 0
    if state.device == "cuda":
        max_cues, max_chars = _TRANSFORMERS_BATCH_MAX_CUES_CUDA, _TRANSFORMERS_BATCH_MAX_CHARS_CUDA
    else:
        max_cues, max_chars = _TRANSFORMERS_BATCH_MAX_CUES, _TRANSFORMERS_BATCH_MAX_CHARS
    for batch in _make_batches(cues, max_cues, max_chars):
        try:
            translated = _translate_transformers_batch(batch, source, target, state)
        except Exception as exc:
            raise TranslationError(f"translate failed: {exc}") from exc
        result.extend(translated)
        processed += len(batch)
        if on_progress:
            on_progress(processed, len(cues))
    return result


def _translate_transformers_batch(
    batch: list[Cue],
    source: str | None,
    target: str,
    state: TranslationModelState,
) -> list[Cue]:
    prompt = _build_batch_prompt(batch, source, target)
    response = _call_transformers([prompt], state)[0]
    texts = _parse_batch_response(response, len(batch))

    if texts is None:
        response = _call_transformers([prompt], state)[0]
        texts = _parse_batch_response(response, len(batch))

    if texts is None:
        per_prompts = [_build_batch_prompt([cue], source, target) for cue in batch]
        raws = _call_transformers(per_prompts, state)
        texts = [_clean_response(raw, cue.text) for raw, cue in zip(raws, batch)]

    return [_with_translation(cue, text, target) for cue, text in zip(batch, texts)]


def _resolve_eos_ids(tok: Any, model: Any) -> list[int]:
    """Stop tokens for generation, including Gemma's ``<end_of_turn>``.

    Greedy generation otherwise runs to ``max_new_tokens`` even after the
    model has emitted its turn terminator — and ``_clean_prose`` already
    discards everything past ``<end_of_turn>``, so those tokens are pure
    wasted decode. Stopping at them yields byte-identical text, faster.
    """
    ids: set[int] = set()
    cfg = getattr(model, "generation_config", None)
    cur = getattr(cfg, "eos_token_id", None) if cfg is not None else None
    if cur is not None:
        ids.update(cur if isinstance(cur, (list, tuple)) else [cur])
    base = getattr(tok, "eos_token_id", None)
    if isinstance(base, int):
        ids.add(base)
    unk = getattr(tok, "unk_token_id", None)
    for name in ("<end_of_turn>", "<eos>"):
        try:
            tid = tok.convert_tokens_to_ids(name)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            ids.add(tid)
    return sorted(ids)


def _call_transformers(
    prompts: list[str], state: TranslationModelState, *, max_new_tokens: int = 1024
) -> list[str]:
    """Run one batched chat-completion pass and return one decoded reply per prompt."""
    import torch

    processor = state.tokenizer
    model = state.model
    tok = getattr(processor, "tokenizer", processor)

    chats: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            # enable_thinking=False keeps Gemma 4 from emitting reasoning text.
            chat = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            chat = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        chats.append(chat)

    inputs = processor(text=chats, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    eos_ids = _resolve_eos_ids(tok, model)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tok, "pad_token_id", None),
            eos_token_id=eos_ids or None,
        )

    generated = outputs[:, input_len:]
    decoded = tok.batch_decode(generated, skip_special_tokens=True)
    return [text.strip() for text in decoded]


# ---------------------------------------------------------------------------
# Prose formatting pass (Markdown "notes" — reformat, do not translate)
# ---------------------------------------------------------------------------

_FORMAT_MAX_NEW_TOKENS = 2048
# Reflow many windows in one generate() pass on GPU; CPU/MPS stay sequential to
# avoid blowing memory for no decode-parallelism gain.
_FORMAT_BATCH_SIZE_CUDA = 8
_FORMAT_BATCH_SIZE = 1


def format_prose(text: str, lang: str, *, model_state: TranslationModelState) -> str:
    """Reformat a raw transcript chunk into clean, readable prose in *lang*.

    This is a *formatting* pass, not translation: punctuation and casing are
    fixed and fragments joined into sentences/paragraphs, with nothing added,
    dropped, summarized, reordered, or translated. Runs on any chat-capable
    backend (transformers / mlx / ollama).
    """
    text = text.strip()
    if not text:
        return ""
    backend = model_state.backend
    prompt = _build_format_prompt(text, lang)
    if backend == "transformers":
        raw = _call_transformers([prompt], model_state, max_new_tokens=_FORMAT_MAX_NEW_TOKENS)[0]
    elif backend == "ollama":
        system = "You reformat speech transcripts into clean prose. Output only the text."
        raw = _call_ollama(f"{system}\n\n{prompt}", model_state)
    else:  # mlx
        raw = _call_mlx(prompt, model_state)
    return _clean_prose(raw, text)


def format_prose_batch(
    texts: list[str],
    lang: str,
    *,
    model_state: TranslationModelState,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[str]:
    """Reflow several transcript windows, batching the generation pass.

    Each window's result is identical to calling :func:`format_prose` on it
    alone (greedy decode over an independent prompt); batching only removes the
    per-window ``generate()`` overhead, which is the dominant cost of the notes
    path. Empty windows pass through as ``""`` without a model call. Only the
    transformers backend batches; mlx/ollama fall back to sequential calls.
    """
    total = len(texts)
    if model_state.backend != "transformers":
        out: list[str] = []
        for i, text in enumerate(texts, 1):
            out.append(format_prose(text, lang, model_state=model_state))
            if on_progress:
                on_progress(i, total)
        return out

    size = _FORMAT_BATCH_SIZE_CUDA if model_state.device == "cuda" else _FORMAT_BATCH_SIZE
    results = [""] * total
    done = 0
    for start in range(0, total, size):
        group = range(start, min(start + size, total))
        idx: list[int] = []
        prompts: list[str] = []
        for i in group:
            stripped = (texts[i] or "").strip()
            if stripped:
                idx.append(i)
                prompts.append(_build_format_prompt(stripped, lang))
        if prompts:
            raws = _call_transformers(
                prompts, model_state, max_new_tokens=_FORMAT_MAX_NEW_TOKENS
            )
            for i, raw in zip(idx, raws):
                results[i] = _clean_prose(raw, (texts[i] or "").strip())
        done += len(group)
        if on_progress:
            on_progress(done, total)
    return results


def _build_format_prompt(text: str, lang: str) -> str:
    name = _lang_name(lang)
    return (
        f"Reformat the following {name} speech transcript into clean, readable prose. "
        f"Fix punctuation and capitalization, and join the fragments into natural "
        f"sentences and paragraphs. Do NOT translate it. Do NOT summarize, add, remove, "
        f"or reorder any content — keep every word's meaning and keep it in {name}. "
        f"Output only the reformatted text.\n\n{text}"
    )


def _clean_prose(raw: str, fallback: str) -> str:
    """Strip turn/control tokens and code fences without collapsing paragraphs."""
    for marker in ("<end_of_turn>", "<eos>", "<bos>", "<|im_end|>", "<|eot_id|>"):
        if marker in raw:
            raw = raw[: raw.index(marker)]
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw)
    raw = raw.strip()
    return raw or fallback


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _with_translation(cue: Cue, text: str, target: str) -> Cue:
    """Return a translated cue, preserving the original transcription as source_*."""
    new_text = text.strip() or cue.text
    return replace(
        cue,
        text=new_text,
        lang=target,
        source_text=cue.source_text or cue.text,
        source_lang=cue.source_lang or cue.lang,
    )


def _clean_response(raw: str, fallback: str) -> str:
    for tok in ("<|im_end|>", "<end_of_turn>", "<start_of_image>", "<eos>", "<bos>", "<|eot_id|>"):
        if tok in raw:
            raw = raw[:raw.index(tok)]
    for marker in ("[Translate this line only]", "[Previous line for context", "[Previous translation"):
        if marker in raw:
            idx = raw.rfind(marker)
            raw = raw[idx + len(marker):]
    raw = re.sub(r"```[a-zA-Z]*", "", raw)
    raw = re.sub(r"^-{2,}$", "", raw, flags=re.MULTILINE)
    raw = raw.replace("**", "")
    raw = re.sub(
        r"(?i)^(turkish|türkçe|tr|translation|çeviri|arabic|english)\s*:\s*", "", raw.strip()
    )
    raw = re.sub(r"\s+([.,!?;:])", r"\1", raw)
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    return lines[0] if lines else fallback


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(code.lower(), code)
