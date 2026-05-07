"""
backend/llm.py — Local Large Language Model
==============================================
This module loads an open-weight chat model entirely on the user's
machine. No API keys, no per-token charges, no data leaving the box.

Trade-offs vs. cloud APIs (GPT-4, Claude, ...):
  + Free at run time, no quotas
  + Total privacy
  + Works offline once the weights are cached
  - Smaller models (1.5–4 B params) are noticeably less capable than
    GPT-4-class models, so wording and reasoning may be weaker
  - First-time download of the weights is slow (1–7 GB)

WHY THE Qwen2.5-Instruct DEFAULT?
  - 1.5 B parameters fits comfortably on CPU machines (≈ 3 GB RAM)
  - Instruction-tuned, so it understands chat-style messages
  - Permissive license, multilingual (50+ languages)
  - Good speed/quality tradeoff for teaching demos

The function `load_llm()` wraps the raw HuggingFace pipeline in a
LangChain `HuggingFacePipeline`, which lets us call it with the same
.invoke() interface used elsewhere.
"""

import torch
from typing import Optional

# transformers gives us tokenizer + model + the high-level pipeline()
from transformers import (
    AutoTokenizer,                # auto-detects the correct tokenizer class
    AutoModelForCausalLM,         # generic causal LM (decoder-only)
    pipeline as hf_pipeline,      # high-level wrapper: text → text
    GenerationConfig,             # sampling hyperparameters
)
# LangChain's wrapper makes the HF pipeline behave like any other LangChain LLM
from langchain_huggingface import HuggingFacePipeline

from backend.config import RAGConfig, DEFAULT_CONFIG
from backend.embedder import get_device


# ════════════════════════════════════════════════════════════
# 1. LOAD the LLM and wrap it for LangChain
# ════════════════════════════════════════════════════════════

def load_llm(config: RAGConfig = DEFAULT_CONFIG) -> HuggingFacePipeline:
    """
    Download (if needed) and load the local LLM.

    Steps:
      1. Resolve the device (cuda / cpu)
      2. Load tokenizer
      3. Load model weights (fp16 on GPU, fp32 on CPU)
      4. Wrap in a transformers Pipeline
      5. Apply our generation hyperparameters
      6. Wrap THAT in HuggingFacePipeline for LangChain

    Returns:
        A LangChain HuggingFacePipeline that can be called with
        `.invoke(prompt_string)` and returns the generated text.
    """
    device = get_device()
    model_id = config.llm_model_name

    # ─── Tokenizer ─────────────────────────────────────
    # The tokenizer defines the vocabulary and turns text ↔ token ids.
    # For chat models, the tokenizer ALSO carries the chat template
    # which we use later in rag_chain.build_prompt().
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Some causal-LM tokenizers don't define a pad_token. Generation
    # warns us if it's missing, so we point pad_token at eos_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Model ─────────────────────────────────────────
    # float16 on GPU halves the memory at no cost in quality.
    # On CPU we MUST stay in float32 — most CPUs can't do fp16 fast.
    # device_map="auto" lets HF spread the model across GPUs if multiple.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # On CPU we manually move it (device_map=None doesn't trigger placement).
    if device == "cpu":
        model = model.to("cpu")

    # ─── HuggingFace high-level pipeline ──────────────
    # `text-generation` reads a prompt and continues it. We DON'T pass
    # `device_map` to pipeline() if the model is already on a specific
    # device — those parameters conflict with each other.
    pipe_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "task": "text-generation",
    }
    if device == "cuda":
        pipe_kwargs["device_map"] = "auto"

    pipe = hf_pipeline(**pipe_kwargs)

    # Set sampling temperature, max_tokens, etc. on the underlying model.
    update_generation_config(pipe, config)

    # ─── LangChain wrapper ────────────────────────────
    # `return_full_text=False` is critical: by default HF returns
    # PROMPT + COMPLETION; we want only the COMPLETION (the generated answer).
    return HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={"return_full_text": False},
    )


# ════════════════════════════════════════════════════════════
# 2. UPDATE the generation hyperparameters at runtime
# ════════════════════════════════════════════════════════════

def update_generation_config(pipe, config: RAGConfig) -> None:
    """
    Push the user's choice of temperature / top_p / max_tokens onto
    the loaded model WITHOUT having to re-load it.

    Called from app.py whenever the user clicks "Apply Settings".
    """
    # Pad token id is needed for batch generation. If the tokenizer
    # doesn't define one, fall back to the end-of-sequence token id.
    pad_id = pipe.tokenizer.pad_token_id or pipe.tokenizer.eos_token_id

    pipe.model.generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        do_sample=True,                        # use sampling, not greedy decoding
        temperature=config.temperature,        # ↑ = more random output
        top_p=config.top_p,                    # nucleus sampling
        repetition_penalty=config.repetition_penalty,  # > 1.0 = avoid loops
        pad_token_id=pad_id,
    )


# ════════════════════════════════════════════════════════════
# 3. POST-PROCESS the LLM output (clean up echoes, repetitions, ...)
# ════════════════════════════════════════════════════════════

# Markers that small models sometimes echo back from the prompt.
# When detected, we keep ONLY the longest segment between markers.
_ECHO_MARKERS = [
    "Answer:", "Context:", "Question:", "Human:",
    "A:", "Q:", "Assistant:", "System:",
]


def clean_llm_output(raw: str) -> str:
    """
    Remove prompt echoes and obvious repetitions from a raw LLM output.

    Even with a proper chat template, tiny LLMs occasionally:
      - Re-emit role markers ("Answer:", "Context:", ...)
      - Loop on the same sentence
      - Stop awkwardly mid-sentence
      - Echo the entire prompt back (when generation kwargs are ignored)

    This function is a safety net — it does light-touch cleanup
    so the user gets something tidy in the UI.

    Args:
        raw : the string returned by `llm.invoke(prompt)`

    Returns:
        A cleaned string. Empty if there was nothing salvageable.
    """
    text = str(raw).strip()
    if not text or text in ("None", "none", ""):
        return ""

    # ─── 0. Strip chat-template role tokens if any leaked through ───
    # When return_full_text=False is ignored, the output may still
    # contain <|im_start|>assistant\n at the start.
    for tok in ("<|im_start|>assistant", "<|im_start|>", "<|im_end|>",
                "<|begin_of_text|>", "<|eot_id|>"):
        text = text.replace(tok, "")
    text = text.strip()

    # ─── 1. Detect "the model echoed the context" failure ───
    # If the response contains BOTH "Context:" AND "[" (chunk source bracket),
    # the model has parroted the prompt back. Try to find the "Answer:" or
    # "assistant" marker and keep only what comes AFTER it.
    if "Context:" in text and "[" in text and "]:" in text:
        # Look for the LAST occurrence of an answer-like marker —
        # everything before it is echoed prompt.
        for sep in ("\nassistant\n", "Answer:", "\nAnswer\n"):
            if sep in text:
                text = text.rsplit(sep, 1)[-1].strip()
                break

    # ─── 2. Strip prompt echoes ───────────────────────
    # If a marker is present, split on it and keep the longest piece —
    # that's heuristically the "real" answer rather than an echo of the prompt.
    for marker in _ECHO_MARKERS:
        if marker in text:
            parts = text.split(marker)
            text = max(parts, key=len).strip()

    # ─── 3. Stop at the first repeated sentence ───────
    # Some small models start looping. We track the first 40 chars of each
    # sentence (lowercased) and stop as soon as we see a duplicate.
    sentences = text.split(". ")
    seen, clean = set(), []
    for s in sentences:
        key = s.strip().lower()[:40]
        if key in seen and len(key) > 5:
            break
        seen.add(key)
        clean.append(s)
    text = ". ".join(clean).strip()

    # ─── 4. Cosmetic: ensure trailing punctuation ─────
    if text and not text.endswith((".", "!", "?")):
        text += "."

    # If after all this the answer is too short, treat it as empty.
    return text if len(text) > 10 else ""
