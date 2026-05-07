"""
advanced/memory_compression.py — Conversation Memory Compression
==================================================================
THE PROBLEM
-----------
Right now, `RAGPipeline` keeps the last 4 turns verbatim in its
history. After a 20-turn conversation that's a LOT of text bloating
every prompt — and remember, prompt tokens cost LLM time AND eat into
the context window.

THE FIX
-------
Periodically REPLACE old turns with a SUMMARY produced by the LLM:

    Turn 1-2: User: ... / Assistant: ...
    Turn 3-4: User: ... / Assistant: ...
    Turn 5-6: User: ... / Assistant: ...
    Turn 7:   User: ... / Assistant: ...
    Turn 8:   User: ... / Assistant: ...

becomes:

    [SUMMARY of turns 1-6]: We discussed deep learning, then chunking
                            parameters, then how embeddings work.
    Turn 7:   User: ... / Assistant: ...
    Turn 8:   User: ... / Assistant: ...

We KEEP the most recent N turns verbatim because near-term context
matters more for follow-up questions.

WHEN TO COMPRESS
----------------
After every K turns, compress everything OLDER than the last N turns.
Defaults: K=2 (compress every 2 new turns), N=2 (keep last 2 verbatim).

COSTS vs BENEFITS
-----------------
Cost:    1 extra LLM call every K turns (~2-3 seconds).
Benefit: prompts stay bounded — a 50-turn conversation prompts about
         the same as a 5-turn one. Without compression, prompts grow
         linearly until the context window is full.

THIS IS WHAT CHATGPT-LIKE PRODUCTS DO INTERNALLY
------------------------------------------------
ChatGPT, Claude.ai etc. don't actually replay your entire chat history
for every message — they summarize older content the same way.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple


# ────────────────────────────────────────────────────────────
# Trace structure (for the playground UI)
# ────────────────────────────────────────────────────────────

@dataclass
class CompressionResult:
    """Result of a compression run."""
    summary: str = ""
    n_turns_compressed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    elapsed: float = 0.0

    @property
    def reduction_pct(self) -> float:
        if self.chars_before == 0:
            return 0.0
        return (1.0 - self.chars_after / self.chars_before) * 100.0


# ────────────────────────────────────────────────────────────
# Prompt template for the summarizer
# ────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """Summarize the following conversation in 2-3 short sentences.
Keep ALL information that might be relevant for follow-up questions
(names, numbers, decisions, preferences). Drop greetings, fillers, and acknowledgments.

Conversation:
{conversation}

Summary:"""


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def summarize_turns(
    turns: List[Tuple[str, str]],
    llm,
) -> str:
    """
    Ask the LLM to compress a list of (role, message) turns into a
    single summary paragraph.

    Args:
        turns : list of (role, message) tuples (role ∈ {"User", "Assistant"})
        llm   : a HuggingFacePipeline

    Returns:
        A short summary string.
    """
    if not turns:
        return ""

    # Format the turns into a clean conversation block.
    formatted = "\n".join(f"{role}: {msg}" for role, msg in turns)

    prompt = _SUMMARY_PROMPT.format(conversation=formatted)

    try:
        hf_pipe = llm.pipeline
        outputs = hf_pipe(
            prompt,
            max_new_tokens=150,        # 2-3 sentences
            do_sample=True,
            temperature=0.3,           # we want faithful summaries, not creative
            top_p=0.9,
            return_full_text=False,
            pad_token_id=hf_pipe.tokenizer.pad_token_id
                          or hf_pipe.tokenizer.eos_token_id,
        )
        summary = outputs[0]["generated_text"].strip() if outputs else ""
    except Exception as e:
        print(f"  ⚠️ Summarization failed ({e}); using truncated original.")
        summary = formatted[:300] + "..."

    return summary


def compress_history(
    history: List[Tuple[str, str]],
    llm,
    keep_last_n: int = 2,
) -> Tuple[List[Tuple[str, str]], CompressionResult]:
    """
    Compress everything older than the last N turns into a summary.

    Args:
        history     : list of (role, message) tuples — the full history
        llm         : LLM used for summarization
        keep_last_n : how many recent TURNS (not pairs) to keep verbatim

    Returns:
        (new_history, CompressionResult)
        new_history is: [("Summary", "...")] + last_n_turns
    """
    t0 = time.time()
    result = CompressionResult()

    # Nothing to compress.
    if len(history) <= keep_last_n:
        return list(history), result

    # Split into "to compress" and "to keep verbatim".
    to_compress = history[:-keep_last_n]
    to_keep     = history[-keep_last_n:]

    # Measure size BEFORE.
    result.chars_before = sum(len(m) for _, m in to_compress)
    result.n_turns_compressed = len(to_compress)

    # Run the summarizer.
    summary = summarize_turns(to_compress, llm)
    result.summary = summary
    result.chars_after = len(summary)
    result.elapsed = time.time() - t0

    # Build the new history: a single "Summary" turn followed by the kept ones.
    new_history: List[Tuple[str, str]] = [("Summary", summary)] + to_keep
    return new_history, result
