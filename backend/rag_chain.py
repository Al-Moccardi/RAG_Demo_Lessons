"""
backend/rag_chain.py — The Complete RAG Pipeline
==================================================
This is where everything comes together. The class `RAGPipeline`
exposes a single user-facing method, `ask(question)`, that runs the
full Retrieval-Augmented Generation flow:

       ┌─────────────────────────────────────────────┐
       │   1. Retrieve top-K relevant chunks         │
       │   2. Build a chat-formatted prompt          │
       │   3. Send it to the LLM                     │
       │   4. Clean the LLM's output                 │
       │   5. Update conversation history            │
       │   6. Return a structured RAGResponse        │
       └─────────────────────────────────────────────┘

────────────────────────────────────────────────────────
THE #1 GOTCHA: chat templates
────────────────────────────────────────────────────────
All modern instruct-tuned LLMs (Qwen, SmolLM2, Phi-3, …) are trained
with a strict role-based format:

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant

If you send them a plain string ending in "Answer:", they tend to
*continue the pattern of the context* — i.e. they regurgitate the
chunks instead of answering.  That's the bug we explicitly fix here
by calling `tokenizer.apply_chat_template()`.

That call is model-AGNOSTIC: each tokenizer ships with the correct
template for its model, so swapping Qwen → SmolLM → Phi-3 just works.
"""

import os
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline

from backend.config import RAGConfig, DEFAULT_CONFIG
from backend.llm import clean_llm_output


# ════════════════════════════════════════════════════════════
# Structured response object
# ════════════════════════════════════════════════════════════

@dataclass
class RAGResponse:
    """
    Everything the chat UI needs to render a single Q&A turn.

    Fields:
        answer    : the cleaned LLM output (the actual reply shown to the user)
        sources   : list of (Document, distance) pairs — the chunks the LLM saw
        query     : the original user question (echoed back for the UI)
        elapsed   : seconds the whole ask() call took (for the "thinking …" UI)
        n_chunks  : how many chunks were retrieved
        prompt    : the EXACT prompt string that was sent to the LLM
                    (kept for debugging in the Playground tab)
    """
    answer: str
    sources: List[Tuple[Document, float]]
    query: str
    elapsed: float
    n_chunks: int
    prompt: str = ""


# ════════════════════════════════════════════════════════════
# The pipeline class
# ════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Glue object holding references to every piece of the RAG system.

    Usage
    -----
    >>> pipeline = RAGPipeline(vectorstore, llm, config)
    >>> response = pipeline.ask("What is deep learning?")
    >>> print(response.answer)
    >>> for doc, score in response.sources:
    ...     print(score, doc.metadata["source"])
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        vectorstore,                      # FAISS instance
        llm: HuggingFacePipeline,         # local LLM
        config: RAGConfig = DEFAULT_CONFIG,
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config

        # Conversation memory: a flat list of (role, message) tuples.
        # We only ever feed the last 4 entries back to the LLM, so this
        # stays bounded.
        self.history: List[Tuple[str, str]] = []

        # Try to grab the underlying HF tokenizer — without it we can't
        # apply the chat template and the bug at the top of this file
        # WILL bite us. See `_get_tokenizer` below.
        self.tokenizer = self._get_tokenizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        """
        Best-effort lookup of the underlying HuggingFace tokenizer.

        The LangChain HuggingFacePipeline keeps the original transformers
        Pipeline as `.pipeline`. From there `.tokenizer` is the AutoTokenizer
        we loaded in llm.py. If the user wired things differently (custom
        LLM class, etc.) we silently return None and fall back to plain text.
        """
        try:
            return self.llm.pipeline.tokenizer
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Prompt construction (public — exposed for the Playground tab)
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        context: str,
        history: str,
        question: str,
    ) -> str:
        """
        Build the FINAL string sent to the LLM.

        We always wrap our content in a system + user message pair:

            system  →  the "instructions" (config.system_prompt)
            user    →  context + history + question

        If the tokenizer exposes `apply_chat_template`, we call it so
        the output uses the model's role tokens (e.g. <|im_start|>...).
        Otherwise we fall back to a plain-text format ending in "Answer:".
        """
        # The user message is a single string assembled from three parts.
        # Order matters: we put the question LAST and label it loudly so the
        # model's attention is drawn to it. We also add a final instruction
        # ("Answer in your own words...") because small LLMs behave better
        # with one extra reminder right before they generate.
        if history and history != "(empty)":
            history_block = f"\n\nPrevious conversation:\n{history}"
        else:
            history_block = ""

        user_msg = (
            f"Use the following context to answer the question.\n\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
            f"{history_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer in your own words, using 2-4 sentences."
        )

        # ─── Preferred path: chat template ─────────────
        # apply_chat_template knows about every supported model's
        # exact role syntax.  This is what makes the LLM behave as
        # an assistant rather than a pattern-completer.
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user",   "content": user_msg},
                ]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,           # we want a string, not token ids
                    add_generation_prompt=True,  # adds "<|im_start|>assistant\n"
                )
            except Exception as e:
                # Some tokenizers refuse a "system" role, etc.
                # Print a warning and fall through to plain text.
                print(f"  ⚠️ apply_chat_template failed ({e}); using plain prompt.")

        # ─── Fallback: plain text ───────────────────────
        # Works for non-chat models or unfamiliar templates.
        return (
            f"{self.config.system_prompt}\n\n"
            f"{user_msg}\n\n"
            f"Answer:"
        )

    # ------------------------------------------------------------------
    # The ONE public method: ask a question, get a structured answer
    # ------------------------------------------------------------------

    def ask(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Steps:
          1. Retrieve top-K relevant chunks (FAISS similarity search)
          2. Build the conversational context string
          3. Format history (last 4 turns)
          4. Apply the model's chat template → final prompt
          5. Invoke the LLM
          6. Clean the raw output
          7. Update self.history
          8. Return a RAGResponse

        Args:
            question: the user's question.

        Returns:
            A RAGResponse with the answer, sources, timing, and the
            actual prompt that was sent (for debugging).
        """
        # Wall-clock timer for the elapsed field of the response.
        t0 = time.time()

        # ─── 1. RETRIEVE ───────────────────────────────
        # `as_retriever` produces a LangChain Retriever object whose
        # .invoke(query) returns Documents (no scores).
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.k}
        )
        docs = retriever.invoke(question)

        # We ALSO want similarity scores for the UI bar visualisation,
        # so we call similarity_search_with_score separately. (The two
        # calls share an embedding cache so the cost is essentially zero.)
        scored = self.vectorstore.similarity_search_with_score(
            question, k=self.config.k
        )

        # ─── 2. BUILD CONTEXT ──────────────────────────
        # Concatenate the retrieved chunks into one string. Each chunk
        # is prefixed with its source filename → handy if the LLM cites it.
        # We trim chunks to 400 chars to keep the prompt short; this is
        # a deliberate tradeoff between context richness and prompt length.
        context = "\n\n".join(
            f"[{os.path.basename(d.metadata.get('source', '?'))}]: "
            f"{d.page_content[:400]}"
            for d in docs
        )

        # ─── 3. BUILD HISTORY ──────────────────────────
        # We give the LLM the last 4 turns (2 Q&A pairs) so it remembers
        # follow-up questions like "tell me more about that".
        if self.history:
            hist_str = "\n".join(
                f"{role}: {msg[:150]}"  # cap each line at 150 chars
                for role, msg in self.history[-4:]
            )
        else:
            hist_str = "(empty)"

        # ─── 4. BUILD PROMPT ───────────────────────────
        # Apply the model's chat template (or plain-text fallback).
        prompt = self.build_prompt(context, hist_str, question)

        # ─── 5. GENERATE ───────────────────────────────
        # We bypass the LangChain wrapper here and call the underlying
        # transformers pipeline DIRECTLY. Why?
        #
        #   • LangChain's HuggingFacePipeline ignores some critical
        #     generation kwargs (notably `return_full_text=False` is
        #     unreliable across versions), so the wrapper sometimes
        #     hands us back PROMPT + COMPLETION instead of just
        #     COMPLETION. That's exactly how the model "echoes" the
        #     context back to the user.
        #
        #   • Calling the HF pipeline directly gives us:
        #       - explicit `return_full_text=False`
        #       - explicit `do_sample=True` and EOS handling
        #       - one less abstraction layer to debug
        #
        # We then strip the prompt prefix as a defensive safety net
        # in case `return_full_text=False` is ignored anyway.
        try:
            hf_pipe = self.llm.pipeline                # transformers Pipeline

            outputs = hf_pipe(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                return_full_text=False,                # we want ONLY the completion
                pad_token_id=hf_pipe.tokenizer.pad_token_id
                              or hf_pipe.tokenizer.eos_token_id,
            )

            # The HF pipeline returns a list of dicts: [{"generated_text": "..."}]
            raw = outputs[0]["generated_text"] if outputs else ""

            # Defensive safety net: if the pipeline ignored `return_full_text`
            # (it can happen with custom wrappers), strip the prompt prefix
            # ourselves so we never echo the context back to the user.
            if raw.startswith(prompt):
                raw = raw[len(prompt):]

        except Exception as e:
            raw = ""
            print(f"LLM generation error: {e}")

        # ─── 6. CLEAN ──────────────────────────────────
        # Even with a chat template, tiny LLMs sometimes echo markers or
        # repeat themselves. clean_llm_output() handles that.
        answer = clean_llm_output(raw) if raw else ""

        # If cleaning left us with nothing usable, fall back gracefully:
        if not answer and docs:
            answer = (
                "I could not synthesize a clean answer. The most relevant "
                "passage I found was:\n\n"
                f"> {docs[0].page_content[:400]}"
            )
        elif not answer:
            answer = "I could not generate an answer. Please try rephrasing your question."

        # ─── 7. UPDATE MEMORY ──────────────────────────
        # Append both turns to history so the NEXT call sees them.
        self.history.append(("User", question))
        self.history.append(("Assistant", answer))

        # ─── 8. RETURN ─────────────────────────────────
        elapsed = time.time() - t0
        return RAGResponse(
            answer=answer,
            sources=scored,        # for the UI source-cards
            query=question,        # echoed back, the UI uses it as a header
            elapsed=elapsed,       # for the "took 2.4s" indicator
            n_chunks=len(docs),    # for the "📚 5 sources" expander label
            prompt=prompt,         # for the Playground tab
        )

    # ------------------------------------------------------------------
    # Small public helpers
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Wipe conversation memory. Called by the 'Clear chat' button."""
        self.history.clear()

    def update_config(self, config: RAGConfig) -> None:
        """
        Swap in a new config (different k, temperature, system_prompt, ...).
        No need to rebuild the chain — the new values are read fresh on
        every ask() call.
        """
        self.config = config
