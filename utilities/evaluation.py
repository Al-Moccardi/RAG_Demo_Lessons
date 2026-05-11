"""
utilities/evaluation.py — RAG Evaluation Metrics
==================================================
HOW DO YOU KNOW IF YOUR RAG IS ANY GOOD?
----------------------------------------
A working pipeline isn't enough. You need to MEASURE quality on a
labelled test set. We compute six metrics across two families:

  LIGHTWEIGHT (always available, fast, no extra deps):
    • retrieval_precision_at_k : did any retrieved chunk contain the
                                  expected answer?
    • answer_similarity        : cosine similarity between the LLM's
                                  answer and the ground-truth answer
    • groundedness             : what fraction of answer n-grams appear
                                  in the retrieved context? (low = hallucination)
    • latency                  : wall-clock time per question

  RAGAS-BASED (when ragas + an LLM-as-judge are installed):
    • faithfulness             : RAGAS — does every CLAIM in the answer
                                  follow from the retrieved context?
    • answer_relevancy         : RAGAS — does the answer actually address
                                  the question?
    • context_precision        : RAGAS — how much of the retrieved context
                                  is useful, ranked by position?

WHY BOTH?
  RAGAS is the standard but it requires an extra LLM-judge call per
  metric per question. Lightweight metrics are instant and zero-cost
  but cruder. Showing both teaches students that there's no "one true
  number" for RAG quality.
"""

import re
import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════
# Test case + result containers
# ════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """A single labelled test case."""
    question:         str
    expected_answer:  str = ""    # ground truth — optional but enables most metrics
    expected_keywords: list = field(default_factory=list)  # keywords that MUST appear


@dataclass
class EvalResult:
    """Result of running ONE EvalCase through the pipeline."""
    case:                       EvalCase
    actual_answer:              str        = ""
    retrieved_chunks:           list       = field(default_factory=list)
    latency_s:                  float      = 0.0

    # Lightweight metrics (always computed)
    retrieval_precision_at_k:   float      = 0.0  # 0 or 1 typically
    answer_similarity:          float      = 0.0  # cosine, 0..1
    groundedness:               float      = 0.0  # fraction, 0..1
    keyword_hit_rate:           float      = 0.0  # fraction of expected keywords found

    # RAGAS metrics (filled by run_ragas if installed)
    faithfulness:               Optional[float] = None
    answer_relevancy:           Optional[float] = None
    context_precision:          Optional[float] = None


# ════════════════════════════════════════════════════════════
# 1. LIGHTWEIGHT METRICS
# ════════════════════════════════════════════════════════════

def _word_ngrams(text: str, n: int = 3) -> set:
    """Return the set of n-grams (lowercased) appearing in `text`."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}


def compute_groundedness(answer: str, contexts: List[str], n: int = 3) -> float:
    """
    Fraction of the answer's n-grams that ALSO appear in the retrieved
    context. A low score means the LLM is making things up.

    Returns a value in [0, 1]. Returns 0 if the answer is too short
    to form any n-gram.
    """
    answer_ngrams = _word_ngrams(answer, n=n)
    if not answer_ngrams:
        return 0.0
    context_ngrams: set = set()
    for ctx in contexts:
        context_ngrams |= _word_ngrams(ctx, n=n)
    if not context_ngrams:
        return 0.0
    return len(answer_ngrams & context_ngrams) / len(answer_ngrams)


def compute_keyword_hit_rate(answer: str, keywords: List[str]) -> float:
    """Fraction of expected keywords that appear (case-insensitively) in the answer."""
    if not keywords:
        return 1.0  # nothing to check → vacuously true
    answer_low = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_low)
    return hits / len(keywords)


def compute_retrieval_precision(
    retrieved_chunks_text: List[str],
    expected_answer: str,
    expected_keywords: List[str],
) -> float:
    """
    Did the retriever find a chunk that REALLY answers the question?

    Definition (proxy): score = 1 if ANY retrieved chunk contains
    the ground-truth answer OR ALL expected keywords; else 0.

    This isn't perfect but it's a defensible cheap proxy.
    """
    if not retrieved_chunks_text:
        return 0.0

    expected_low = expected_answer.lower().strip() if expected_answer else ""
    kw_low = [k.lower() for k in expected_keywords]

    for chunk in retrieved_chunks_text:
        chunk_low = chunk.lower()
        if expected_low and expected_low in chunk_low:
            return 1.0
        if kw_low and all(k in chunk_low for k in kw_low):
            return 1.0
    return 0.0


def compute_answer_similarity(
    answer: str,
    expected_answer: str,
    st_model,
) -> float:
    """
    Cosine similarity between the embedding of the LLM answer and the
    embedding of the ground-truth answer. Range: roughly [0, 1] (slightly
    negative is possible but rare).
    """
    if not answer or not expected_answer:
        return 0.0
    embeddings = st_model.encode(
        [answer, expected_answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return float(np.dot(embeddings[0], embeddings[1]))


# ════════════════════════════════════════════════════════════
# 2. END-TO-END: run a pipeline on a list of cases, get results
# ════════════════════════════════════════════════════════════

def run_evaluation(
    pipeline,
    cases: List[EvalCase],
    st_model,
    progress_callback=None,
) -> List[EvalResult]:
    """
    Run every test case through the pipeline and compute lightweight
    metrics. RAGAS metrics are added later by `run_ragas()` if available.

    Args:
        pipeline          : a RAGPipeline
        cases             : labelled test cases
        st_model          : SentenceTransformer (for answer similarity)
        progress_callback : optional fn(i, total, case) for UI updates

    Returns:
        List of EvalResult — same length as `cases`.
    """
    results: List[EvalResult] = []

    for i, case in enumerate(cases):
        if progress_callback:
            progress_callback(i, len(cases), case)

        t0 = time.time()
        response = pipeline.ask(case.question)
        latency = time.time() - t0

        # Pull out the raw text of every retrieved chunk for downstream metrics.
        retrieved_texts = [doc.page_content for doc, _score in response.sources]

        res = EvalResult(
            case=case,
            actual_answer=response.answer,
            retrieved_chunks=retrieved_texts,
            latency_s=latency,
        )

        # Lightweight metrics (always computed)
        res.retrieval_precision_at_k = compute_retrieval_precision(
            retrieved_texts, case.expected_answer, case.expected_keywords,
        )
        res.answer_similarity = compute_answer_similarity(
            response.answer, case.expected_answer, st_model,
        )
        res.groundedness = compute_groundedness(
            response.answer, retrieved_texts, n=3,
        )
        res.keyword_hit_rate = compute_keyword_hit_rate(
            response.answer, case.expected_keywords,
        )

        results.append(res)

    return results


# ════════════════════════════════════════════════════════════
# 3. RAGAS integration (optional)
# ════════════════════════════════════════════════════════════

def ragas_available() -> tuple:
    """Check whether the `ragas` package is importable."""
    try:
        import ragas  # noqa: F401
        from ragas.metrics import faithfulness, answer_relevancy, context_precision  # noqa: F401
        return True, f"RAGAS {getattr(ragas, '__version__', '?')} is installed."
    except ImportError as e:
        return False, (
            "RAGAS not installed. To enable LLM-as-judge metrics:\n"
            "    pip install ragas datasets\n\n"
            "RAGAS also needs a 'judge' LLM. With our local pipeline you can "
            "point RAGAS at the same HuggingFace model the app already loaded "
            "(see run_ragas()), but quality is best with a stronger judge "
            "(e.g. an OpenAI key in OPENAI_API_KEY)."
        )


def run_ragas(
    results: List[EvalResult],
    pipeline=None,
) -> List[EvalResult]:
    """
    Augment a list of EvalResult with RAGAS metrics.

    RAGAS uses an LLM as a judge to ask, for each (question, context,
    answer) triple:
        • faithfulness:      are all claims in the answer supported by context?
        • answer_relevancy:  does the answer actually address the question?
        • context_precision: are retrieved chunks ranked optimally for relevance?

    If RAGAS isn't installed OR the judge fails, the function falls back
    silently — the `faithfulness`/`answer_relevancy`/`context_precision`
    fields stay at None and the lightweight metrics still work.

    Args:
        results  : list of EvalResult produced by run_evaluation()
        pipeline : a RAGPipeline (used to extract the local LLM as judge
                   if no API key is available)
    """
    ok, _msg = ragas_available()
    if not ok:
        return results

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision,
        )
    except Exception as e:
        print(f"  ⚠️ RAGAS import failed: {e}")
        return results

    # Build the RAGAS-expected dataset shape.
    data = {
        "question":   [r.case.question for r in results],
        "answer":     [r.actual_answer for r in results],
        "contexts":   [r.retrieved_chunks for r in results],
        # RAGAS calls this "ground_truth" (singular, but expects strings).
        "ground_truth": [r.case.expected_answer or "" for r in results],
    }
    ds = Dataset.from_dict(data)

    try:
        # By default RAGAS uses langchain-openai. If the user has set
        # OPENAI_API_KEY it'll just work. Otherwise we'd need to wire
        # up a local judge here — left as a future extension since the
        # API surface keeps changing across RAGAS versions.
        report = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        df = report.to_pandas()
        for i, r in enumerate(results):
            r.faithfulness      = float(df.iloc[i].get("faithfulness", float("nan")))
            r.answer_relevancy  = float(df.iloc[i].get("answer_relevancy", float("nan")))
            r.context_precision = float(df.iloc[i].get("context_precision", float("nan")))
    except Exception as e:
        print(f"  ⚠️ RAGAS evaluation failed: {e}")

    return results


# ════════════════════════════════════════════════════════════
# 4. Built-in demo dataset (so the tab works without any upload)
# ════════════════════════════════════════════════════════════

DEMO_CASES = [
    EvalCase(
        question="What is deep learning?",
        expected_answer="Deep learning is a subset of machine learning that uses neural networks with many layers.",
        expected_keywords=["neural network", "layers"],
    ),
    EvalCase(
        question="What does a transformer model do?",
        expected_answer="Transformers use attention to process sequences and have revolutionized NLP.",
        expected_keywords=["attention", "NLP"],
    ),
    EvalCase(
        question="Who invented the Python programming language?",
        expected_answer="Python was created by Guido van Rossum.",
        expected_keywords=["Guido", "Rossum"],
    ),
    EvalCase(
        question="What is RAG?",
        expected_answer="Retrieval-Augmented Generation combines a retriever and a generator.",
        expected_keywords=["retrieval", "generation"],
    ),
    EvalCase(
        question="What does CNN stand for?",
        expected_answer="Convolutional Neural Network, commonly used for image tasks.",
        expected_keywords=["convolutional", "image"],
    ),
]
