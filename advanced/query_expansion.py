"""
advanced/query_expansion.py — Query Expansion for RAG
=======================================================
THE PROBLEM
-----------
A user types: "How does it work?"
The embedding model sees just those four words and looks for chunks
about "how" and "work" — it has no idea what "it" refers to. The
retrieval is going to be terrible.

THE FIX
-------
Before searching FAISS, we ask the LLM to REWRITE the query in a way
that's better suited for retrieval:

  • If it's vague, make it specific
  • If it's a single concept, generate 3-4 related sub-queries
  • If it depends on conversation history, resolve the references

Then we either:
  (a) Search using EACH expanded query, union the results, and dedupe.
      This is called "multi-query retrieval".
  (b) Concatenate the queries into one mega-query and search once.
      Faster but loses precision.

We implement (a) here because it's more interesting pedagogically.

WHY IT WORKS
------------
Embedding models are sensitive to wording. "How does deep learning
work?" and "What is the architecture of neural networks?" land in
DIFFERENT places in the 768-dim space, even though they're asking
related things. By searching from multiple angles we cover more
ground.

COST
----
Every query now requires:
  1 LLM call to expand (~1-3 seconds)
  + N FAISS searches instead of 1 (FAISS is fast — negligible)
  + post-processing to merge & dedupe

So we're trading ~2 extra seconds per query for noticeably better
retrieval. Worth it for hard questions, overkill for easy ones.
"""

from typing import List, Tuple
from langchain_core.documents import Document


# Prompt template used to ask the LLM to expand a query.
# Kept short and very directive — small models follow it best.
_EXPANSION_PROMPT = """You are a query expansion assistant for a search engine.

Given the user's question, rewrite it as 3 SHORT alternative search queries
that approach the same information need from different angles. The queries
should be diverse — use different keywords and synonyms.

Output ONLY the 3 queries, one per line. No numbering, no explanation.

User question: {question}

Three alternative search queries:"""


def expand_query(
    question: str,
    llm,
    n_queries: int = 3,
) -> List[str]:
    """
    Use the LLM to generate alternative phrasings of the user's question.

    Args:
        question  : the original user question
        llm       : a HuggingFacePipeline (we use llm.pipeline directly
                    for fine-grained control, same as rag_chain does)
        n_queries : how many alternatives to ask for

    Returns:
        A list of expanded queries. ALWAYS includes the original first.
    """
    prompt = _EXPANSION_PROMPT.format(question=question)

    # We deliberately use a slightly higher temperature (0.7) here than
    # the default 0.3 for answer generation — we WANT diversity in the
    # rewordings, not the most likely paraphrase three times.
    try:
        hf_pipe = llm.pipeline
        outputs = hf_pipe(
            prompt,
            max_new_tokens=120,           # 3 short queries → ~80 tokens
            do_sample=True,
            temperature=0.7,              # higher = more diverse
            top_p=0.95,
            return_full_text=False,
            pad_token_id=hf_pipe.tokenizer.pad_token_id
                          or hf_pipe.tokenizer.eos_token_id,
        )
        raw = outputs[0]["generated_text"] if outputs else ""
    except Exception as e:
        print(f"  ⚠️ Query expansion failed ({e}); using original only.")
        return [question]

    # Parse: keep non-empty lines, strip leading numbering ("1. ", "- ", etc.)
    lines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Drop empty lines and pure markers.
        if not line or line in ("-", "*"):
            continue
        # Strip common bullet/number prefixes.
        for prefix in ("1.", "2.", "3.", "4.", "5.", "-", "*", "•"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            lines.append(line)

    # Always include the original — it's our safety net if expansion went wrong.
    expanded = [question] + lines[:n_queries]

    # De-duplicate while preserving order (case-insensitive).
    seen, unique = set(), []
    for q in expanded:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def multi_query_search(
    queries: List[str],
    vectorstore,
    k_per_query: int = 3,
    final_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Search FAISS with EVERY query, then merge and rank the union.

    Args:
        queries     : list of (expanded) queries
        vectorstore : a FAISS instance
        k_per_query : how many chunks to fetch per individual query
        final_k     : how many chunks to keep AFTER merging

    Returns:
        Top-`final_k` (Document, score) tuples, sorted by best score.
    """
    # We dedupe by chunk content (a chunk found by 2 queries is the same chunk).
    # We keep the BEST score across all queries that found it.
    seen: dict = {}  # content_hash -> (Document, best_score)

    for q in queries:
        results = vectorstore.similarity_search_with_score(q, k=k_per_query)
        for doc, score in results:
            # Cheap content hash — first 200 chars is enough to dedupe.
            key = doc.page_content[:200]
            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    # Sort by score ascending (FAISS returns L2 distance — lower is better).
    merged = sorted(seen.values(), key=lambda t: t[1])
    return merged[:final_k]
