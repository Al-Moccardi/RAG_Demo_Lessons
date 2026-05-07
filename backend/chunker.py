"""
backend/chunker.py — Document Chunking
========================================
Chunking is the operation of splitting long documents into shorter,
overlapping passages. It happens AFTER loading and BEFORE embedding.

WHY DO WE CHUNK?
  1. LLMs have limited context windows (4k–128k tokens).
     We can't feed an entire textbook into one prompt.
  2. Smaller chunks → more PRECISE retrieval.
     Embedding a focused paragraph captures one specific topic;
     embedding a whole chapter dilutes everything together.
  3. Embedding models themselves have an input cap (typically 512 tokens
     ≈ 2000 characters). Bigger inputs are silently truncated.

THE SPLITTER WE USE: RecursiveCharacterTextSplitter
  This LangChain class tries a list of separators IN ORDER, falling back
  only when the previous one wouldn't keep chunks under `chunk_size`:

      1st priority:  "\n\n"  (paragraph break — strongest semantic boundary)
      2nd priority:  "\n"    (line break)
      3rd priority:  ". "    (sentence end)
      4th priority:  " "     (word boundary)
      Last resort:   ""      (split mid-word; rarely needed)

  This produces chunks that respect natural prose structure whenever
  possible. The OVERLAP between consecutive chunks ensures that a
  concept straddling a boundary isn't lost.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import RAGConfig, DEFAULT_CONFIG


def chunk_documents(
    documents: List[Document],
    config: RAGConfig = DEFAULT_CONFIG,
) -> List[Document]:
    """
    Split a list of (long) Documents into many short, overlapping chunks.

    Args:
        documents : output of data_loader.load_documents() —
                    typically one item per TXT and one per PDF page.
        config    : provides chunk_size, chunk_overlap, min_chunk_length.

    Returns:
        A NEW list of Document objects with the SAME metadata as their
        parent. Length will be much greater than `len(documents)`.
    """
    # Build the splitter once with the user's parameters.
    # Note: chunk_size and chunk_overlap are measured in CHARACTERS,
    # not tokens. This is approximate but good enough — for English,
    # 1 token ≈ 4 characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # `split_documents` preserves each Document's metadata in the children,
    # which is crucial — we later use metadata["source"] to cite chunks
    # in the chat UI.
    chunks = splitter.split_documents(documents)

    # Filter out chunks that are too short to be meaningful.
    # In practice these come from headers, footers, table-of-contents
    # entries, "References" lists, etc.
    chunks = [
        c for c in chunks
        if len(c.page_content.split()) >= config.min_chunk_length
    ]

    return chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Compute a few descriptive statistics about a chunked dataset.

    Used by the sidebar status box and the home-page metrics so the
    user can sanity-check their chunking parameters.
    """
    if not chunks:
        return {"count": 0}

    # `lengths` in characters, `tokens` is a rough word count.
    lengths = [len(c.page_content) for c in chunks]
    tokens = [len(c.page_content.split()) for c in chunks]

    # Unique source files, just to know how many docs the chunks span.
    sources = list(set(c.metadata.get("source", "?") for c in chunks))

    return {
        "count": len(chunks),
        "sources": len(sources),
        "avg_chars": sum(lengths) / len(lengths),
        "avg_tokens": sum(tokens) / len(tokens),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
    }
