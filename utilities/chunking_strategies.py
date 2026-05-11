"""
utilities/chunking_strategies.py — Five chunking algorithms
=============================================================
The default pipeline uses `RecursiveCharacterTextSplitter` (see
`backend/chunker.py`). It's a good default but it's not the ONLY way
to split text — and showing students the alternatives is one of the
most pedagogically useful comparisons in RAG.

We implement five strategies. Each takes a string and returns a list
of chunks (also strings).  The choice has REAL consequences:

  1. CHARACTER       — naive fixed-size slicing. Breaks mid-word.
                       Fastest. Worst respect for meaning.
  2. RECURSIVE       — what the default pipeline uses.
                       Tries paragraphs → lines → sentences → words.
                       Good general-purpose balance.
  3. TOKEN-BASED     — slices on LLM tokens instead of characters.
                       Most accurate respect for "how much will the
                       LLM eat" but requires a tokenizer.
  4. SENTENCE-BASED  — splits on sentence boundaries (regex / nltk).
                       Each chunk is N full sentences. Best for prose,
                       very poor for code or structured documents.
  5. SEMANTIC        — embed each sentence, find spots where consecutive
                       sentences are LEAST similar, split there.
                       Slowest but most "meaning-aware" — boundaries
                       fall at TOPIC CHANGES instead of word counts.

Each function has the same signature: (text, **kwargs) -> List[str].
The Playground UI compares all five side-by-side on the same input.
"""

import re
from typing import List, Optional


# ════════════════════════════════════════════════════════════
# 1. CHARACTER — fixed-size slicing
# ════════════════════════════════════════════════════════════

def chunk_character(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Slice the text every `chunk_size` characters, with `chunk_overlap`
    characters of overlap between consecutive chunks.

    NO respect for word boundaries — a chunk may start or end mid-word.
    This is the worst option for RAG but useful pedagogically as the
    "naive baseline" to compare the others against.
    """
    if not text:
        return []
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size  # safety: avoid infinite loop
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks


# ════════════════════════════════════════════════════════════
# 2. RECURSIVE — what the default pipeline uses
# ════════════════════════════════════════════════════════════

def chunk_recursive(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    LangChain's RecursiveCharacterTextSplitter. Tries the separators
    in order: paragraph → line → sentence → word → character.

    This is the same splitter the production pipeline uses; we expose
    it here for side-by-side comparison.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# ════════════════════════════════════════════════════════════
# 3. TOKEN-BASED — slice on LLM tokens, not characters
# ════════════════════════════════════════════════════════════

def chunk_token(
    text: str,
    chunk_tokens: int = 120,
    chunk_overlap_tokens: int = 12,
    tokenizer=None,
) -> List[str]:
    """
    Split on LLM-TOKENS, not characters. Each chunk has at most
    `chunk_tokens` tokens with `chunk_overlap_tokens` overlap.

    WHY THIS IS USEFUL:
        The LLM's context window is measured in TOKENS, not characters.
        With character-based chunking you have to GUESS how many tokens
        you're actually packing in — a chunk_size=500 character chunk
        could be anywhere between 80 and 200 tokens depending on the
        language and content. Token chunking gives you exact control.

    Args:
        text                 : input string
        chunk_tokens         : max tokens per chunk
        chunk_overlap_tokens : tokens shared between adjacent chunks
        tokenizer            : a HuggingFace tokenizer. If None we lazy-load
                               the embedding model's tokenizer.
    """
    if tokenizer is None:
        # Lazy import to avoid pulling transformers when not needed.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

    # Encode the entire text into token IDs.
    ids = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    step = chunk_tokens - chunk_overlap_tokens
    if step <= 0:
        step = chunk_tokens
    for i in range(0, len(ids), step):
        chunk_ids = ids[i:i + chunk_tokens]
        # Decode token IDs back to a string. skip_special_tokens=True
        # avoids leaking [CLS] / [SEP] / etc.
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if i + chunk_tokens >= len(ids):
            break
    return chunks


# ════════════════════════════════════════════════════════════
# 4. SENTENCE-BASED — N full sentences per chunk
# ════════════════════════════════════════════════════════════

# Simple regex-based sentence splitter. Doesn't handle abbreviations
# perfectly ("Dr. Smith" etc.) but is good enough for educational
# purposes and has zero extra dependencies.
_SENT_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\(])')


def split_into_sentences(text: str) -> List[str]:
    """Cheap regex sentence tokenizer. No dependencies."""
    text = re.sub(r"\s+", " ", text.strip())
    sentences = _SENT_END_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentence(
    text: str,
    sentences_per_chunk: int = 4,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Group N consecutive sentences into one chunk, with overlap.

    Best for narrative prose where sentences are the natural unit
    of meaning. Bad for code, lists, or text with lots of headings
    where "sentence" isn't a well-defined concept.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    chunks = []
    step = max(1, sentences_per_chunk - overlap_sentences)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
        if i + sentences_per_chunk >= len(sentences):
            break
    return chunks


# ════════════════════════════════════════════════════════════
# 5. SEMANTIC — split at topic-change points
# ════════════════════════════════════════════════════════════

def chunk_semantic(
    text: str,
    embedding_model=None,
    similarity_threshold: float = 0.55,
    min_chunk_sentences: int = 2,
) -> List[str]:
    """
    Embed each sentence, walk through them in order, and start a NEW
    chunk whenever consecutive sentences have low similarity (i.e. a
    topic change).

    Algorithm:
        1. Split text into sentences.
        2. Embed every sentence.
        3. For each adjacent pair (s_i, s_{i+1}), compute cosine
           similarity.
        4. If sim < threshold, that's a TOPIC BOUNDARY — close the
           current chunk and start a new one.
        5. Enforce min_chunk_sentences to avoid 1-sentence chunks.

    Args:
        text                 : input string
        embedding_model      : a SentenceTransformer. Auto-loaded if None.
        similarity_threshold : below this, split (0.5-0.7 is typical)
        min_chunk_sentences  : minimum sentences per chunk

    Returns:
        List of chunk strings.
    """
    import numpy as np

    sentences = split_into_sentences(text)
    if len(sentences) <= min_chunk_sentences:
        return [" ".join(sentences)]

    # Lazy-load the embedding model. If the caller has one in session
    # state (st_model from the main pipeline), they should pass it in
    # to avoid reloading.
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

    # Embed each sentence individually. Normalize so dot product = cosine sim.
    embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Walk through adjacent pairs and group.
    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        # If similarity is high → same topic → keep accumulating.
        # If low → topic change → flush current chunk, start a new one.
        if sim < similarity_threshold and len(current) >= min_chunk_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    # Don't forget the last chunk.
    if current:
        chunks.append(" ".join(current))

    return chunks


# ════════════════════════════════════════════════════════════
# Registry — used by the Playground UI
# ════════════════════════════════════════════════════════════

CHUNKING_STRATEGIES = {
    "character":  {
        "label":  "1. Character (naive)",
        "func":   chunk_character,
        "tagline": "Slice every N characters. Fast, breaks mid-word.",
    },
    "recursive":  {
        "label":  "2. Recursive (default)",
        "func":   chunk_recursive,
        "tagline": "Try paragraph → line → sentence → word. Good general-purpose.",
    },
    "token":      {
        "label":  "3. Token-based",
        "func":   chunk_token,
        "tagline": "Slice on LLM tokens. Exact context-window control.",
    },
    "sentence":   {
        "label":  "4. Sentence-based",
        "func":   chunk_sentence,
        "tagline": "Group N sentences. Best for narrative prose.",
    },
    "semantic":   {
        "label":  "5. Semantic",
        "func":   chunk_semantic,
        "tagline": "Embed sentences, split at topic changes. Smartest, slowest.",
    },
}
