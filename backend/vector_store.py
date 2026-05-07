"""
backend/vector_store.py — FAISS Vector Store
==============================================
After every chunk has been embedded into a numerical vector, we need
some way to quickly find the K vectors closest to a given query
vector. Doing this naively would mean computing the dot product
between the query and EVERY stored vector — O(n) per query.

That works fine for a few hundred chunks. For thousands or millions,
we need a proper INDEXING data structure. That's what FAISS provides.

WHAT IS FAISS?
  Facebook AI Similarity Search — a C++ library with Python bindings,
  designed specifically for nearest-neighbor search over dense vectors.
  Even the simplest FAISS index (`IndexFlatL2`) is highly optimized
  with SIMD instructions; more advanced indexes (HNSW, IVF, ...)
  trade some accuracy for sub-linear search time.

WHY NOT A "REAL" VECTOR DATABASE?
  Real vector DBs (Pinecone, Weaviate, Qdrant, ...) add features like
  multi-tenancy, persistence, scaling, hybrid search, etc. For a
  teaching project all of that is overkill — FAISS in-memory + a
  simple save_local() to disk is plenty.

THREE OPERATIONS WE EXPOSE:
  1. build_vector_store  — create a fresh index from chunks
  2. save / load         — persist the index between sessions
  3. search              — given a query string, return top-K chunks
"""

from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import RAGConfig, DEFAULT_CONFIG, FAISS_INDEX_PATH
from backend.embedder import create_lc_embeddings


# ════════════════════════════════════════════════════════════
# 1. BUILD an index from chunks
# ════════════════════════════════════════════════════════════

def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings = None,
    config: RAGConfig = DEFAULT_CONFIG,
    save: bool = True,
) -> FAISS:
    """
    Build a FAISS index from a list of chunked Documents.

    `FAISS.from_documents()` is a convenience constructor that does
    THREE things in a single call:
      1. Run the embedding model on every chunk's `.page_content`
      2. Add each resulting vector to the FAISS index
      3. Keep a parallel mapping vector_id → original Document
         (so search results carry their text + metadata back to us)

    Args:
        chunks     : output of chunker.chunk_documents()
        embeddings : an embedding model (created here if None)
        config     : config object
        save       : if True, also persist the index to assets/faiss_index/

    Returns:
        A populated FAISS object ready for similarity_search().
    """
    # Lazy-init the embedding model if the caller didn't pass one.
    if embeddings is None:
        embeddings = create_lc_embeddings(config)

    # The big call: encode + index + link in one shot.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Auto-save so the user can quit Streamlit and come back without
    # having to re-encode every chunk.
    if save:
        save_vector_store(vectorstore)

    return vectorstore


# ════════════════════════════════════════════════════════════
# 2. SAVE / LOAD between sessions
# ════════════════════════════════════════════════════════════

def save_vector_store(vectorstore: FAISS, path: Path = FAISS_INDEX_PATH) -> None:
    """
    Write the index to disk.

    `save_local` writes two files: index.faiss (the raw FAISS index)
    and index.pkl (the documents + metadata). They sit in the
    directory `path`.
    """
    vectorstore.save_local(str(path))


def load_vector_store(
    embeddings: HuggingFaceEmbeddings = None,
    path: Path = FAISS_INDEX_PATH,
    config: RAGConfig = DEFAULT_CONFIG,
) -> Optional[FAISS]:
    """
    Load a previously-saved index, or return None if there isn't one.

    NOTE the `allow_dangerous_deserialization=True` flag: FAISS's
    .pkl file uses Python pickle, which can in principle execute
    arbitrary code on load. LangChain forces you to acknowledge this
    by setting the flag. It's safe in our case because WE wrote the
    file ourselves a moment ago — but never load random pickles
    you found on the internet.
    """
    index_file = path / "index.faiss"
    if not index_file.exists():
        return None

    if embeddings is None:
        embeddings = create_lc_embeddings(config)

    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ════════════════════════════════════════════════════════════
# 3. SEARCH
# ════════════════════════════════════════════════════════════

def search(
    vectorstore: FAISS,
    query: str,
    k: int = 5,
) -> List[tuple]:
    """
    Find the K chunks closest to a query string.

    `similarity_search_with_score` does this:
      1. Embed the query string with the SAME embedding model
         that was used to build the index.
      2. Ask FAISS for the K nearest stored vectors (L2 distance).
      3. Look up each result's original Document and return
         (Document, distance) pairs ordered by distance ascending
         — i.e. most similar FIRST.

    NOTE: the score is L2 DISTANCE, not similarity. Smaller = better.
    The chat UI converts this to a 0..1 "similarity-like" scale with
    `sim = max(0, 1 - score / 2)`.

    Returns:
        List of (Document, float) tuples.
    """
    return vectorstore.similarity_search_with_score(query, k=k)
