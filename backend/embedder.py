"""
backend/embedder.py — Text → Numerical Vectors
================================================
Embedding is the operation of converting a piece of text into a fixed-
length list of numbers (typically 384 or 768 dimensions). Two pieces
of text with similar MEANING land at nearby points in this vector
space — this is what makes semantic search possible.

EXAMPLE (intuition only — actual numbers are not interpretable):

    "Python is a programming language"  →  [ 0.12, -0.45, 0.78, ... ]
    "Java is a programming language"    →  [ 0.11, -0.43, 0.76, ... ]   ← close!
    "I love margarita pizza"            →  [ 0.89,  0.23,-0.56, ... ]   ← far away

  Cosine similarity between the first two will be ~0.95 (very similar);
  between sentence 1 and 3 it'll be ~0.10 (essentially unrelated).

WHY THE TWO WRAPPERS?
  - `HuggingFaceEmbeddings`  is a LangChain interface used by FAISS and
                             other LangChain components.
  - `SentenceTransformer`    is the underlying model used DIRECTLY for
                             analytics (e.g. plotting embeddings in 2D/3D
                             or computing similarity outside FAISS).

  Both load the SAME model weights — they're just two views of the
  same object.

WHY NORMALIZE?
  We pass `normalize_embeddings=True` to both wrappers. This makes
  every vector unit-length (norm 2 = 1). When all vectors are unit
  length, cosine similarity equals the dot product:

        cos(a, b) = (a · b) / (||a|| · ||b||) = a · b   (when ||a||=||b||=1)

  → much faster to compute on large batches.
"""

# ── Imports ───────────────────────────────────────────────
import torch                              # checks for GPU availability
import numpy as np                        # vector arithmetic
from typing import List

# Underlying model: returns numpy arrays directly.
from sentence_transformers import SentenceTransformer

# LangChain wrapper: implements the `embed_documents` / `embed_query`
# interface that FAISS expects.
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import RAGConfig, DEFAULT_CONFIG


# ────────────────────────────────────────────────────────────
# Device detection
# ────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Return "cuda" if an NVIDIA GPU is available, else "cpu".

    We don't currently support Apple Silicon's "mps" because
    sentence-transformers has subtle issues there with some models.
    Plain CPU is always a safe fallback.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────────────────────────────────────
# 1. LangChain-style embedding model (used by FAISS)
# ────────────────────────────────────────────────────────────

def create_lc_embeddings(config: RAGConfig = DEFAULT_CONFIG) -> HuggingFaceEmbeddings:
    """
    Instantiate a LangChain-compatible embedding model.

    This object exposes `.embed_documents(list[str])` and `.embed_query(str)`,
    which are the methods FAISS calls during indexing and search.
    """
    return HuggingFaceEmbeddings(
        # The HF Hub model id — first call downloads, subsequent calls
        # use the local cache (~/.cache/huggingface).
        model_name=config.embedding_model_name,

        # `model_kwargs` is forwarded to the underlying SentenceTransformer
        # at load time. We use it just to pin the device.
        model_kwargs={"device": get_device()},

        # `encode_kwargs` is passed at every encode() call. Critical:
        # normalize_embeddings makes cosine sim = dot product later.
        encode_kwargs={"normalize_embeddings": True},
    )


# ────────────────────────────────────────────────────────────
# 2. Raw SentenceTransformer (used by analytics & playground)
# ────────────────────────────────────────────────────────────

def create_st_model(config: RAGConfig = DEFAULT_CONFIG) -> SentenceTransformer:
    """
    Instantiate the underlying SentenceTransformer directly.

    Why? Because for visualisation we want the raw numpy array, not
    the LangChain wrapper. The model weights are SHARED with the
    LangChain wrapper — both download/load only once thanks to HF caching.
    """
    return SentenceTransformer(config.embedding_model_name, device=get_device())


# ────────────────────────────────────────────────────────────
# 3. Batch-encode raw text → numpy matrix
# ────────────────────────────────────────────────────────────

def embed_texts(
    texts: List[str],
    model: SentenceTransformer = None,
    config: RAGConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Encode a list of strings into a 2D numpy array of shape (n, dim).

    Args:
        texts  : list of strings
        model  : optional pre-loaded SentenceTransformer
                 (creates one if None — slow if called repeatedly!)
        config : config object

    Returns:
        numpy.ndarray of shape (len(texts), embedding_dim).
        Each row is a unit-length vector when normalize_embeddings=True.
    """
    # Lazy-load the model only if the caller didn't already have one.
    # In the real pipeline (app.py), the model is created ONCE during
    # "Build Pipeline" and reused everywhere → fast.
    if model is None:
        model = create_st_model(config)

    return model.encode(
        texts,
        batch_size=64,                       # bigger = faster but more RAM
        show_progress_bar=len(texts) > 50,   # only show bar for large jobs
        convert_to_numpy=True,               # we want np arrays, not torch tensors
        normalize_embeddings=True,           # so cosine sim == dot product
    )


# ────────────────────────────────────────────────────────────
# 4. Cosine similarity between query and a batch of docs
# ────────────────────────────────────────────────────────────

def compute_similarity(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between one query and every document.

    Both inputs are assumed to be NORMALIZED, so we can skip the
    division by norm — the dot product alone gives cosine similarity.

    Args:
        query_embedding : shape (dim,) or (1, dim)
        doc_embeddings  : shape (n, dim)

    Returns:
        1D numpy array of length n, where higher values = more similar.
    """
    # `.T` is a no-op when query is 1D; here it just makes the shapes
    # broadcast correctly. `.flatten()` collapses the result to 1D.
    return np.dot(doc_embeddings, query_embedding.T).flatten()
