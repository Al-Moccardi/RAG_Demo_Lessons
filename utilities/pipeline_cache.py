"""
utilities/pipeline_cache.py — Persist the built pipeline to disk
==================================================================
Re-running embeddings on every Streamlit restart is wasteful: encoding
hundreds of chunks with `all-mpnet-base-v2` takes 30-60 seconds, and the
result is fully deterministic for a given (documents, embedding_model)
pair. Caching it on disk turns a 60s startup into a 2s one.

WHAT GETS PERSISTED
-------------------
The FAISS index already serialises itself via `vector_store.save_local()`,
which writes two files:
    assets/faiss_index/index.faiss   — the raw FAISS index
    assets/faiss_index/index.pkl     — chunks + metadata (LangChain pickle)

The FAISS pickle CONTAINS the chunks already, so we don't need a
separate file for them. We DO save:
    assets/embeddings.npy            — raw (n, dim) matrix for analytics
    assets/cache_meta.json           — fingerprint to validate cache freshness

WHAT'S A "FINGERPRINT"?
-----------------------
The cache is only valid if it was built with:
  • the same embedding model      (different model → vectors incompatible)
  • the same chunking parameters  (chunk_size, overlap, min_chunk_length)
  • the same document set         (we hash filename + size + mtime)

If ANY of those changes, the cache is stale and we silently rebuild.
This is the same idea as a Makefile timestamp dependency or a Docker
layer hash — it just guarantees correctness without bothering the user.

WHAT'S NOT CACHED
-----------------
The LLM weights are NOT cached by us — HuggingFace already caches them
in ~/.cache/huggingface, so a second `load_llm()` call is essentially
free (just file I/O, no download). The SentenceTransformer model is
similarly cached by sentence-transformers.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import asdict

import numpy as np

from backend.config import (
    RAGConfig, ASSETS_DIR, FAISS_INDEX_PATH, DATA_DIR,
)


# ── File names inside ASSETS_DIR ──
_EMBEDDINGS_FILE = ASSETS_DIR / "embeddings.npy"
_META_FILE       = ASSETS_DIR / "cache_meta.json"


# ════════════════════════════════════════════════════════════
# 1. FINGERPRINT — what makes a cache valid?
# ════════════════════════════════════════════════════════════

def _hash_data_dir(directory: Path = DATA_DIR) -> str:
    """
    Build a short hash representing the CURRENT state of data/.

    We don't hash file CONTENTS (slow on big PDFs) — instead we hash
    a tuple of (filename, size_bytes, mtime) for every file. This
    catches add / delete / replace, which is everything we care about.
    """
    if not directory.exists():
        return "empty"

    entries = []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in {".txt", ".pdf", ".jpg", ".jpeg", ".png",
                       ".tiff", ".tif", ".bmp"}:
            continue
        stat = f.stat()
        entries.append((f.name, stat.st_size, int(stat.st_mtime)))

    # Sorted tuple → deterministic hash
    blob = repr(sorted(entries)).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def _config_fingerprint(config: RAGConfig) -> dict:
    """
    Return the subset of config fields that affect the cached vectors.

    Other fields (k, temperature, system_prompt, llm_model_name) only
    affect retrieval / generation at query time — they DON'T require
    re-embedding, so we don't include them in the fingerprint.
    """
    return {
        "embedding_model_name": config.embedding_model_name,
        "chunk_size":           config.chunk_size,
        "chunk_overlap":        config.chunk_overlap,
        "min_chunk_length":     config.min_chunk_length,
    }


# ════════════════════════════════════════════════════════════
# 2. SAVE — persist after a successful build
# ════════════════════════════════════════════════════════════

def save_cache(
    embeddings: np.ndarray,
    config: RAGConfig,
    n_chunks: int,
    n_docs: int,
) -> None:
    """
    Persist the embeddings matrix + a metadata fingerprint.

    The FAISS index itself is saved separately by
    `backend.vector_store.build_vector_store(save=True)`.

    Args:
        embeddings : the (n_chunks, embedding_dim) numpy matrix
        config     : the RAGConfig used to build the embeddings
        n_chunks   : for display in the UI
        n_docs     : for display in the UI
    """
    # 1. Save the embeddings matrix in numpy's native binary format.
    #    np.save preserves shape and dtype, no precision loss.
    np.save(_EMBEDDINGS_FILE, embeddings)

    # 2. Save the fingerprint as human-readable JSON.
    meta = {
        "fingerprint": _config_fingerprint(config),
        "data_hash":   _hash_data_dir(),
        "n_chunks":    int(n_chunks),
        "n_docs":      int(n_docs),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    _META_FILE.write_text(json.dumps(meta, indent=2))


# ════════════════════════════════════════════════════════════
# 3. LOAD — try to restore on startup
# ════════════════════════════════════════════════════════════

def load_cache(
    config: RAGConfig,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Attempt to load a previously-saved cache.

    Returns:
        (embeddings, meta_dict) if the cache exists AND is valid for
        the given config. None if there's nothing to load or the cache
        is stale.
    """
    # 1. Files exist?
    if not (_META_FILE.exists() and _EMBEDDINGS_FILE.exists()):
        return None

    # 2. The FAISS index files must also exist (otherwise we can't
    #    reconstruct the searchable store).
    if not (FAISS_INDEX_PATH / "index.faiss").exists():
        return None

    # 3. Read and validate metadata.
    try:
        meta = json.loads(_META_FILE.read_text())
    except Exception:
        return None

    # 4. Fingerprint must match. If the user changed the embedding model
    #    or chunking parameters, the cached vectors are useless.
    if meta.get("fingerprint") != _config_fingerprint(config):
        return None

    # 5. Documents on disk must match what was indexed. If the user added
    #    or removed documents since the last build, the cache is stale.
    if meta.get("data_hash") != _hash_data_dir():
        return None

    # 6. Load the embeddings matrix.
    try:
        embeddings = np.load(_EMBEDDINGS_FILE)
    except Exception:
        return None

    return embeddings, meta


# ════════════════════════════════════════════════════════════
# 4. CLEAR — invalidate cache on demand
# ════════════════════════════════════════════════════════════

def clear_cache() -> None:
    """
    Remove every cached file. Useful when the user wants a clean rebuild.
    """
    for p in (_EMBEDDINGS_FILE, _META_FILE):
        if p.exists():
            p.unlink()
    # Also nuke the FAISS index folder so we don't leave a partial state.
    if FAISS_INDEX_PATH.exists():
        for f in FAISS_INDEX_PATH.iterdir():
            if f.is_file():
                f.unlink()


# ════════════════════════════════════════════════════════════
# 5. PEEK — read metadata without loading the embeddings
# ════════════════════════════════════════════════════════════

def cache_info(config: Optional[RAGConfig] = None) -> Optional[dict]:
    """
    Return cache metadata if a cache exists, plus a `valid` flag
    indicating whether it matches the current config.

    Used by the sidebar to show a status indicator without paying the
    cost of loading the actual embeddings.
    """
    if not _META_FILE.exists():
        return None
    try:
        meta = json.loads(_META_FILE.read_text())
    except Exception:
        return None

    if config is not None:
        meta["valid_for_current_config"] = (
            meta.get("fingerprint") == _config_fingerprint(config)
            and meta.get("data_hash") == _hash_data_dir()
        )
    return meta
