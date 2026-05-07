"""
backend/config.py — Centralized Configuration
================================================
This module is the single source of truth for every tuneable parameter
of the RAG system. Every other file imports `RAGConfig` from here, so
changing a default value here propagates through the entire app.

WHY a single config file?
  In real-world ML projects, hyperparameters quickly become "magic
  numbers" scattered across the codebase. Centralising them makes it:
    • obvious which knobs exist
    • trivial to expose them in a UI (see settings_ui.py)
    • easy for students to experiment by changing ONE place
"""

# ── Imports ────────────────────────────────────────────────
import os                                # standard lib, used for env access
from dataclasses import dataclass, field # @dataclass auto-generates __init__
from pathlib import Path                 # cleaner than os.path for paths


# ────────────────────────────────────────────────────────────
# 1. Paths — where the project keeps its data on disk
# ────────────────────────────────────────────────────────────
# Path(__file__) is the path of THIS file (config.py).
# .parent twice goes one level up (backend/) and another (project root).
PROJECT_ROOT = Path(__file__).parent.parent

# Where Wikipedia .txt files and uploaded PDFs live.
DATA_DIR = PROJECT_ROOT / "data"

# Where caches go: FAISS index, model downloads, etc.
ASSETS_DIR = PROJECT_ROOT / "assets"

# Where docs (architecture HTML) live.
DOCS_DIR = PROJECT_ROOT / "docs"

# The exact subfolder for the FAISS index files.
FAISS_INDEX_PATH = ASSETS_DIR / "faiss_index"

# Create directories if they don't exist (idempotent — safe on every import).
DATA_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────
# 2. Default knowledge base — Wikipedia article titles
# ────────────────────────────────────────────────────────────
# These titles are passed to the Wikipedia REST API by data_loader.py.
# Using underscores instead of spaces matches Wikipedia's URL convention.
WIKI_TOPICS = [
    "Artificial_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Natural_language_processing",
    "Large_language_model",
    "Retrieval-augmented_generation",
    "Neural_network_(machine_learning)",
    "Computer_vision",
    "Transformer_(deep_learning_architecture)",
    "Python_(programming_language)",
]


# ────────────────────────────────────────────────────────────
# 3. Embedding model registry
# ────────────────────────────────────────────────────────────
# Embedding models convert TEXT into NUMERICAL VECTORS. Two pieces of
# meaning that are similar (e.g. "dog" / "puppy") end up close in vector
# space. This is what makes semantic search possible.
#
# Trade-offs:
#   - Smaller models (384 dim) are faster and use less RAM but capture
#     fewer nuances.
#   - Larger models (768 dim) capture more meaning but use more memory.
#   - Multilingual variants are needed only if your documents aren't
#     purely English.
#
# All models in this dict run LOCALLY via sentence-transformers / HuggingFace,
# so there are no API keys, no costs, and no data leaves the user's machine.
EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "label": "MiniLM-L6 — Fast & Small",
        "dim": 384,
        "size": "80 MB",
        "notes": "Best for quick experiments and limited RAM.",
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "label": "MPNet-base — Balanced (default)",
        "dim": 768,
        "size": "420 MB",
        "notes": "Best general-purpose English embeddings.",
    },
    "BAAI/bge-small-en-v1.5": {
        "label": "BGE-small — High quality, small",
        "dim": 384,
        "size": "130 MB",
        "notes": "BAAI's compact model with great retrieval scores.",
    },
    "BAAI/bge-base-en-v1.5": {
        "label": "BGE-base — Top retrieval quality",
        "dim": 768,
        "size": "440 MB",
        "notes": "Top of the MTEB leaderboard for its size.",
    },
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {
        "label": "Multilingual MPNet — 50+ languages",
        "dim": 768,
        "size": "1.1 GB",
        "notes": "Use this for non-English documents.",
    },
}


# ────────────────────────────────────────────────────────────
# 4. LLM (large language model) registry
# ────────────────────────────────────────────────────────────
# These are CHAT-tuned models — they're trained to follow a "system /
# user / assistant" pattern. We MUST format prompts using their chat
# template (see rag_chain.build_prompt) or they will misbehave.
#
# All listed models are NOT GATED on HuggingFace, so they download
# without needing a token.
#
# Trade-offs:
#   - 0.5–1.5B params: fast, runs on CPU, weaker reasoning
#   - 3–4B params: needs ~10 GB RAM, much better at synthesis
LLM_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "label": "Qwen 0.5B — Fastest",
        "params": "0.5B",
        "size": "~1 GB",
        "notes": "Tiny model. Great for low-RAM machines, less coherent.",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "label": "Qwen 1.5B — Balanced (default)",
        "params": "1.5B",
        "size": "~3 GB",
        "notes": "Good speed/quality tradeoff. Recommended starting point.",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "label": "Qwen 3B — Higher quality",
        "params": "3B",
        "size": "~6 GB",
        "notes": "Noticeably better answers, needs ~10 GB RAM.",
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "label": "SmolLM2 1.7B — Compact alternative",
        "params": "1.7B",
        "size": "~3.4 GB",
        "notes": "Recent model from HuggingFace, similar to Qwen 1.5B.",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "label": "Phi-3 mini — Strong reasoning",
        "params": "3.8B",
        "size": "~7 GB",
        "notes": "Microsoft's compact model. Excellent for technical Q&A.",
    },
}


# ────────────────────────────────────────────────────────────
# 5. The configuration dataclass
# ────────────────────────────────────────────────────────────
# Using @dataclass means Python auto-generates __init__, __repr__ and
# __eq__ for us, and every field is type-annotated.
#
# Students: change any default value and the whole pipeline adapts.
@dataclass
class RAGConfig:
    """
    All tuneable parameters for the RAG pipeline, in one place.

    CHUNKING — controls how documents are split before embedding.
        chunk_size       : larger = more context per chunk, but noisier
        chunk_overlap    : prevents info loss at chunk boundaries
        min_chunk_length : drops tiny chunks (headers, footers, fragments)

    EMBEDDING — converts text into vectors.
        embedding_model_name : any model from EMBEDDING_MODELS

    VECTOR STORE — FAISS index settings.
        faiss_metric : "cosine" (similarity) or "l2" (distance)

    RETRIEVAL — how many chunks to feed the LLM.
        k : top-K chunks. More = more context, but also more noise + tokens.

    GENERATION — controls the LLM.
        llm_model_name      : any model from LLM_MODELS
        temperature         : 0.1=factual, 1.0=creative, 1.5=chaotic
        max_new_tokens      : max length of the answer in tokens
        top_p               : nucleus sampling (0.9 = consider top 90% likely)
        repetition_penalty  : >1.0 discourages repeating words
    """
    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_length: int = 30  # min words per chunk to keep it

    # --- Embedding (default: balanced general-purpose English) ---
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # --- Vector store ---
    faiss_metric: str = "cosine"

    # --- Retrieval ---
    k: int = 5

    # --- LLM (default: 1.5B params, ~3 GB) ---
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.3
    max_new_tokens: int = 300
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # --- The "instructions" given to the LLM before every question ---
    # This becomes the SYSTEM message in the chat template.
    # Small LLMs (1.5B params) need very explicit instructions or they
    # tend to echo the context back. The phrasing below is deliberately
    # imperative and short.
    system_prompt: str = (
        "You are a helpful AI assistant. "
        "Read the context below and answer the user's question in your own words. "
        "Do NOT repeat or quote the context verbatim. "
        "Write 2-4 sentences explaining the answer naturally, like a teacher "
        "would explain it to a student. "
        "If the context does not contain the answer, say so briefly."
    )


# A ready-to-use default instance. Most modules accept `config: RAGConfig = DEFAULT_CONFIG`.
DEFAULT_CONFIG = RAGConfig()
