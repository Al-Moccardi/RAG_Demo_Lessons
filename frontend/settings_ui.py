"""
frontend/settings_ui.py — Pipeline Settings Sidebar
=====================================================
All tuneable RAG parameters live in the sidebar:
  - Model choice (embedding + LLM)
  - Chunking parameters
  - Retrieval k
  - Generation hyperparameters
  - Custom system prompt

Returns a fresh RAGConfig on every Streamlit rerun.
"""

import streamlit as st
from backend.config import RAGConfig, EMBEDDING_MODELS, LLM_MODELS, DEFAULT_CONFIG


def render_settings() -> RAGConfig:
    """Render the settings sidebar and return the current config."""

    st.sidebar.markdown("## ⚙️ Pipeline Settings")

    # ─── Model selection ────────────────────────────────
    st.sidebar.markdown("### 🤖 Models")

    # Embedding model
    emb_keys = list(EMBEDDING_MODELS.keys())
    emb_default_idx = emb_keys.index(DEFAULT_CONFIG.embedding_model_name) \
        if DEFAULT_CONFIG.embedding_model_name in emb_keys else 1
    embedding_model_name = st.sidebar.selectbox(
        "Embedding model",
        options=emb_keys,
        index=emb_default_idx,
        format_func=lambda k: EMBEDDING_MODELS[k]["label"],
        help="Converts text into numerical vectors. "
             "Smaller models = faster but slightly less accurate.",
    )
    emb_meta = EMBEDDING_MODELS[embedding_model_name]
    st.sidebar.caption(
        f"📐 {emb_meta['dim']}-dim · {emb_meta['size']} · {emb_meta['notes']}"
    )

    # LLM model
    llm_keys = list(LLM_MODELS.keys())
    llm_default_idx = llm_keys.index(DEFAULT_CONFIG.llm_model_name) \
        if DEFAULT_CONFIG.llm_model_name in llm_keys else 1
    llm_model_name = st.sidebar.selectbox(
        "LLM model",
        options=llm_keys,
        index=llm_default_idx,
        format_func=lambda k: LLM_MODELS[k]["label"],
        help="Generates the final answer from retrieved context. "
             "Larger = better quality but slower and more RAM.",
    )
    llm_meta = LLM_MODELS[llm_model_name]
    st.sidebar.caption(
        f"🧠 {llm_meta['params']} params · {llm_meta['size']} · {llm_meta['notes']}"
    )

    st.sidebar.info(
        "ℹ️ Changing models requires clicking **🔨 Build Pipeline** again."
    )

    # ─── Chunking ───────────────────────────────────────
    st.sidebar.markdown("### ✂️ Chunking")
    chunk_size = st.sidebar.slider(
        "Chunk size (chars)", 200, 2000, 500, 50,
        help="Larger = more context per chunk",
    )
    chunk_overlap = st.sidebar.slider(
        "Overlap (chars)", 0, 400, 50, 25,
        help="Prevents info loss at boundaries",
    )

    # ─── Retrieval ──────────────────────────────────────
    st.sidebar.markdown("### 🔍 Retrieval")
    k = st.sidebar.slider(
        "k (chunks to retrieve)", 1, 15, 5, 1,
        help="More chunks = more context but slower",
    )

    # ─── Generation ─────────────────────────────────────
    st.sidebar.markdown("### 🧠 Generation")
    temperature = st.sidebar.slider(
        "Temperature", 0.1, 1.5, 0.3, 0.1,
        help="0.1 = factual, 1.5 = creative",
    )
    max_tokens = st.sidebar.slider(
        "Max tokens", 64, 512, 300, 32,
        help="Answer length limit",
    )

    # ─── System Prompt ──────────────────────────────────
    st.sidebar.markdown("### 📝 System Prompt")
    system_prompt = st.sidebar.text_area(
        "Prompt",
        value=DEFAULT_CONFIG.system_prompt,
        height=120,
        help="The instruction given to the LLM before each question.",
    )

    return RAGConfig(
        embedding_model_name=embedding_model_name,
        llm_model_name=llm_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
        temperature=temperature,
        max_new_tokens=max_tokens,
        system_prompt=system_prompt,
    )
