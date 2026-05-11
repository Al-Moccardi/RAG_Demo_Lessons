"""
app.py — RAG Studio Main Application
======================================
Entry point for the Streamlit app.

Run with:
    streamlit run app.py

This file:
  1. Sets up the page layout
  2. Loads the sidebar settings (model selection + hyperparameters)
  3. AUTO-LOADS a previously-cached pipeline if one exists
  4. Manages the pipeline lifecycle
       Download → Load → Chunk → Embed → Index → Load LLM → Chain → Save cache
  5. Routes between eight pages:
       🏠 Home  ·  💬 Chat  ·  📊 Analytics  ·  📚 Knowledge Base  ·
       🎮 Playground  ·  🧪 Advanced  ·  📏 Evaluation  ·  🚀 GitHub Push

PERSISTENCE
-----------
The first time you click "Build Pipeline", everything is computed
from scratch and the result is saved to assets/. On the next restart
the app detects the cache and reloads it in ~2 seconds instead of
re-computing for ~60 seconds. See utilities/pipeline_cache.py for
details.
"""

import streamlit as st
import numpy as np

# ── Page Config (must be first Streamlit call) ──
st.set_page_config(
    page_title="RAG Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──
from frontend.styles import CUSTOM_CSS
from frontend.landing import render_landing
from frontend.chat_ui import render_chat
from frontend.settings_ui import render_settings
from frontend.analytics_ui import render_analytics
from frontend.data_management_ui import render_data_management
from frontend.playground_ui import render_playground
from frontend.advanced_ui import render_advanced
from frontend.evaluation_ui import render_evaluation
from frontend.github_push_ui import render_github_push

from backend.config import RAGConfig, DEFAULT_CONFIG, WIKI_TOPICS
from backend.data_loader import download_all_articles, load_documents
from backend.chunker import chunk_documents, get_chunk_stats
from backend.embedder import create_lc_embeddings, create_st_model, embed_texts
from backend.vector_store import build_vector_store, load_vector_store
from backend.llm import load_llm, update_generation_config
from backend.rag_chain import RAGPipeline

from utilities.pipeline_cache import (
    save_cache,
    load_cache,
    clear_cache,
    cache_info,
)

# ── Inject CSS ──
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# SIDEBAR: Settings + Pipeline Controls
# ═══════════════════════════════════════════════

# Settings (model selection + hyperparameters)
config = render_settings()
st.session_state["config"] = config

st.sidebar.markdown("---")
st.sidebar.markdown("## 🚀 Pipeline")


# ───────────────────────────────────────────────
# AUTO-LOAD: try to restore the cache on startup
# ───────────────────────────────────────────────
# This runs on every Streamlit rerun, but only does real work the FIRST
# time after a fresh launch (because once we've populated session_state,
# the `"pipeline" in st.session_state` check short-circuits).
def _try_autoload():
    """Restore a previously-built pipeline from disk if possible."""
    if "pipeline" in st.session_state:
        return  # already loaded in this session

    cache = load_cache(config)
    if cache is None:
        return  # no valid cache for current config

    embeddings_np, meta = cache

    with st.spinner("📂 Restoring cached pipeline from disk..."):
        # Re-create the embedding model (loads from local HF cache, fast).
        lc_embeddings = create_lc_embeddings(config)
        st_model = create_st_model(config)

        # Reload the FAISS index from assets/faiss_index/.
        vectorstore = load_vector_store(lc_embeddings, config=config)
        if vectorstore is None:
            return  # FAISS files missing despite metadata — give up

        # The chunks live INSIDE the FAISS pickle — extract them.
        # vectorstore.docstore._dict maps internal id → Document.
        try:
            chunks = list(vectorstore.docstore._dict.values())
        except Exception:
            chunks = []

        # Load the LLM (cached on disk by HuggingFace).
        llm = load_llm(config)

        # Reassemble the pipeline.
        pipeline = RAGPipeline(vectorstore, llm, config)

        # Populate session_state — same keys the Build button uses.
        st.session_state["pipeline"]      = pipeline
        st.session_state["vectorstore"]   = vectorstore
        st.session_state["llm"]           = llm
        st.session_state["chunks"]        = chunks
        st.session_state["embeddings_np"] = embeddings_np
        st.session_state["st_model"]      = st_model
        st.session_state["stats"] = {
            "n_docs":     meta.get("n_docs", "?"),
            "n_chunks":   meta.get("n_chunks", len(chunks)),
            "avg_tokens": 0,
        }
        st.session_state["pipeline_stale"]  = False
        st.session_state["loaded_from_cache"] = True


_try_autoload()


# ───────────────────────────────────────────────
# Cache status display
# ───────────────────────────────────────────────
ci = cache_info(config)
if ci is not None:
    if ci.get("valid_for_current_config"):
        st.sidebar.caption(
            f"💾 Cache: {ci.get('n_chunks', '?')} chunks · "
            f"{ci.get('embedding_dim', '?')}d ✅"
        )
    else:
        st.sidebar.caption(
            "💾 Cache exists but is **stale** "
            "(model/chunking/docs changed). Rebuilding will refresh it."
        )


# ── Download Data Button ──
if st.sidebar.button("📥 Download AI Articles", use_container_width=True):
    with st.sidebar:
        progress = st.progress(0, text="Downloading...")
        paths = []

        def update_progress(current, total, topic):
            progress.progress(
                (current + 1) / total,
                text=f"Downloading {topic.replace('_', ' ')}..."
            )

        try:
            paths = download_all_articles(WIKI_TOPICS, progress_callback=update_progress)
        except Exception as e:
            st.error(f"Download failed: {e}")
            paths = []

        if paths:
            progress.progress(1.0, text=f"✅ {len(paths)} articles downloaded!")
            st.session_state["data_downloaded"] = True
            st.session_state["pipeline_stale"] = True
        else:
            progress.progress(1.0, text="⚠️ No articles downloaded. Check your internet connection.")


# ── Build Pipeline Button ──
def _run_full_build():
    """Run the complete pipeline build and persist the cache."""
    status = st.status("Building RAG pipeline...", expanded=True)

    # 1. Load documents
    status.update(label="📄 Loading documents...")
    docs = load_documents()
    if not docs:
        status.update(
            label="❌ No documents found. Download articles or upload PDFs first!",
            state="error",
        )
        st.stop()

    # 2. Chunk
    status.update(label=f"✂️ Splitting {len(docs)} documents into chunks...")
    chunks = chunk_documents(docs, config)
    stats = get_chunk_stats(chunks)

    # 3. Embed
    status.update(label=f"🧮 Embedding {stats['count']} chunks with {config.embedding_model_name}...")
    lc_embeddings = create_lc_embeddings(config)
    st_model = create_st_model(config)

    texts = [c.page_content for c in chunks]
    embeddings_np = embed_texts(texts, st_model, config)

    # 4. Build FAISS index (this also auto-saves to assets/faiss_index/)
    status.update(label="🗄️ Building FAISS index...")
    vectorstore = build_vector_store(chunks, lc_embeddings, config)

    # 5. Load LLM
    status.update(label=f"🧠 Loading {config.llm_model_name} (first time may take minutes)...")
    llm = load_llm(config)

    # 6. Assemble pipeline
    status.update(label="🔗 Assembling RAG chain...")
    pipeline = RAGPipeline(vectorstore, llm, config)

    # 7. Persist the cache for next startup
    status.update(label="💾 Saving cache to disk...")
    try:
        save_cache(
            embeddings=embeddings_np,
            config=config,
            n_chunks=stats["count"],
            n_docs=len(docs),
        )
    except Exception as e:
        # Don't fail the whole build over a cache save error.
        print(f"  ⚠️ Cache save failed: {e}")

    # 8. Save to session state
    st.session_state["pipeline"]      = pipeline
    st.session_state["vectorstore"]   = vectorstore
    st.session_state["llm"]           = llm
    st.session_state["chunks"]        = chunks
    st.session_state["embeddings_np"] = embeddings_np
    st.session_state["st_model"]      = st_model
    st.session_state["stats"] = {
        "n_docs":     len(docs),
        "n_chunks":   stats["count"],
        "avg_tokens": int(stats.get("avg_tokens", 0)),
    }
    st.session_state["pipeline_stale"]    = False
    st.session_state["loaded_from_cache"] = False

    status.update(
        label=f"✅ Pipeline ready! {stats['count']} chunks indexed and cached.",
        state="complete",
    )


if st.sidebar.button("🔨 Build Pipeline", type="primary", use_container_width=True):
    with st.sidebar:
        _run_full_build()


# ── Apply Settings Button (no rebuild) ──
if st.sidebar.button("🔄 Apply Settings", use_container_width=True,
                     help="Updates k, temperature, prompt etc. without re-indexing."):
    pipeline = st.session_state.get("pipeline")
    llm = st.session_state.get("llm")
    if pipeline and llm:
        pipeline.update_config(config)
        if hasattr(llm, 'pipeline'):
            update_generation_config(llm.pipeline, config)
        st.sidebar.success("✅ Settings applied!")
    else:
        st.sidebar.warning("Build the pipeline first.")


# ── Clear Cache Button ──
with st.sidebar.expander("🗑️ Cache management"):
    st.caption(
        "The pipeline (embeddings + FAISS index) is saved to `assets/` "
        "after each build. Subsequent restarts skip the embedding step."
    )
    if st.button("Clear cache (forces full rebuild next time)",
                 use_container_width=True):
        clear_cache()
        # Also drop from current session so the badge updates immediately.
        for k in ("pipeline", "vectorstore", "llm", "chunks",
                  "embeddings_np", "st_model", "stats", "loaded_from_cache"):
            st.session_state.pop(k, None)
        st.sidebar.success("✅ Cache cleared.")
        st.rerun()


# ── Pipeline status ──
st.sidebar.markdown("---")
if "pipeline" in st.session_state and not st.session_state.get("pipeline_stale"):
    stats = st.session_state.get("stats", {})
    src_label = "💾 from cache" if st.session_state.get("loaded_from_cache") else "🔨 freshly built"
    st.sidebar.success(
        f"✅ Pipeline active ({src_label})\n\n"
        f"📄 {stats.get('n_docs', '?')} docs · "
        f"📦 {stats.get('n_chunks', '?')} chunks"
    )
elif st.session_state.get("pipeline_stale"):
    st.sidebar.warning("⚠️ Knowledge base changed.\nRebuild the pipeline.")
else:
    st.sidebar.info(
        "Pipeline not built yet.\n\n"
        "1. Add documents (Wikipedia or upload)\n"
        "2. Click **Build Pipeline**"
    )


# ═══════════════════════════════════════════════
# MAIN CONTENT: Page Router (5 tabs)
# ═══════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Home",
    "💬 Chat",
    "📊 Analytics",
    "📚 Knowledge Base",
    "🎮 Playground",
    "🧪 Advanced",
    "📏 Evaluation",
    "🚀 GitHub Push",
])

with tab1:
    render_landing()

with tab2:
    render_chat()

with tab3:
    render_analytics()

with tab4:
    render_data_management()

with tab5:
    render_playground()

with tab6:
    render_advanced()

with tab7:
    render_evaluation()

with tab8:
    render_github_push()
