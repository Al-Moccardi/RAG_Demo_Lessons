"""
frontend/landing.py — Home Page
==================================
Renders an interactive architecture overview by embedding the
docs/architecture.html file directly into the Streamlit page using
streamlit.components.v1.html.

The HTML contains every backend / frontend module as a clickable card
that opens a detail panel showing inputs, outputs, key functions and
the actual source code of that module.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from backend.config import DOCS_DIR


# Read the architecture HTML once at import time
_ARCH_HTML_PATH = DOCS_DIR / "architecture.html"
_ARCH_HTML = _ARCH_HTML_PATH.read_text(encoding="utf-8") \
    if _ARCH_HTML_PATH.exists() else ""


def render_landing():
    """Render the home page with the embedded interactive architecture."""

    # ─── Quick stats from the current pipeline (if built) ───
    stats = st.session_state.get("stats", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Documents", stats.get("n_docs", 0))
    with c2:
        st.metric("Chunks", stats.get("n_chunks", 0))
    with c3:
        config = st.session_state.get("config")
        emb = config.embedding_model_name.split("/")[-1] if config else "—"
        st.metric("Embedding", emb)
    with c4:
        llm = config.llm_model_name.split("/")[-1] if config else "—"
        st.metric("LLM", llm)

    st.markdown("")  # spacer

    # ─── Embedded interactive architecture ──────────────
    if _ARCH_HTML:
        # Tall iframe so users can scroll through the full diagram + details
        components.html(_ARCH_HTML, height=2400, scrolling=True)
    else:
        st.warning(
            "📄 Architecture diagram not found at `docs/architecture.html`. "
            "Make sure the docs/ folder ships with the project."
        )
