"""
frontend/analytics_ui.py — Embedding Explorer
===============================================
Interactive 2D / 3D visualizations of the embedding space.

Features:
  - Toggle between 2D and 3D scatter plots
  - Choose PCA or t-SNE projection
  - Optional query — embed it live and see where it lands
  - Retrieved chunks are highlighted (red star in 2D, red diamond in 3D)
  - Below the plot: full content of each retrieved chunk + similarity bar
"""

import os
import streamlit as st
import numpy as np

from utilities.analytics import (
    plot_embedding_space,
    plot_embedding_space_3d,
    plot_similarity_histogram,
    detect_topic,
    TOPIC_COLORS,
)
from backend.embedder import compute_similarity


def render_analytics():
    """Render the embedding-space explorer."""

    st.markdown("### 🗺️ Embedding Space Explorer")

    chunks = st.session_state.get("chunks")
    embeddings = st.session_state.get("embeddings_np")
    st_model = st.session_state.get("st_model")

    if chunks is None or embeddings is None:
        st.info("📊 Build the pipeline first to explore the embedding space.")
        return

    # ─── Plot controls ──────────────────────────────────
    col_dim, col_method, col_query = st.columns([1, 1, 3])
    with col_dim:
        view = st.radio("View", ["3D", "2D"], index=0, horizontal=True)
    with col_method:
        method = st.radio("Projection", ["PCA", "t-SNE"], index=0, horizontal=True)
    with col_query:
        query = st.text_input(
            "🔍 Enter a query to highlight retrieved chunks:",
            placeholder="e.g., What is deep learning?",
        )

    # ─── Compute top-K if query provided ────────────────
    top_indices = None
    scores = None
    if query and st_model:
        q_emb = st_model.encode(query, normalize_embeddings=True)
        scores = compute_similarity(q_emb, embeddings)
        config = st.session_state.get("config", None)
        k_val = config.k if config else 5
        top_indices = np.argsort(scores)[-k_val:][::-1].tolist()

    # ─── Render plot ────────────────────────────────────
    if view == "3D":
        fig = plot_embedding_space_3d(
            chunks=chunks,
            embeddings=embeddings,
            method=method.lower(),
            query=query if query else None,
            top_indices=top_indices,
        )
    else:
        fig = plot_embedding_space(
            chunks=chunks,
            embeddings=embeddings,
            method=method.lower(),
            query=query if query else None,
            top_indices=top_indices,
        )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Retrieved chunks: full content ─────────────────
    if top_indices is not None and scores is not None:
        st.markdown("### 📦 Retrieved chunks")
        st.caption(
            "These are the top-K chunks closest to the query in the 768-dim space. "
            "The numbers correspond to the red markers in the plot above."
        )

        for rank, idx in enumerate(top_indices, start=1):
            chunk = chunks[idx]
            score = float(scores[idx])
            source = os.path.basename(chunk.metadata.get("source", "?"))
            topic = detect_topic(chunk.page_content)
            color = TOPIC_COLORS.get(topic, "#90A4AE")

            # Header with rank, topic chip, source, score
            bar_pct = max(0, min(100, score * 100))

            with st.expander(
                f"#{rank} — 📄 {source} · 🎯 similarity {score:.3f}",
                expanded=(rank == 1),
            ):
                # Topic + similarity bar + length info
                meta_html = f"""
                <div style="display:flex; gap:1rem; flex-wrap:wrap; align-items:center; margin-bottom:0.6rem;">
                    <span style="background:{color}22; color:{color}; padding:3px 10px;
                                 border-radius:100px; font-size:0.8rem; font-weight:500">
                        {topic}
                    </span>
                    <span style="color:#666; font-size:0.85rem">
                        chunk_id={idx} · {len(chunk.page_content)} chars · {len(chunk.page_content.split())} words
                    </span>
                </div>
                <div style="background:#eee; border-radius:6px; height:8px; overflow:hidden; margin-bottom:0.8rem;">
                    <div style="background:linear-gradient(90deg, #e53935, #fdd835, #43a047);
                                width:{bar_pct:.1f}%; height:100%;"></div>
                </div>
                """
                st.markdown(meta_html, unsafe_allow_html=True)
                # Full chunk content
                st.markdown(
                    f"<div style='background:#f8f9fa; padding:1rem; border-radius:8px; "
                    f"border-left:4px solid {color}; font-size:0.9rem; line-height:1.5;'>"
                    f"{chunk.page_content}</div>",
                    unsafe_allow_html=True,
                )

    # ─── Similarity histogram ───────────────────────────
    if query and st_model and scores is not None:
        st.markdown("### 📊 Similarity Distribution")
        config = st.session_state.get("config")
        k_val = config.k if config else 5
        hist_fig = plot_similarity_histogram(scores, query, k=k_val)
        st.plotly_chart(hist_fig, use_container_width=True)

        st.markdown(f"""
        **Stats for query:** "{query}"
        - Chunks with similarity > 0.5: **{(scores > 0.5).sum()}** (highly relevant)
        - Chunks with similarity > 0.3: **{(scores > 0.3).sum()}** (somewhat relevant)
        - Chunks with similarity < 0.2: **{(scores < 0.2).sum()}** (irrelevant)
        """)

    # ─── Topic distribution ─────────────────────────────
    st.markdown("### 📊 Topic Distribution")
    topics = [detect_topic(c.page_content) for c in chunks]
    topic_counts = {}
    for t in topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        pct = count / len(topics) * 100
        st.markdown(f"- **{topic}**: {count} chunks ({pct:.0f}%)")
