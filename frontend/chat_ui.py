"""
frontend/chat_ui.py — Chat Interface
======================================
"""

import os
import streamlit as st
import numpy as np
from backend.rag_chain import RAGResponse


def render_chat():
    """Render the chat interface."""

    st.markdown("### 💬 Chat with your documents")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources_html" in msg:
                with st.expander(f"📚 {msg.get('n_sources', 0)} sources retrieved in {msg.get('time', '?')}s"):
                    st.markdown(msg["sources_html"], unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about AI, machine learning, transformers, RAG..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        pipeline = st.session_state.get("pipeline")
        if pipeline is None:
            with st.chat_message("assistant"):
                st.warning("⚠️ Pipeline not ready. Click **Build Pipeline** in the sidebar first.")
            return

        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving & generating..."):
                response: RAGResponse = pipeline.ask(prompt)

            # Display answer
            st.markdown(response.answer)

            # Build sources display
            sources_html = _build_sources_html(response)
            with st.expander(f"📚 {response.n_chunks} sources retrieved in {response.elapsed:.1f}s"):
                st.markdown(sources_html, unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources_html": sources_html,
            "n_sources": response.n_chunks,
            "time": f"{response.elapsed:.1f}",
        })

    # Clear button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            pipeline = st.session_state.get("pipeline")
            if pipeline:
                pipeline.clear_history()
            st.rerun()


def _build_sources_html(response: RAGResponse) -> str:
    """Build HTML for the sources expander."""
    html = f"**Query:** {response.query}\n\n"

    for i, (doc, score) in enumerate(response.sources):
        source = os.path.basename(doc.metadata.get("source", "?"))
        # FAISS returns L2 distance; convert to a 0-1 similarity-like scale
        sim = max(0, 1 - score / 2)  # approximate
        bar_len = int(sim * 10)
        bar = "█" * bar_len + "░" * (10 - bar_len)

        preview = doc.page_content[:250].replace("\n", " ")

        html += f"""<div class="source-card">
<b>📦 Chunk {i+1}</b> &nbsp;|&nbsp; 📄 <code>{source}</code> &nbsp;|&nbsp; 🎯 <code>{sim:.3f}</code> {bar}<br>
<small>{preview}...</small>
</div>\n"""

    return html
