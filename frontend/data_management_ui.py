"""
frontend/data_management_ui.py — Knowledge Base Tab
======================================================
Lets the user manage the documents indexed by the RAG system:

  • Drag-and-drop PDFs / TXT files to add them
  • See a list of every file currently in data/
  • Delete individual files or wipe everything
  • Quick visual reminder that changes require a pipeline rebuild
"""

import streamlit as st
from pathlib import Path

from backend.data_loader import (
    save_uploaded_file,
    list_documents,
    delete_document,
    clear_all_documents,
    ALLOWED_EXTENSIONS,
)


def render_data_management():
    """Render the Knowledge Base tab."""

    st.markdown("### 📚 Knowledge Base")
    st.markdown(
        "Add or remove documents from your RAG knowledge base. "
        "After any change, click **🔨 Build Pipeline** in the sidebar to re-index."
    )

    # ─── Upload section ─────────────────────────────────
    st.markdown("#### ⬆️ Upload documents")
    st.caption(
        "PDFs and TXTs are loaded directly. Images and scanned PDFs are "
        "automatically routed through **OCR (Tesseract)** to extract text — "
        "see the 🎮 Playground tab for a step-by-step OCR walkthrough."
    )
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "txt", "jpg", "jpeg", "png", "tiff", "bmp"],
        accept_multiple_files=True,
        help=f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
    )

    if uploaded_files:
        col_save, col_info = st.columns([1, 4])
        with col_save:
            if st.button("💾 Save uploaded files", type="primary", use_container_width=True):
                saved, skipped = [], []
                for uf in uploaded_files:
                    path = save_uploaded_file(uf)
                    if path:
                        saved.append(path.name)
                    else:
                        skipped.append(uf.name)
                if saved:
                    st.success(f"✅ Saved: {', '.join(saved)}")
                if skipped:
                    st.warning(f"⚠️ Skipped (unsupported): {', '.join(skipped)}")
                st.session_state["pipeline_stale"] = True
                st.rerun()
        with col_info:
            st.caption(
                f"{len(uploaded_files)} file(s) ready. "
                "Click **Save** to add them to data/, then rebuild the pipeline."
            )

    st.markdown("---")

    # ─── Document list ──────────────────────────────────
    st.markdown("#### 📂 Documents currently indexed")
    docs = list_documents()

    if not docs:
        st.info("No documents yet. Upload PDFs/TXTs above, or click "
                "**📥 Download AI Articles** in the sidebar.")
        return

    # Header row
    header_cols = st.columns([0.5, 4, 1.5, 1, 1])
    header_cols[0].markdown("**#**")
    header_cols[1].markdown("**File**")
    header_cols[2].markdown("**Source**")
    header_cols[3].markdown("**Size**")
    header_cols[4].markdown("**Action**")

    for i, doc in enumerate(docs, start=1):
        cols = st.columns([0.5, 4, 1.5, 1, 1])
        cols[0].markdown(f"{i}")
        # Pick an icon based on the extension
        ext = doc["ext"]
        if ext == ".pdf":
            ext_icon = "📄"
        elif ext == ".txt":
            ext_icon = "📝"
        else:
            ext_icon = "🖼️"  # any image format
        cols[1].markdown(f"{ext_icon} `{doc['name']}`")
        cols[2].markdown(doc["source"])
        cols[3].markdown(f"{doc['size_kb']:.1f} KB")
        if cols[4].button("🗑️", key=f"del_{doc['name']}", help=f"Delete {doc['name']}"):
            if delete_document(doc["name"]):
                st.success(f"Deleted {doc['name']}")
                st.session_state["pipeline_stale"] = True
                st.rerun()

    st.markdown("---")

    # ─── Bulk actions ───────────────────────────────────
    col_clear, col_status = st.columns([1, 4])
    with col_clear:
        if st.button("🧹 Delete ALL", help="Remove every file in data/"):
            n = clear_all_documents()
            st.success(f"Removed {n} file(s).")
            st.session_state["pipeline_stale"] = True
            st.rerun()

    with col_status:
        if st.session_state.get("pipeline_stale"):
            st.warning(
                "⚠️ The knowledge base has changed since the pipeline was built. "
                "Click **🔨 Build Pipeline** in the sidebar to re-index."
            )
        elif "pipeline" in st.session_state:
            st.success(f"✅ Pipeline is in sync with {len(docs)} document(s).")
