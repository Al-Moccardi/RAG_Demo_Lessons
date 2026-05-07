"""
frontend/playground_ui.py — Interactive Pipeline Playground
=============================================================
Lets students explore each stage of the RAG pipeline in isolation.

Five sub-stages, each calling the *actual* backend functions:

  1. ✂️  Chunking         — paste text, see RecursiveCharacterTextSplitter at work
  2. 🧮  Embedding        — encode 1-2 sentences, see the vector + similarity
  3. 🔍  Retrieval        — query FAISS, see ranked chunks with scores
  4. 📝  Prompt Builder   — see the EXACT prompt sent to the LLM
  5. 🚀  Full Trace       — end-to-end ask() with per-stage timings

Stages 3-5 require a built pipeline; 1 and 2 work with anything loaded.
"""

import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from langchain_core.documents import Document

from backend.config import RAGConfig, DEFAULT_CONFIG
from backend.chunker import chunk_documents, get_chunk_stats
from backend.embedder import compute_similarity


CHUNK_COLORS = ["#5dcaa5", "#afa9ec", "#f0997b", "#85b7eb",
                "#ed93b1", "#efaf27", "#97c459", "#85ebd9"]


def render_playground():
    """Top-level router for the Playground tab."""

    st.markdown("### 🎮 RAG Playground")
    st.caption(
        "Experiment with each stage of the RAG pipeline in isolation. "
        "Stages call the same backend functions used by the real app."
    )

    stage = st.radio(
        "Pick a stage:",
        [
            "🔎  0. OCR (image → text)",
            "✂️  1. Chunking",
            "🧮  2. Embedding",
            "🔍  3. Retrieval",
            "📝  4. Prompt Builder",
            "🎯  5. Full Input View",
            "🚀  6. Full Trace",
            "💾  7. Persistence Demo",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if "OCR" in stage:
        _playground_ocr()
    elif "Chunking" in stage:
        _playground_chunking()
    elif "Embedding" in stage:
        _playground_embedding()
    elif "Retrieval" in stage:
        _playground_retrieval()
    elif "Prompt" in stage:
        _playground_prompt()
    elif "Full Input" in stage:
        _playground_full_input()
    elif "Full Trace" in stage:
        _playground_full_trace()
    elif "Persistence" in stage:
        _playground_persistence()


# ──────────────────────────────────────────────
# 0. OCR playground — visualize the image → text flow
# ──────────────────────────────────────────────

def _playground_ocr():
    """
    Step-by-step OCR demo:

      1. User uploads an image (or uses a sample)
      2. Show the ORIGINAL image
      3. Run preprocess_image() and show the PREPROCESSED image
      4. Call pytesseract.image_to_data() and show:
            - The recognized text
            - Mean confidence + per-word table
            - Bounding boxes drawn on top of the preprocessed image
    """
    import io
    import pandas as pd
    from PIL import Image, ImageDraw

    st.markdown("#### 🔎 OCR demo")
    st.caption(
        "**Library:** Tesseract 5 (via `pytesseract`). "
        "Walks through the three stages every OCR system performs: "
        "**load → preprocess → recognize**."
    )

    # ─── PRE-FLIGHT: is Tesseract installed? ───────────
    # `pytesseract` is just a Python wrapper around a real binary.
    # If the binary isn't on PATH, EVERY OCR call will throw the
    # same error. Catching it once here gives users a much friendlier
    # experience than letting them hit the error mid-flow.
    try:
        from utilities.ocr import (
            tesseract_available,
            preprocess_image,
        )
        import pytesseract
    except ImportError as e:
        st.error(
            f"❌ Python OCR libraries not installed.\n\n"
            f"Run: `pip install pytesseract pdf2image Pillow`\n\n"
            f"Detailed error: {e}"
        )
        return

    is_ok, msg = tesseract_available()
    if not is_ok:
        st.error("❌ **Tesseract is not installed**")
        # Show the multi-line install instructions in a code block so newlines render.
        st.code(msg, language="text")
        st.markdown(
            "---\n"
            "**Why this is needed:** Tesseract is the open-source OCR engine "
            "(originally Google → HP) that does the actual character recognition. "
            "`pytesseract` is just a Python wrapper that shells out to it — "
            "without the underlying binary, no recognition can happen."
        )
        return

    # Show a small green status when everything is healthy.
    st.success(f"✅ {msg}")

    # ─── Upload an image ───────────────────────────────
    uploaded = st.file_uploader(
        "Upload an image (JPG/PNG/TIFF). Tip: photos of book pages or "
        "screenshots of text work great.",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
    )

    # If the user hasn't uploaded yet, show a friendly placeholder
    if uploaded is None:
        st.info(
            "👆 Drop an image above to start. The walkthrough will:\n"
            "1. Show the original image\n"
            "2. Show the preprocessed (grayscale + 2×) version\n"
            "3. Run Tesseract and show the extracted text + per-word confidence"
        )
        return

    # ─── Settings: language + preprocessing toggle ─────
    col1, col2 = st.columns(2)
    with col1:
        lang = st.selectbox(
            "Tesseract language",
            options=["eng", "ita", "ita+eng", "fra", "deu", "spa"],
            index=0,
            help="`eng` = English, `ita` = Italian, "
                 "`ita+eng` = bilingual mode (slower). "
                 "Non-English language packs may need to be installed separately "
                 "(`apt install tesseract-ocr-ita`).",
        )
    with col2:
        do_preprocess = st.checkbox(
            "Apply preprocessing (grayscale + 2× upscale)",
            value=True,
            help="Tesseract works best on clean, high-DPI, B/W images.",
        )

    # ─── 1. Load original image ────────────────────────
    image_bytes = uploaded.read()
    original = Image.open(io.BytesIO(image_bytes))
    original.load()  # force loading so the file handle can close

    st.markdown("##### 1️⃣  Original image")
    st.caption(
        f"Size: {original.size[0]} × {original.size[1]} px · "
        f"Mode: `{original.mode}` · {len(image_bytes) / 1024:.1f} KB"
    )
    st.image(original, use_container_width=True)

    # ─── 2. Preprocess ─────────────────────────────────
    if do_preprocess:
        preprocessed = preprocess_image(original)
        st.markdown("##### 2️⃣  Preprocessed image")
        st.caption(
            f"Size: {preprocessed.size[0]} × {preprocessed.size[1]} px · "
            f"Mode: `{preprocessed.mode}` (grayscale, upscaled 2×)"
        )
        st.image(preprocessed, use_container_width=True)
    else:
        preprocessed = original
        st.markdown("##### 2️⃣  Preprocessing skipped")

    # ─── 3. Run Tesseract ──────────────────────────────
    st.markdown("##### 3️⃣  Tesseract OCR")
    with st.spinner("Running Tesseract..."):
        try:
            data = pytesseract.image_to_data(
                preprocessed,
                lang=lang,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError as e:
            # Most common cause: requested language pack is missing.
            err = str(e)
            if "language" in err.lower() or "tessdata" in err.lower():
                st.error(
                    f"❌ Tesseract language pack missing for `{lang}`.\n\n"
                    f"Install it:\n"
                    f"  • Ubuntu/Debian: `sudo apt install tesseract-ocr-{lang}`\n"
                    f"  • macOS (brew): comes bundled with `brew install tesseract-lang`\n\n"
                    f"Original error: {err}"
                )
            else:
                st.error(f"Tesseract error: {err}")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    # Build the per-word table
    rows = []
    for i in range(len(data["text"])):
        txt = data["text"][i]
        if not txt or not txt.strip():
            continue
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf < 0:
            continue
        rows.append({
            "word": txt,
            "conf": conf,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        })

    if not rows:
        st.warning(
            "Tesseract didn't recognize any words. "
            "Try a higher-resolution image, or untoggle preprocessing."
        )
        return

    full_text = " ".join(r["word"] for r in rows)
    mean_conf = sum(r["conf"] for r in rows) / len(rows)

    # ─── Top metrics ──────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words detected", len(rows))
    c2.metric("Mean confidence", f"{mean_conf:.1f}%")
    c3.metric("Total chars", len(full_text))
    c4.metric("Tesseract version", pytesseract.get_tesseract_version().__str__().split()[0])

    # ─── Recognized text ──────────────────────────────
    st.markdown("**Recognized text:**")
    st.success(full_text)

    # ─── Bounding-box visualization ────────────────────
    st.markdown("**Per-word bounding boxes** (green = high confidence, red = low):")
    overlay = preprocessed.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for r in rows:
        # Color: red below 60%, yellow 60-80%, green above 80%
        if r["conf"] >= 80:
            color = (67, 160, 71)
        elif r["conf"] >= 60:
            color = (253, 216, 53)
        else:
            color = (229, 57, 53)
        draw.rectangle(
            [(r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"])],
            outline=color,
            width=2,
        )
    st.image(overlay, use_container_width=True)

    # ─── Per-word confidence table ─────────────────────
    with st.expander("🔬 Per-word confidence table"):
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=300)

    # ─── How it would feed the RAG pipeline ────────────
    with st.expander("➡️ How this enters the RAG pipeline"):
        st.markdown(
            "If you saved this file in the **Knowledge Base** tab, "
            "`backend/data_loader.load_documents()` would call "
            "`utilities.ocr.ocr_image_file(path)` on it. The output above "
            "would become a LangChain `Document` with this metadata:"
        )
        st.json({
            "source": uploaded.name,
            "page": 1,
            "ocr": True,
            "ocr_conf": round(mean_conf, 2),
        })
        st.markdown(
            "From there it flows through the same chunker → embedder → "
            "FAISS path as any other document."
        )


# ──────────────────────────────────────────────
# 1. CHUNKING playground
# ──────────────────────────────────────────────

def _playground_chunking():
    st.markdown("#### ✂️ Chunking demo")
    st.caption(
        "**Function:** `chunker.chunk_documents(documents, config)` → "
        "`RecursiveCharacterTextSplitter` tries `\\n\\n` → `\\n` → `\". \"` → `\" \"` → `\"\"`."
    )

    DEFAULT_TEXT = (
        "Deep learning is a subset of machine learning that uses neural networks "
        "with many layers to model complex patterns in data.\n\n"
        "Convolutional neural networks (CNNs) excel at image tasks. "
        "Recurrent neural networks (RNNs) handle sequential data such as text and audio.\n\n"
        "Since 2017, the Transformer architecture has dominated natural language processing. "
        "Models like BERT and GPT are based on transformers and have revolutionized NLP. "
        "Modern large language models (LLMs) routinely have billions of parameters and "
        "are trained on hundreds of gigabytes of text. They can write code, answer questions, "
        "summarize documents and translate between dozens of languages."
    )

    text = st.text_area("Input text:", value=DEFAULT_TEXT, height=180)

    col1, col2, col3 = st.columns(3)
    with col1:
        chunk_size = st.slider("Chunk size", 50, 1000, 200, 25)
    with col2:
        chunk_overlap = st.slider("Overlap", 0, 200, 40, 10)
    with col3:
        min_words = st.slider("Min words / chunk", 1, 50, 5, 1)

    if not text.strip():
        st.warning("Type some text above.")
        return

    # Run the actual chunker
    config = RAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_words,
    )
    docs = [Document(page_content=text, metadata={"source": "playground"})]
    chunks = chunk_documents(docs, config)
    stats = get_chunk_stats(chunks)

    # ─── Stats ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", stats.get("count", 0))
    c2.metric("Avg chars", f"{stats.get('avg_chars', 0):.0f}")
    c3.metric("Avg words", f"{stats.get('avg_tokens', 0):.0f}")
    c4.metric("Input chars", len(text))

    # ─── Visualise each chunk in a coloured card ───────
    if not chunks:
        st.warning(
            "No chunks survived the `min_chunk_length` filter. "
            "Lower the slider or add more text."
        )
        return

    st.markdown("**Chunks produced:**")
    for i, chunk in enumerate(chunks):
        color = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        n_chars = len(chunk.page_content)
        n_words = len(chunk.page_content.split())
        st.markdown(
            f"""<div style="background:{color}1A; border-left:4px solid {color};
                          padding:0.7rem 1rem; border-radius:0 8px 8px 0;
                          margin:0.4rem 0;">
                <div style="font-size:0.78rem; color:{color}; font-weight:600; margin-bottom:0.4rem;">
                    Chunk #{i+1} · {n_chars} chars · {n_words} words
                </div>
                <div style="font-family: ui-monospace, Menlo, Consolas, monospace;
                            font-size:0.85rem; line-height:1.5; color:#222;
                            white-space:pre-wrap;">{chunk.page_content}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# 2. EMBEDDING playground
# ──────────────────────────────────────────────

def _playground_embedding():
    st.markdown("#### 🧮 Embedding demo")
    st.caption(
        "**Function:** `SentenceTransformer.encode(text, normalize_embeddings=True)`. "
        "Cosine similarity = dot product because embeddings are L2-normalized."
    )

    st_model = st.session_state.get("st_model")
    if st_model is None:
        st.info(
            "ℹ️ The embedding model is loaded only after **🔨 Build Pipeline**. "
            "Build the pipeline once, then come back here."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_input("Sentence A:",
                               "Deep learning uses neural networks with many layers.")
    with col2:
        text_b = st.text_input("Sentence B:",
                               "Machine learning trains algorithms on large datasets.")

    if not (text_a and text_b):
        st.warning("Enter both sentences.")
        return

    emb_a = st_model.encode(text_a, normalize_embeddings=True, convert_to_numpy=True)
    emb_b = st_model.encode(text_b, normalize_embeddings=True, convert_to_numpy=True)
    sim = float(np.dot(emb_a, emb_b))

    # ─── Big numbers ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cosine similarity", f"{sim:.4f}")
    c2.metric("Dim", emb_a.shape[0])
    c3.metric("‖A‖₂", f"{float(np.linalg.norm(emb_a)):.3f}")
    c4.metric("‖B‖₂", f"{float(np.linalg.norm(emb_b)):.3f}")

    # ─── Interpretation ────────────────────────────────
    if sim > 0.7:
        st.success(f"🟢 Very similar meaning ({sim:.2f}).")
    elif sim > 0.4:
        st.info(f"🟡 Somewhat related ({sim:.2f}).")
    else:
        st.warning(f"🔴 Largely unrelated ({sim:.2f}).")

    # ─── Bar chart of first 32 dims ─────────────────────
    st.markdown(f"**First 32 dimensions** (out of {emb_a.shape[0]}):")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(32)), y=emb_a[:32].tolist(),
                         name="Sentence A", marker_color="#5dcaa5"))
    fig.add_trace(go.Bar(x=list(range(32)), y=emb_b[:32].tolist(),
                         name="Sentence B", marker_color="#afa9ec"))
    fig.update_layout(
        barmode="group",
        xaxis_title="Dimension",
        yaxis_title="Value",
        height=320,
        margin=dict(t=10, b=40, l=40, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔬 Show full vector A (first 100 dims)"):
        st.code(np.array2string(emb_a[:100], precision=4, separator=", "), language="text")


# ──────────────────────────────────────────────
# 3. RETRIEVAL playground
# ──────────────────────────────────────────────

def _playground_retrieval():
    st.markdown("#### 🔍 Retrieval demo")
    st.caption(
        "**Function:** `vectorstore.similarity_search_with_score(query, k)`. "
        "Returns the K chunks closest to the query embedding (FAISS L2 distance)."
    )

    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    col_q, col_k = st.columns([4, 1])
    with col_q:
        query = st.text_input("Query:", "What is deep learning?")
    with col_k:
        k = st.slider("K", 1, 10, 5)

    if not query:
        return

    t0 = time.time()
    results = vectorstore.similarity_search_with_score(query, k=k)
    elapsed = (time.time() - t0) * 1000

    st.success(f"FAISS returned {len(results)} chunks in **{elapsed:.1f} ms**")

    for i, (doc, score) in enumerate(results, start=1):
        sim = max(0.0, 1.0 - score / 2.0)
        source = os.path.basename(doc.metadata.get("source", "?"))
        bar_pct = sim * 100

        st.markdown(
            f"""<div style="border:1px solid #e0e0e0; border-radius:10px;
                          padding:0.9rem 1rem; margin:0.5rem 0; background:#fafafa;">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                    <div style="background:#1976d2; color:white; padding:2px 10px;
                                border-radius:100px; font-size:0.78rem; font-weight:600;">
                        #{i}
                    </div>
                    <code style="font-size:0.82rem;">{source}</code>
                    <span style="margin-left:auto; font-family:monospace; font-size:0.82rem; color:#666;">
                        L2={score:.3f} · sim={sim:.3f}
                    </span>
                </div>
                <div style="background:#e0e0e0; border-radius:6px; height:6px; overflow:hidden; margin-bottom:0.6rem;">
                    <div style="background:linear-gradient(90deg, #e53935, #fdd835, #43a047);
                                width:{bar_pct:.1f}%; height:100%;"></div>
                </div>
                <div style="font-size:0.88rem; line-height:1.5; color:#222;">
                    {doc.page_content[:600]}{'...' if len(doc.page_content) > 600 else ''}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# 4. PROMPT BUILDER playground
# ──────────────────────────────────────────────

def _playground_prompt():
    st.markdown("#### 📝 Prompt construction")
    st.caption(
        "See the four logical sections (`system`, `context`, `history`, `question`) "
        "and **the actual chat-templated string** that `RAGPipeline.build_prompt()` "
        "sends to the LLM."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    question = st.text_input("Question:", "What is deep learning?")
    if not question:
        return

    # ─── Retrieve ───────────────────────────────────────
    retriever = pipeline.vectorstore.as_retriever(
        search_kwargs={"k": pipeline.config.k}
    )
    docs = retriever.invoke(question)

    # ─── Build context (same logic as rag_chain.ask) ───
    context = "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))}]: {d.page_content[:400]}"
        for d in docs
    )

    # ─── History ────────────────────────────────────────
    if pipeline.history:
        hist_str = "\n".join(
            f"{role}: {msg[:150]}" for role, msg in pipeline.history[-4:]
        )
    else:
        hist_str = "(empty)"

    # ─── Show each logical section colour-coded ─────────
    st.markdown("##### 1. The four logical sections")

    sections = [
        ("system",   "#8e24aa", pipeline.config.system_prompt),
        ("context",  "#1976d2", context),
        ("history",  "#fb8c00", hist_str),
        ("question", "#43a047", question),
    ]
    for label, color, value in sections:
        st.markdown(
            f"""<div style="margin:0.6rem 0;">
                <div style="font-family:monospace; color:{color}; font-weight:600;
                            font-size:0.82rem; margin-bottom:0.2rem;">
                    {label}
                </div>
                <div style="background:{color}11; border-left:3px solid {color};
                            padding:0.7rem 1rem; border-radius:0 6px 6px 0;
                            font-family: ui-monospace, monospace; font-size:0.82rem;
                            white-space:pre-wrap; max-height:240px; overflow:auto;">
                    {value}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ─── Actual prompt sent to LLM (after chat-template) ─
    st.markdown("##### 2. 🎯 Actual prompt sent to the LLM (with chat template)")
    full_prompt = pipeline.build_prompt(context, hist_str, question)

    used_chat_template = (
        pipeline.tokenizer is not None
        and ("<|im_start|>" in full_prompt or "<|begin_of_text|>" in full_prompt
             or "<s>" in full_prompt or "[INST]" in full_prompt)
    )
    if used_chat_template:
        st.success("✅ The model's chat template was applied — the LLM will see proper role tokens.")
    else:
        st.warning("⚠️ Plain-text fallback (no chat template) — older or non-chat models.")

    st.code(full_prompt, language="text")

    c1, c2 = st.columns(2)
    c1.metric("Prompt length", f"{len(full_prompt):,} chars")
    c2.metric("Approx. tokens", f"~{len(full_prompt) // 4:,}")


# ──────────────────────────────────────────────
# 5. FULL INPUT VIEW playground
# ──────────────────────────────────────────────

def _playground_full_input():
    """
    Show every piece of information that flows INTO the LLM, in one place.

      1. The raw query
      2. Each retrieved chunk in full (untruncated)
      3. The exact chat-templated prompt the LLM receives

    This is the most "X-ray" view of the RAG system — useful when an answer
    looks wrong and you want to find out which step is to blame.
    """
    st.markdown("#### 🎯 Full input view")
    st.caption(
        "The complete picture: query + retrieved chunks (untruncated) + "
        "exact chat-templated prompt. This is everything the LLM sees."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    # ─── Query input ──────────────────────────────────
    query = st.text_input("Query:", "What is deep learning?")
    if not query:
        return

    # Run retrieval to get the chunks the LLM will see
    retriever = pipeline.vectorstore.as_retriever(
        search_kwargs={"k": pipeline.config.k}
    )
    docs = retriever.invoke(query)
    scored = pipeline.vectorstore.similarity_search_with_score(
        query, k=pipeline.config.k
    )

    # ─── 1. The query itself ──────────────────────────
    st.markdown("##### 1️⃣  The query")
    st.markdown(
        f"""<div style="background:#43a04722; border-left:4px solid #43a047;
                      padding:1rem 1.2rem; border-radius:0 10px 10px 0;
                      font-size:1.05rem; margin-bottom:1rem;">
            <span style="color:#43a047; font-family:monospace; font-size:0.78rem;
                         font-weight:600">USER QUERY</span><br>
            <span style="font-size:1.1rem; color:#222">{query}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ─── 2. The retrieved chunks (full, untruncated) ─
    st.markdown(f"##### 2️⃣  Retrieved chunks ({len(docs)} total)")
    st.caption(
        "These are the **full, untruncated** chunks. "
        "Note: in `rag_chain.ask()` they're truncated to 400 chars before being "
        "concatenated into the prompt — that's why the prompt below may look shorter."
    )

    for rank, (doc, score) in enumerate(scored, start=1):
        sim = max(0.0, 1.0 - score / 2.0)
        source = os.path.basename(doc.metadata.get("source", "?"))
        n_chars = len(doc.page_content)
        n_words = len(doc.page_content.split())
        bar_pct = sim * 100

        # Card with full chunk content
        st.markdown(
            f"""<div style="border:1px solid #ddd; border-radius:10px;
                          padding:1rem 1.2rem; margin:0.6rem 0;
                          background:#fafafa;">
                <div style="display:flex; align-items:center; gap:0.6rem;
                            margin-bottom:0.5rem; flex-wrap:wrap;">
                    <span style="background:#1976d2; color:white;
                                 padding:3px 12px; border-radius:100px;
                                 font-size:0.78rem; font-weight:600;">
                        CHUNK #{rank}
                    </span>
                    <code style="font-size:0.82rem;">📄 {source}</code>
                    <span style="font-family:monospace; font-size:0.78rem; color:#666;">
                        {n_chars} chars · {n_words} words
                    </span>
                    <span style="margin-left:auto; font-family:monospace;
                                 font-size:0.82rem; color:#666;">
                        L2={score:.3f} · sim={sim:.3f}
                    </span>
                </div>
                <div style="background:#e0e0e0; border-radius:6px; height:6px;
                            overflow:hidden; margin-bottom:0.7rem;">
                    <div style="background:linear-gradient(90deg,
                                #e53935, #fdd835, #43a047);
                                width:{bar_pct:.1f}%; height:100%;"></div>
                </div>
                <div style="font-size:0.9rem; line-height:1.6; color:#222;
                            white-space:pre-wrap; max-height:400px; overflow:auto;
                            padding:0.5rem; background:white; border-radius:6px;
                            border:1px solid #eee;">
                    {doc.page_content}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ─── 3. The chat-templated prompt ─────────────────
    st.markdown("##### 3️⃣  The exact prompt sent to the LLM")
    st.caption(
        "After retrieval, the chunks are truncated to 400 chars each, joined, "
        "and wrapped in the model's chat template by `pipeline.build_prompt()`."
    )

    # Build context the same way rag_chain.ask() does
    context = "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))}]: "
        f"{d.page_content[:400]}"
        for d in docs
    )
    hist_str = (
        "\n".join(f"{r}: {m[:150]}" for r, m in pipeline.history[-4:])
        if pipeline.history else "(empty)"
    )
    full_prompt = pipeline.build_prompt(context, hist_str, query)

    # Detect whether chat template was applied
    used_chat_template = (
        pipeline.tokenizer is not None
        and (
            "<|im_start|>" in full_prompt or "<|begin_of_text|>" in full_prompt
            or "<s>" in full_prompt or "[INST]" in full_prompt
        )
    )
    if used_chat_template:
        st.success(
            "✅ Chat template applied — the LLM sees proper role tokens "
            "(`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`)."
        )
    else:
        st.warning(
            "⚠️ Plain-text fallback — no chat template was applied. "
            "The model may produce poor answers."
        )

    st.code(full_prompt, language="text")

    # ─── Summary metrics ──────────────────────────────
    st.markdown("##### 📊 Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieved chunks", len(docs))
    c2.metric("Total chunk chars", sum(len(d.page_content) for d in docs))
    c3.metric("Final prompt chars", f"{len(full_prompt):,}")
    c4.metric("Approx. tokens", f"~{len(full_prompt) // 4:,}")


# ──────────────────────────────────────────────
# 6. FULL TRACE playground
# ──────────────────────────────────────────────

def _playground_full_trace():
    st.markdown("#### 🚀 End-to-end trace")
    st.caption(
        "Calls `RAGPipeline.ask(question)` and shows every intermediate step "
        "with its timing — exactly what happens when you click Send in the Chat tab."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    question = st.text_input("Question:", "What is deep learning?")
    if not st.button("🚀 Run trace", type="primary"):
        return
    if not question:
        st.warning("Type a question first.")
        return

    timings = {}

    # ─── Step 1: retrieve ──────────────────────────────
    t = time.time()
    retriever = pipeline.vectorstore.as_retriever(
        search_kwargs={"k": pipeline.config.k}
    )
    docs = retriever.invoke(question)
    scored = pipeline.vectorstore.similarity_search_with_score(
        question, k=pipeline.config.k
    )
    timings["retrieve"] = time.time() - t

    with st.expander(
        f"1️⃣  Retrieve — {timings['retrieve']*1000:.1f} ms",
        expanded=True,
    ):
        st.markdown(f"**Found {len(docs)} chunks** (k={pipeline.config.k})")
        for i, (d, s) in enumerate(scored, 1):
            src = os.path.basename(d.metadata.get("source", "?"))
            st.markdown(f"- **#{i}** `{src}` — score {s:.3f}")

    # ─── Step 2: build context ─────────────────────────
    t = time.time()
    context = "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))}]: {d.page_content[:400]}"
        for d in docs
    )
    hist_str = "\n".join(f"{r}: {m[:150]}" for r, m in pipeline.history[-4:]) \
               if pipeline.history else "(empty)"
    timings["context"] = time.time() - t

    with st.expander(f"2️⃣  Build context — {timings['context']*1000:.2f} ms"):
        preview = context[:2000] + ("..." if len(context) > 2000 else "")
        st.code(preview, language="text")

    # ─── Step 3: chat-templated prompt ─────────────────
    t = time.time()
    chat_prompt = pipeline.build_prompt(context, hist_str, question)
    timings["prompt"] = time.time() - t

    with st.expander(f"3️⃣  Apply chat template — {timings['prompt']*1000:.2f} ms"):
        st.caption(
            "The model's tokenizer wraps everything in role tokens "
            "(`<|im_start|>system`, `<|im_start|>user`, `<|im_start|>assistant`) "
            "so the LLM knows it's its turn to answer."
        )
        st.code(chat_prompt, language="text")

    # ─── Step 4: full ask (LLM generation) ─────────────
    t = time.time()
    response = pipeline.ask(question)
    timings["full"] = time.time() - t

    with st.expander(
        f"4️⃣  Generate (full ask) — {timings['full']*1000:.0f} ms",
        expanded=True,
    ):
        st.markdown("**Answer:**")
        st.success(response.answer)

    # ─── Summary ────────────────────────────────────────
    st.markdown("##### Pipeline timing summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieve", f"{timings['retrieve']*1000:.1f} ms")
    c2.metric("Context+Template", f"{(timings['context']+timings['prompt'])*1000:.2f} ms")
    c3.metric("Total ask()", f"{response.elapsed:.2f} s")
    c4.metric("Chunks used", response.n_chunks)


# ──────────────────────────────────────────────
# 7. PERSISTENCE DEMO
# ──────────────────────────────────────────────

def _playground_persistence():
    """
    Side-by-side, hands-on demo of the TWO kinds of persistence used
    by this app:

      LEFT  — Streamlit `session_state` (in-memory, per browser session)
      RIGHT — Disk persistence (files in assets/, survives everything)

    Students can:
      • Click +1 on a counter and watch it survive reruns
      • Save a string to disk and see it appear on next page load
      • Inspect what's currently in session_state vs assets/
    """
    import json
    import datetime as dt
    from pathlib import Path
    from backend.config import ASSETS_DIR, FAISS_INDEX_PATH

    st.markdown("#### 💾 Persistence demo")
    st.caption(
        "Every Streamlit app deals with two completely different kinds of state. "
        "This page shows both at work, side by side, on real values from THIS app."
    )

    # Quick conceptual primer
    st.info(
        "**TL;DR**\n\n"
        "• `st.session_state` lives in your browser session. Survives clicks "
        "and reruns. **Dies when Streamlit restarts.**\n\n"
        "• Files in `assets/` live on disk. **Survive everything** — restarts, "
        "reboots, even moving the project to another machine."
    )

    col_left, col_right = st.columns(2, gap="large")

    # ╔═════════════════════════════════════════════════╗
    # ║  LEFT — Streamlit session_state                 ║
    # ╚═════════════════════════════════════════════════╝
    with col_left:
        st.markdown(
            "<div style='background:#1976d211; border-left:4px solid #1976d2; "
            "padding:0.6rem 0.9rem; border-radius:0 8px 8px 0; margin-bottom:0.8rem;'>"
            "<b style='color:#1976d2'>🧠 Streamlit `session_state`</b><br>"
            "<span style='font-size:0.82rem; color:#555'>"
            "In-memory dict, per browser session. Resets on server restart.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ─── Live counter demo ────────────────────────
        st.markdown("**Live counter** (uses `st.session_state.demo_counter`)")
        if "demo_counter" not in st.session_state:
            st.session_state.demo_counter = 0

        st.metric("Current value", st.session_state.demo_counter)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ +1", use_container_width=True, key="ss_inc"):
                st.session_state.demo_counter += 1
                st.rerun()
        with c2:
            if st.button("🔄 Reset", use_container_width=True, key="ss_reset"):
                st.session_state.demo_counter = 0
                st.rerun()

        st.caption(
            "Click +1 a few times. Refresh the browser tab — counter stays. "
            "Stop & restart `streamlit run app.py` — counter resets to 0."
        )

        st.markdown("---")

        # ─── Live introspection of session_state ──────
        st.markdown("**Live `st.session_state` from this very app**")
        st.caption("Everything Streamlit is keeping in memory right now:")

        # Build a friendly summary of everything currently in session_state.
        # Some values (FAISS index, LLM, numpy arrays) are huge — we just
        # show their type and size, never the value itself.
        rows = []
        for key in sorted(st.session_state.keys()):
            val = st.session_state[key]
            type_name = type(val).__name__

            # Pretty-print common types
            if isinstance(val, (int, float, bool, str)) and len(str(val)) < 60:
                summary = repr(val)
            elif isinstance(val, list):
                summary = f"list with {len(val)} items"
            elif isinstance(val, dict):
                summary = f"dict with {len(val)} keys"
            elif type_name == "ndarray":
                summary = f"shape={val.shape}, dtype={val.dtype}"
            elif hasattr(val, "__len__"):
                try:
                    summary = f"{type_name} (len={len(val)})"
                except Exception:
                    summary = f"<{type_name}>"
            else:
                summary = f"<{type_name}>"

            rows.append({"key": key, "type": type_name, "summary": summary})

        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=320,
                         hide_index=True)
        else:
            st.caption("(empty — nothing stored yet in this session)")

    # ╔═════════════════════════════════════════════════╗
    # ║  RIGHT — Disk persistence                       ║
    # ╚═════════════════════════════════════════════════╝
    with col_right:
        st.markdown(
            "<div style='background:#43a04711; border-left:4px solid #43a047; "
            "padding:0.6rem 0.9rem; border-radius:0 8px 8px 0; margin-bottom:0.8rem;'>"
            "<b style='color:#43a047'>💾 Disk persistence</b><br>"
            "<span style='font-size:0.82rem; color:#555'>"
            "Files in `assets/`. Survive restarts, reboots, even relocations.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ─── Persistent test value ────────────────────
        st.markdown("**Live test value** (saved to `assets/test_value.json`)")

        TEST_FILE = ASSETS_DIR / "test_value.json"

        # Read current value if any
        if TEST_FILE.exists():
            try:
                saved = json.loads(TEST_FILE.read_text())
                saved_at = dt.datetime.fromtimestamp(TEST_FILE.stat().st_mtime)
                st.success(
                    f"✅ Currently on disk: **{saved.get('value', '?')}**  \n"
                    f"_(saved {saved_at.strftime('%Y-%m-%d %H:%M:%S')})_"
                )
            except Exception as e:
                st.warning(f"File exists but could not parse: {e}")
        else:
            st.caption("No test value saved yet.")

        new_value = st.text_input(
            "Save a value to disk:", key="disk_input",
            placeholder="e.g., 'hello world' or your name",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", use_container_width=True, key="disk_save"):
                payload = {
                    "value": new_value,
                    "saved_at": dt.datetime.now().isoformat(),
                }
                TEST_FILE.write_text(json.dumps(payload, indent=2))
                st.rerun()
        with c2:
            if st.button("🗑️ Delete", use_container_width=True, key="disk_del"):
                if TEST_FILE.exists():
                    TEST_FILE.unlink()
                st.rerun()

        st.caption(
            "Save a value, then **stop streamlit** (Ctrl+C) and restart. "
            "When you reopen this page the value will still be here."
        )

        st.markdown("---")

        # ─── Live listing of assets/ ──────────────────
        st.markdown("**Live contents of `assets/`**")
        st.caption("Everything the app has persisted to disk so far:")

        files = []
        if ASSETS_DIR.exists():
            for f in sorted(ASSETS_DIR.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(ASSETS_DIR)
                    size_kb = f.stat().st_size / 1024
                    mtime = dt.datetime.fromtimestamp(f.stat().st_mtime)
                    files.append({
                        "path": str(rel),
                        "size": f"{size_kb:,.1f} KB",
                        "modified": mtime.strftime("%H:%M:%S"),
                    })

        if files:
            import pandas as pd
            st.dataframe(
                pd.DataFrame(files),
                use_container_width=True, height=180, hide_index=True,
            )
        else:
            st.caption(
                "(empty — build the pipeline once to populate `assets/` "
                "with the embeddings + FAISS index)"
            )

        # ─── Pretty-print cache_meta.json if it exists ─
        meta_file = ASSETS_DIR / "cache_meta.json"
        if meta_file.exists():
            with st.expander("📄 `assets/cache_meta.json` (pipeline cache fingerprint)"):
                try:
                    meta = json.loads(meta_file.read_text())
                    st.json(meta)
                except Exception as e:
                    st.error(f"Could not parse: {e}")

    # ╔═════════════════════════════════════════════════╗
    # ║  Comparison table                               ║
    # ╚═════════════════════════════════════════════════╝
    st.markdown("---")
    st.markdown("##### When to use which?")

    comparison = [
        ("Survives a script rerun (clicking a button)",   "✅",  "✅"),
        ("Survives a browser tab refresh (F5)",           "✅",  "✅"),
        ("Survives a Streamlit restart (Ctrl+C / start)", "❌",  "✅"),
        ("Survives a system reboot",                       "❌",  "✅"),
        ("Per-user (different sessions = different data)", "✅",  "❌ (shared)"),
        ("Free / instant",                                  "✅",  "✅"),
        ("Good for ML models / large arrays",               "✅",  "✅"),
        ("Good for the result of a 60s computation",        "❌",  "✅"),
    ]
    import pandas as pd
    df = pd.DataFrame(comparison, columns=[
        "Property", "session_state", "Disk (assets/)",
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(
        "##### How RAG Studio uses both\n"
        "- **session_state** holds the in-memory pipeline objects: "
        "the `RAGPipeline`, the loaded LLM, the vectorstore reference, "
        "the chunks, the embeddings matrix. They're fast to access during "
        "a chat session and never need to be re-created within one run.\n"
        "- **Disk (`assets/`)** holds the things that took a long time to compute: "
        "the FAISS index (`faiss_index/index.faiss` + `index.pkl`), the raw "
        "embeddings matrix (`embeddings.npy`) and the cache fingerprint "
        "(`cache_meta.json`). On startup, `_try_autoload()` in `app.py` "
        "uses the fingerprint to decide if the disk cache is still valid "
        "for the current settings — if yes, it reloads everything in ~2s "
        "instead of re-embedding for ~60s."
    )

