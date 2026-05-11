"""
frontend/evaluation_ui.py — RAG Evaluation tab
================================================
Lets the user measure their RAG pipeline against a test set:
  • Use the built-in 5-case demo set, OR
  • Upload a CSV with columns: question, expected_answer, expected_keywords

Shows:
  • Lightweight metrics (always available)
  • RAGAS metrics (when ragas + OPENAI_API_KEY are configured)
  • Per-case results in a sortable table
  • Aggregated charts
"""

import io
import os
import streamlit as st
import pandas as pd

from utilities.evaluation import (
    EvalCase,
    EvalResult,
    DEMO_CASES,
    run_evaluation,
    ragas_available,
    run_ragas,
)


def render_evaluation():
    """Top-level renderer for the Evaluation tab."""

    st.markdown("### 📏 Evaluation")
    st.caption(
        "Measure your RAG pipeline against a labelled test set. "
        "Two metric families are computed: lightweight ones (always) and "
        "RAGAS ones (if available)."
    )

    pipeline = st.session_state.get("pipeline")
    st_model = st.session_state.get("st_model")

    if pipeline is None or st_model is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    # ─── Test set: built-in or upload ────────────────────
    st.markdown("#### 1️⃣  Pick a test set")
    src = st.radio(
        "Source",
        ["Built-in 5-case demo", "Upload CSV", "Paste manually"],
        horizontal=True,
    )

    cases: list = []

    if src == "Built-in 5-case demo":
        cases = list(DEMO_CASES)
        with st.expander("Preview the demo cases"):
            df = pd.DataFrame([
                {
                    "question":         c.question,
                    "expected_answer":  c.expected_answer,
                    "expected_keywords": ", ".join(c.expected_keywords),
                }
                for c in cases
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

    elif src == "Upload CSV":
        st.caption(
            "Expected columns: `question`, `expected_answer`, "
            "`expected_keywords` (comma-separated, optional)."
        )
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                for _, row in df.iterrows():
                    kws_raw = str(row.get("expected_keywords", ""))
                    kws = [k.strip() for k in kws_raw.split(",") if k.strip()]
                    cases.append(EvalCase(
                        question=str(row["question"]),
                        expected_answer=str(row.get("expected_answer", "")),
                        expected_keywords=kws,
                    ))
                st.success(f"Loaded {len(cases)} cases.")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    else:  # paste manually
        st.caption("Pipe-separated rows: `question | expected_answer | keyword1,keyword2`")
        raw = st.text_area("One case per line", height=160,
            value="What is deep learning?|Deep learning uses neural networks with many layers.|neural,layers\n"
                  "What is FAISS?|FAISS is a library for fast similarity search of dense vectors.|similarity,vectors")
        for line in raw.strip().split("\n"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 1 and parts[0]:
                kws = [k.strip() for k in parts[2].split(",")] if len(parts) > 2 else []
                cases.append(EvalCase(
                    question=parts[0],
                    expected_answer=parts[1] if len(parts) > 1 else "",
                    expected_keywords=kws,
                ))

    if not cases:
        st.warning("No cases loaded yet.")
        return

    # ─── Options ──────────────────────────────────────
    st.markdown("#### 2️⃣  Options")
    col_a, col_b = st.columns(2)
    with col_a:
        ragas_ok, ragas_msg = ragas_available()
        use_ragas = st.checkbox(
            "Compute RAGAS metrics (slower, may need an LLM API key)",
            value=False,
            disabled=not ragas_ok,
        )
        if not ragas_ok:
            with st.expander("Why RAGAS is disabled"):
                st.code(ragas_msg, language="text")
    with col_b:
        st.caption(
            f"Will run **{len(cases)} questions** through the current pipeline. "
            "Expect ~5-15s per question depending on the model."
        )

    # ─── Run ──────────────────────────────────────────
    if not st.button("🚀 Run evaluation", type="primary"):
        return

    progress = st.progress(0.0, text="Starting...")

    def _cb(i, total, case):
        progress.progress((i + 1) / total,
                           text=f"Q{i+1}/{total}: {case.question[:60]}...")

    with st.spinner("Running pipeline on every case..."):
        results = run_evaluation(pipeline, cases, st_model, progress_callback=_cb)

    progress.progress(1.0, text="Done!")

    if use_ragas:
        with st.spinner("Computing RAGAS metrics (may call an LLM judge)..."):
            results = run_ragas(results, pipeline=pipeline)

    # ─── Aggregate ────────────────────────────────────
    st.markdown("#### 3️⃣  Results")

    n = len(results)
    avg_precision   = sum(r.retrieval_precision_at_k for r in results) / n
    avg_similarity  = sum(r.answer_similarity         for r in results) / n
    avg_groundness  = sum(r.groundedness              for r in results) / n
    avg_keywords    = sum(r.keyword_hit_rate          for r in results) / n
    avg_latency     = sum(r.latency_s                 for r in results) / n

    st.markdown("##### 📊 Lightweight metrics (averages)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retrieval precision@k",
              f"{avg_precision:.1%}",
              help="Did the retriever find a chunk containing the answer?")
    c2.metric("Answer similarity",
              f"{avg_similarity:.2f}",
              help="Cosine similarity between LLM answer and ground truth.")
    c3.metric("Groundedness",
              f"{avg_groundness:.1%}",
              help="Fraction of answer trigrams that appear in retrieved chunks.")
    c4.metric("Keyword hit rate",
              f"{avg_keywords:.1%}",
              help="Fraction of expected keywords found in the answer.")
    c5.metric("Avg latency",
              f"{avg_latency:.2f}s")

    # RAGAS metrics (if any were filled)
    has_ragas = any(r.faithfulness is not None for r in results)
    if has_ragas:
        st.markdown("##### 🎯 RAGAS metrics (LLM-as-judge averages)")
        def _mean_or_dash(field):
            vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
            return f"{sum(vals)/len(vals):.2f}" if vals else "—"
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness",      _mean_or_dash("faithfulness"))
        c2.metric("Answer relevancy",  _mean_or_dash("answer_relevancy"))
        c3.metric("Context precision", _mean_or_dash("context_precision"))

    # ─── Per-case detail table ────────────────────────
    st.markdown("##### 📋 Per-case results")
    rows = []
    for r in results:
        row = {
            "Question":     r.case.question[:80],
            "Precision@k":  f"{r.retrieval_precision_at_k:.0f}",
            "Sim":          f"{r.answer_similarity:.2f}",
            "Ground":       f"{r.groundedness:.0%}",
            "Keywords":     f"{r.keyword_hit_rate:.0%}",
            "Latency":      f"{r.latency_s:.1f}s",
        }
        if has_ragas:
            row["Faith"]    = f"{r.faithfulness:.2f}"  if r.faithfulness is not None else "—"
            row["Rel"]      = f"{r.answer_relevancy:.2f}" if r.answer_relevancy is not None else "—"
            row["CtxPrec"]  = f"{r.context_precision:.2f}" if r.context_precision is not None else "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ─── Per-case expanders (for diving deeper) ───────
    st.markdown("##### 🔬 Inspect each case")
    for i, r in enumerate(results, 1):
        # Choose an emoji based on the precision@k result
        icon = "✅" if r.retrieval_precision_at_k >= 1.0 else "❌"
        with st.expander(f"{icon} Q{i}: {r.case.question}"):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Expected:**")
                st.success(r.case.expected_answer or "(no ground truth)")
                if r.case.expected_keywords:
                    st.caption("Keywords: " + ", ".join(r.case.expected_keywords))
            with cb:
                st.markdown("**Actual:**")
                st.info(r.actual_answer)

            st.markdown("**Retrieved chunks:**")
            for j, c in enumerate(r.retrieved_chunks, 1):
                preview = c[:300].replace("\n", " ")
                st.caption(f"#{j} — {preview}...")

    # ─── Export ───────────────────────────────────────
    st.markdown("##### 💾 Export results")
    out_df = pd.DataFrame([{
        "question":       r.case.question,
        "expected":       r.case.expected_answer,
        "actual":         r.actual_answer,
        "precision_at_k": r.retrieval_precision_at_k,
        "similarity":     r.answer_similarity,
        "groundedness":   r.groundedness,
        "keyword_hits":   r.keyword_hit_rate,
        "latency_s":      r.latency_s,
        "faithfulness":   r.faithfulness,
        "answer_rel":     r.answer_relevancy,
        "ctx_precision":  r.context_precision,
    } for r in results])
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv_buf.getvalue(),
        file_name="rag_evaluation.csv",
        mime="text/csv",
    )
