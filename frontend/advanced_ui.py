"""
frontend/advanced_ui.py — Advanced RAG techniques tab
=======================================================
A dedicated tab presenting three optional enhancements to the basic RAG
pipeline. Each technique is implemented in its own module under
`advanced/` and visualised here in an interactive way:

  🌱 Query Expansion    — let the LLM rewrite the query into multiple
                          phrasings, then union the FAISS results
  🤖 ReAct Agent        — Thought → Action → Observation loop with tools
                          (search, calculator, finish)
  🗜️ Memory Compression — periodically summarize old conversation turns
                          to keep prompts bounded as chats grow long

Each section calls the actual `advanced/*` module so students see the
real implementation, not a mock.
"""

import os
import streamlit as st


# ────────────────────────────────────────────────────────────
# Top-level router
# ────────────────────────────────────────────────────────────

def render_advanced():
    """Top-level renderer for the Advanced tab."""

    st.markdown("### 🧪 Advanced RAG techniques")
    st.caption(
        "Three optional enhancements that go beyond the basic RAG flow. "
        "Each calls a real implementation under `advanced/`."
    )

    technique = st.radio(
        "Pick a technique:",
        [
            "🌱  Query Expansion",
            "🤖  ReAct Agent",
            "🗜️  Memory Compression",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if "Query Expansion" in technique:
        _advanced_query_expansion()
    elif "ReAct" in technique:
        _advanced_react()
    elif "Memory Compression" in technique:
        _advanced_memory_compression()


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _show_chunks_compact(results):
    """Display a list of (Document, score) tuples compactly."""
    if not results:
        st.caption("(no chunks)")
        return
    for i, (doc, score) in enumerate(results, 1):
        sim = max(0.0, 1.0 - score / 2.0)
        source = os.path.basename(doc.metadata.get("source", "?"))
        st.markdown(
            f"<div style='background:#fafafa; border:1px solid #e0e0e0; "
            f"border-radius:8px; padding:0.5rem 0.7rem; margin:0.3rem 0;'>"
            f"<span style='font-family:monospace; font-size:0.78rem; color:#666;'>"
            f"#{i} · {source} · sim={sim:.3f}</span><br>"
            f"<span style='font-size:0.85rem;'>"
            f"{doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}"
            f"</span></div>",
            unsafe_allow_html=True,
        )



# ════════════════════════════════════════════════════════════
# ADVANCED PLAYGROUND STAGES
# ════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────
# 8. QUERY EXPANSION
# ──────────────────────────────────────────────

def _advanced_query_expansion():
    """
    Side-by-side: naive single-query retrieval vs. multi-query
    retrieval after LLM-based expansion.
    """
    st.markdown("#### 🌱 Query Expansion")
    st.caption(
        "**Module:** `advanced/query_expansion.py`. "
        "Asks the LLM to rewrite the user's question into 3 alternative "
        "search queries, then unions the FAISS results."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    from advanced.query_expansion import expand_query, multi_query_search

    query = st.text_input("Query:",
                          "How does it work?",
                          help="Try a vague or ambiguous query — that's where expansion helps most.")
    n_queries = st.slider("Number of expansions", 2, 5, 3)

    if not st.button("🌱 Expand & retrieve", type="primary"):
        return
    if not query:
        st.warning("Type a query first.")
        return

    # ─── Step 1: expand ───────────────────────────────
    with st.spinner("Asking the LLM for alternative phrasings..."):
        expanded = expand_query(query, pipeline.llm, n_queries=n_queries)

    st.markdown("##### 1️⃣  Expanded queries")
    for i, q in enumerate(expanded):
        prefix = "🔵 (original)" if i == 0 else f"🟢 (variant {i})"
        st.markdown(f"{prefix}  `{q}`")

    # ─── Step 2: side-by-side retrieval ───────────────
    st.markdown("##### 2️⃣  Retrieval comparison")
    col_naive, col_expanded = st.columns(2)

    with col_naive:
        st.markdown("**Naive (1 query)**")
        naive_results = pipeline.vectorstore.similarity_search_with_score(
            query, k=pipeline.config.k
        )
        _show_chunks_compact(naive_results)

    with col_expanded:
        st.markdown(f"**Expanded ({len(expanded)} queries → unioned)**")
        expanded_results = multi_query_search(
            expanded,
            pipeline.vectorstore,
            k_per_query=3,
            final_k=pipeline.config.k,
        )
        _show_chunks_compact(expanded_results)

    # ─── Step 3: which chunks are NEW? ────────────────
    naive_keys = {d.page_content[:200] for d, _ in naive_results}
    new_chunks = [
        (d, s) for d, s in expanded_results
        if d.page_content[:200] not in naive_keys
    ]

    st.markdown("##### 3️⃣  What expansion found that the naive query missed")
    if not new_chunks:
        st.info("Expansion didn't surface any NEW chunks for this query — "
                "the original was good enough on its own.")
    else:
        st.success(
            f"Expansion found {len(new_chunks)} **new** chunk(s) "
            "that single-query retrieval missed:"
        )
        _show_chunks_compact(new_chunks)



# ──────────────────────────────────────────────
# 9. ReAct AGENT
# ──────────────────────────────────────────────

def _advanced_react():
    """
    Interactive ReAct loop: shows every Thought/Action/Observation
    triplet as it unfolds.
    """
    st.markdown("#### 🤖 ReAct Agent")
    st.caption(
        "**Module:** `advanced/react_agent.py`. "
        "The LLM plans step-by-step and uses tools (`search`, `calculator`, `finish`). "
        "Best for multi-hop questions that need several lookups or computations."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    from advanced.react_agent import run_react

    # ─── Friendly examples to seed ─────────────────────
    examples = {
        "Multi-hop lookup": "What is deep learning, and what is its relation to neural networks?",
        "With arithmetic":  "If a chunk is 500 characters and 1 token is roughly 4 characters, "
                            "how many tokens fit in a chunk?",
        "Comparison":       "Compare convolutional and recurrent neural networks.",
    }

    col_q, col_ex = st.columns([3, 1])
    with col_ex:
        choice = st.selectbox("Examples", ["—"] + list(examples.keys()))
    with col_q:
        default_q = examples.get(choice, "")
        question = st.text_input("Question:", value=default_q,
                                 placeholder="Ask something that needs >1 lookup...")

    max_iters = st.slider("Max iterations", 2, 8, 4,
                          help="Hard cap on the Thought→Action→Observation loop.")

    if not st.button("🚀 Run ReAct", type="primary"):
        return
    if not question:
        st.warning("Type a question first.")
        return

    with st.spinner("Running the ReAct loop..."):
        trace = run_react(question, pipeline.llm, pipeline.vectorstore,
                          max_iterations=max_iters)

    # ─── Summary metrics ──────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations", trace.n_iterations)
    c2.metric("Stopped", trace.stopped_reason)
    c3.metric("Elapsed", f"{trace.elapsed:.1f} s")
    c4.metric("Tools used", len({s.action_name for s in trace.steps}))

    # ─── Step-by-step trace ───────────────────────────
    st.markdown("##### Step-by-step trace")
    if not trace.steps:
        st.warning("The agent didn't produce any valid step. "
                   "Try a more capable LLM (Phi-3 or Qwen 3B) for ReAct.")

    palette = {"search": "#1976d2", "calculator": "#fb8c00", "finish": "#43a047"}
    for i, step in enumerate(trace.steps, 1):
        color = palette.get(step.action_name, "#757575")
        st.markdown(
            f"""<div style="border-left:3px solid {color};
                          background:{color}0A;
                          border-radius:0 8px 8px 0;
                          padding:0.7rem 1rem; margin:0.5rem 0;">
                <div style="font-family:monospace; font-size:0.74rem; color:{color};
                            font-weight:600; margin-bottom:0.4rem;">
                    STEP {i}
                </div>
                <div style="font-size:0.9rem; margin-bottom:0.3rem;">
                    💭 <b>Thought:</b> {step.thought}
                </div>
                <div style="font-size:0.9rem; margin-bottom:0.3rem;">
                    ⚡ <b>Action:</b>
                    <code style="color:{color}; font-weight:600;">
                        {step.action_name}({step.action_input})
                    </code>
                </div>
                <div style="font-size:0.85rem; background:white;
                            padding:0.5rem 0.7rem; border-radius:6px;
                            border:1px solid #eee; white-space:pre-wrap;
                            max-height:160px; overflow:auto;">
                    👁️ <b>Observation:</b> {step.observation}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ─── Final answer ─────────────────────────────────
    st.markdown("##### Final answer")
    st.success(trace.final_answer or "(no answer produced)")


# ──────────────────────────────────────────────
# 10. MEMORY COMPRESSION
# ──────────────────────────────────────────────

def _advanced_memory_compression():
    """
    Show how a long conversation history can be replaced by a
    short LLM-generated summary while preserving the key info.
    """
    st.markdown("#### 🗜️ Memory Compression")
    st.caption(
        "**Module:** `advanced/memory_compression.py`. "
        "When chat history grows long, prompts bloat. Compression "
        "replaces old turns with a short summary to keep prompts bounded."
    )

    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.info("ℹ️ Build the pipeline first.")
        return

    from advanced.memory_compression import compress_history

    # ─── Sample history (or use the live one) ─────────
    sample_history = [
        ("User", "Hi! Can you help me understand deep learning?"),
        ("Assistant", "Of course! Deep learning is a subset of machine learning that uses "
                      "neural networks with many layers."),
        ("User", "What about chunking? What size should I use?"),
        ("Assistant", "Chunk size is a tradeoff. 500 characters is a good default — "
                      "smaller loses context, larger dilutes retrieval precision."),
        ("User", "And embeddings — which model do you recommend?"),
        ("Assistant", "For English, all-mpnet-base-v2 is a great default. It's 768-dimensional "
                      "and well-balanced between speed and accuracy."),
        ("User", "What's the difference between PCA and t-SNE for visualization?"),
        ("Assistant", "PCA is a linear projection — fast, preserves global structure. "
                      "t-SNE is non-linear — slower, better at revealing local clusters."),
        ("User", "Got it. So what about temperature for the LLM?"),
        ("Assistant", "Temperature controls randomness. 0.3 is good for factual answers, "
                      "0.8+ for creative writing."),
    ]

    use_live = st.checkbox(
        "Use live conversation history (from the Chat tab)",
        value=False,
        help="If unchecked, a 10-turn sample is used so the demo always works.",
    )

    if use_live and pipeline.history:
        history = list(pipeline.history)
    else:
        history = sample_history
        if use_live:
            st.caption("(no live history yet — falling back to the sample)")

    keep_last_n = st.slider(
        "Keep last N turns verbatim", 0, 6, 2,
        help="Recent turns are KEPT as-is; older ones get summarized.",
    )

    # ─── Show the BEFORE state ────────────────────────
    st.markdown("##### Before compression")
    chars_before = sum(len(m) for _, m in history)
    st.caption(f"{len(history)} turns · {chars_before:,} chars total")

    with st.expander("📜 Full history (before)", expanded=True):
        for role, msg in history:
            st.markdown(
                f"**{role}:** {msg}"
            )

    # ─── Run compression on click ─────────────────────
    if not st.button("🗜️ Compress", type="primary"):
        return

    with st.spinner("Asking the LLM to summarize old turns..."):
        new_history, result = compress_history(
            history, pipeline.llm, keep_last_n=keep_last_n,
        )

    # ─── Metrics ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turns compressed", result.n_turns_compressed)
    c2.metric("Before", f"{result.chars_before:,} chars")
    c3.metric("After", f"{result.chars_after:,} chars")
    c4.metric("Reduction", f"{result.reduction_pct:.0f}%")

    # ─── AFTER ───────────────────────────────────────
    st.markdown("##### After compression")
    chars_after_total = sum(len(m) for _, m in new_history)
    st.caption(f"{len(new_history)} entries · {chars_after_total:,} chars total")

    for role, msg in new_history:
        if role == "Summary":
            st.markdown(
                f"<div style='background:#fb8c0011; border-left:3px solid #fb8c00;"
                f"padding:0.7rem 1rem; border-radius:0 8px 8px 0; margin:0.4rem 0;'>"
                f"<b style='color:#fb8c00;'>📝 SUMMARY of older turns</b><br>"
                f"<span style='font-size:0.9rem;'>{msg}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{role}:** {msg}")

    st.info(
        "💡 In a real app, this compression would run automatically every "
        "few turns inside `RAGPipeline.ask()`, keeping prompts bounded "
        "regardless of conversation length."
    )
