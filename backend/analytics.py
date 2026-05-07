"""
backend/analytics.py — Embedding Analytics & Visualization
============================================================
Tools for understanding the embedding space and retrieval quality.

VISUALIZATIONS:
  - PCA: Fast, shows global structure
  - t-SNE: Slower, shows local clusters better
  - Query projection: See where a query lands relative to chunks
  - Similarity distribution: Histogram of scores for a query
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

import os


def detect_topic(text: str) -> str:
    """Auto-detect the topic of a chunk by keyword matching."""
    t = text.lower()
    if any(w in t for w in ["neural network", "deep learn", "backpropag", "perceptron"]): return "Neural Networks"
    if any(w in t for w in ["nlp", "natural language", "token", "sentiment", "text mining"]): return "NLP"
    if any(w in t for w in ["computer vision", "image", "convolut", "object detect"]): return "Computer Vision"
    if any(w in t for w in ["transformer", "attention", "bert", "gpt"]): return "Transformers"
    if any(w in t for w in ["retrieval", "rag", "augment", "vector"]): return "RAG"
    if any(w in t for w in ["python", "programming", "code", "software"]): return "Programming"
    if any(w in t for w in ["machine learn", "supervised", "unsupervised", "train"]): return "Machine Learning"
    if any(w in t for w in ["artificial intellig", "ai ", "turing"]): return "AI General"
    if any(w in t for w in ["large language", "llm", "chatgpt", "instruct"]): return "LLMs"
    return "Other"


TOPIC_COLORS = {
    "Neural Networks": "#E53935", "NLP": "#1E88E5", "Computer Vision": "#43A047",
    "Transformers": "#FB8C00", "RAG": "#8E24AA", "Programming": "#00ACC1",
    "Machine Learning": "#F4511E", "AI General": "#3949AB", "LLMs": "#C0CA33",
    "Other": "#90A4AE",
}


def compute_embeddings_2d(
    embeddings: np.ndarray,
    method: str = "pca"
) -> np.ndarray:
    """Reduce embeddings to 2D for visualization."""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        perp = min(30, embeddings.shape[0] - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=max(2, perp))
    return reducer.fit_transform(embeddings)


def plot_embedding_space(
    chunks: List[Document],
    embeddings: np.ndarray,
    method: str = "pca",
    query: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    top_indices: Optional[List[int]] = None,
) -> go.Figure:
    """
    Create an interactive Plotly scatter plot of the embedding space.
    
    Args:
        chunks:          List of Document objects
        embeddings:      numpy array of shape (n_chunks, dim)
        method:          "pca" or "tsne"
        query:           Optional query string to highlight
        query_embedding: Optional query vector
        top_indices:     Optional list of retrieved chunk indices
    """
    coords = compute_embeddings_2d(embeddings, method)
    topics = [detect_topic(c.page_content) for c in chunks]
    sources = [os.path.basename(c.metadata.get("source", "?")) for c in chunks]
    previews = [c.page_content[:120] + "..." for c in chunks]

    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "topic": topics, "source": sources, "preview": previews,
        "chunk_id": list(range(len(chunks))),
    })

    fig = px.scatter(
        df, x="x", y="y", color="topic",
        color_discrete_map=TOPIC_COLORS,
        hover_data=["chunk_id", "source", "preview"],
        title=f"Embedding Space ({method.upper()})",
        width=800, height=550,
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white")))

    # Highlight retrieved chunks
    if top_indices:
        fig.add_trace(go.Scatter(
            x=coords[top_indices, 0], y=coords[top_indices, 1],
            mode="markers", marker=dict(size=18, symbol="star", color="red",
                                         line=dict(width=2, color="black")),
            name="Retrieved", showlegend=True,
        ))

    # Show query position (approximate: nearest retrieved chunk)
    if query and top_indices:
        qx, qy = coords[top_indices[0], 0], coords[top_indices[0], 1]
        fig.add_trace(go.Scatter(
            x=[qx], y=[qy], mode="markers+text",
            marker=dict(size=22, symbol="x", color="blue",
                        line=dict(width=3, color="black")),
            text=[f'Query: "{query[:30]}..."'], textposition="top center",
            name="Query", showlegend=True,
        ))

    fig.update_layout(legend=dict(font=dict(size=11)), margin=dict(t=40, b=20))
    return fig


def plot_similarity_histogram(
    scores: np.ndarray,
    query: str,
    k: int = 5,
) -> go.Figure:
    """Plot the distribution of similarity scores for a query."""
    threshold = np.sort(scores)[-k] if len(scores) >= k else scores.min()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=scores, nbinsx=40, marker_color="#1976D2", opacity=0.7, name="All chunks"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Top-{k} threshold ({threshold:.3f})")
    fig.update_layout(
        title=f'Similarity Distribution: "{query[:40]}..."',
        xaxis_title="Cosine Similarity", yaxis_title="Count",
        width=700, height=350,
    )
    return fig
