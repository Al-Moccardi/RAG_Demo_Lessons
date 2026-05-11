# 🤖 RAG Studio

> A complete, fully-local, didactic **Retrieval-Augmented Generation** system.
> Built with **LangChain · FAISS · HuggingFace · Streamlit**.
> Designed for teaching, not for production — but it works on real documents.


---

## 📋 Table of contents

- [What is this?](#-what-is-this)
- [Quick start (clone & run)](#-quick-start-clone--run)
- [Detailed installation](#-detailed-installation)
- [First run — your first question](#-first-run--your-first-question)
- [Tour of the eight tabs](#-tour-of-the-eight-tabs)
- [Project structure](#-project-structure)
- [Configuration cheat sheet](#-configuration-cheat-sheet)
- [Available models](#-available-models)
- [Troubleshooting](#-troubleshooting)
- [Pushing your fork to GitHub](#-pushing-your-fork-to-github)
- [License & credits](#-license--credits)

---

## 🎯 What is this?

RAG Studio is an end-to-end RAG demo that lets students:

- **Build** a RAG pipeline from scratch on Wikipedia, PDFs, scanned images, or their own documents
- **Chat** with a fully local LLM (no API keys, no costs, no data leaving the machine)
- **Inspect** every internal step in an interactive **Playground** (chunking, embedding, retrieval, prompt assembly, ReAct, …)
- **Measure** quality with built-in **lightweight metrics** + optional **RAGAS** integration
- **Compare** 5 embedding models and 5 LLMs by flicking a sidebar dropdown

### Highlights

| Feature | Notes |
|---|---|
| 🏠 Interactive architecture diagram | Click any module → see its inputs/outputs/source code |
| 💬 Streaming chat with citations | Tokens stream live; every answer cites the chunks it used |
| 📊 3D / 2D embedding explorer | PCA / t-SNE with query overlay, retrieved chunks highlighted |
| 📚 Drag-and-drop knowledge base | PDF · TXT · JPG · PNG (auto-OCR for images and scanned PDFs) |
| 🎮 Playground with 8 stages | OCR · 5-way Chunking comparison · Embedding · Retrieval · Prompt · Full Input · Full Trace · Persistence |
| 🧪 Advanced techniques | Query Expansion · ReAct Agent · Memory Compression |
| 📏 Evaluation tab | Precision@k · answer similarity · groundedness · latency, + optional RAGAS |
| 🚀 GitHub Push tab | Guided git init → commit → push (copy-paste **and** click-to-run) |
| 💾 Persistent pipeline cache | First build: ~60 s. Subsequent restarts: ~2 s. |
| 🔎 Tesseract OCR | Built-in, with bounding-box visualisation in the playground |

---

## 🚀 Quick start (clone & run)

If you already have **Python 3.9+**, **git**, and **Tesseract** installed, the whole thing is four commands:

```bash
git clone https://github.com/YOUR_USERNAME/rag-studio.git
cd rag-studio
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Click **📥 Download AI Articles** in the sidebar, then **🔨 Build Pipeline**, then chat away.

> ⚠️ **First build takes ~60–90 seconds** because HuggingFace has to download the embedding model (~420 MB) and the LLM (~3 GB) the first time. Subsequent runs are near-instant thanks to the cache.

---

## 🔧 Detailed installation

### Prerequisites

| Tool | Why you need it | How to install |
|---|---|---|
| Python ≥ 3.9 | runs the app | [python.org](https://www.python.org/downloads/) |
| git | clone the repo | [git-scm.com](https://git-scm.com/downloads) |
| Tesseract | OCR for images / scanned PDFs (optional but recommended) | see below |
| Poppler | rasterise PDF pages for OCR (optional) | see below |

### Step 1 — Clone the repository

```bash
# Pick a folder where you want the project to live
cd ~/Documents          # or wherever you keep your projects

# Clone
git clone https://github.com/YOUR_USERNAME/rag-studio.git](https://github.com/Al-Moccardi/RAG_Demo_Lessons

# Enter the project
cd rag-studio
```

> If you don't have a GitHub account yet, you can also just download the ZIP from the project page (the green "Code" button → "Download ZIP") and unzip it.

### Step 2 — Create a virtual environment (strongly recommended)

A virtualenv keeps RAG Studio's dependencies isolated from the rest of your system Python.

```bash
# Create
python -m venv venv

# Activate it
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\Activate.ps1     # Windows (PowerShell)
# venv\Scripts\activate.bat       # Windows (cmd)
```

You should see `(venv)` at the start of your prompt.

### Step 3 — Install the Python dependencies

```bash
pip install -r requirements.txt
```

This pulls Streamlit, LangChain, FAISS, sentence-transformers, transformers, Plotly, scikit-learn, pypdf, pytesseract, etc. (≈ 1–3 minutes on a normal connection).

### Step 4 — Install Tesseract + Poppler (for OCR)

OCR lives **outside** Python and must be installed at the OS level. The app still runs without it — you just won't be able to upload images or scanned PDFs.

#### macOS (with Homebrew)
```bash
brew install tesseract poppler
```

#### Ubuntu / Debian / WSL
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

For non-English documents, also install the relevant language pack:
```bash
sudo apt install tesseract-ocr-ita     # Italian
sudo apt install tesseract-ocr-fra     # French
# etc.
```

#### Windows

The Tesseract installer does **not** add itself to PATH — but RAG Studio auto-detects the standard install locations:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe
```

1. Tesseract installer: <https://github.com/UB-Mannheim/tesseract/wiki>
2. Poppler binaries: <https://github.com/oschwartz10612/poppler-windows/releases>

If Tesseract ends up somewhere else, set this environment variable **before** launching Streamlit:

```powershell
$env:TESSERACT_CMD = "C:\your\path\to\tesseract.exe"
$env:POPPLER_PATH = "C:\poppler\Library\bin"        # only for OCR-ing PDFs
streamlit run app.py
```

### Step 5 — (Optional) Install RAGAS for advanced evaluation metrics

```bash
pip install ragas datasets
```

RAGAS provides LLM-as-judge metrics (faithfulness, answer relevancy, context precision). By default it calls OpenAI — set `OPENAI_API_KEY` if you want those metrics. Without it, the **lightweight metrics** in the Evaluation tab still work perfectly.

### Step 6 — Launch the app

```bash
streamlit run app.py
```

Your default browser opens at `http://localhost:8501`. If it doesn't, copy the URL Streamlit prints into your browser manually.

---

## 🎬 First run — your first question

When the page opens you'll see eight tabs at the top. Here's the happy path:

1. **Sidebar**: leave the defaults (Qwen 1.5B + MPNet) for the first run.
2. Click **📥 Download AI Articles** in the sidebar.
   *This downloads 10 Wikipedia articles about AI / ML into `data/`. Takes ≈ 15 seconds.*
3. Click **🔨 Build Pipeline**.
   *First time: ~60 s for model downloads + embedding. Watch the status box in the sidebar — it tells you exactly what's happening.*
4. When the sidebar shows ✅ **Pipeline active**, switch to the **💬 Chat** tab.
5. Type a question and hit Enter:
   - "What is deep learning?"
   - "Who created Python?"
   - "Explain transformers in two sentences."

You'll see the answer stream in token-by-token, followed by an expandable **"📚 5 sources retrieved"** showing which chunks the LLM actually used.

> 💡 **Restart any time.** Stop with `Ctrl+C`, relaunch with `streamlit run app.py` — the pipeline auto-loads from disk in ~2 seconds.

---

## 🗺️ Tour of the eight tabs

| Tab | What you do there |
|---|---|
| **🏠 Home** | Interactive architecture diagram — click any module to see its inputs / outputs / source code |
| **💬 Chat** | Ask questions, see streaming answers + cited chunks. Toggle streaming off for debugging. |
| **📊 Analytics** | Visualise the embedding space (2D / 3D, PCA / t-SNE). Type a query → see retrieved chunks highlighted. |
| **📚 Knowledge Base** | Drag-and-drop PDFs / TXTs / images. View / delete what's currently indexed. |
| **🎮 Playground** | Step through every pipeline stage in isolation. **5-way chunking comparison** is the star. |
| **🧪 Advanced** | Query Expansion · ReAct Agent · Memory Compression — the next level up |
| **📏 Evaluation** | Measure your RAG on a labelled test set with lightweight + RAGAS metrics |
| **🚀 GitHub Push** | Guided commit & push, with copy-paste **and** click-to-run buttons |

---

## 📁 Project structure

```
rag-studio/
├── app.py                       # Entry point — sidebar + 8-tab router
├── requirements.txt
├── README.md                    # This file
├── .gitignore
│
├── backend/                     # The RAG pipeline (and only the pipeline)
│   ├── config.py                # @dataclass RAGConfig + EMBEDDING_MODELS + LLM_MODELS
│   ├── data_loader.py           # Wikipedia API + PDF/TXT/image upload + delete
│   ├── chunker.py               # RecursiveCharacterTextSplitter (the default)
│   ├── embedder.py              # any HF sentence-transformer → 768-d vectors
│   ├── vector_store.py          # FAISS build + search + save / load
│   ├── llm.py                   # any HF causal LM (Qwen / SmolLM / Phi-3 / …)
│   └── rag_chain.py             # RAGPipeline — chat-template aware + streaming
│
├── utilities/                   # External helpers (not part of the core pipeline)
│   ├── analytics.py             # PCA / t-SNE 2D & 3D + Plotly + topic detection
│   ├── ocr.py                   # Tesseract OCR — images + scanned PDFs
│   ├── pipeline_cache.py        # Persist & restore embeddings + FAISS index
│   ├── chunking_strategies.py   # 5 chunkers: char / recursive / token / sentence / semantic
│   └── evaluation.py            # Lightweight metrics + RAGAS integration
│
├── advanced/                    # Optional RAG enhancements
│   ├── query_expansion.py       # LLM-rewrites query → multi-query retrieval
│   ├── react_agent.py           # ReAct loop (Thought → Action → Observation)
│   └── memory_compression.py    # Summarize old turns to keep prompts bounded
│
├── frontend/                    # Streamlit UI
│   ├── landing.py               # Home tab — embeds the interactive architecture
│   ├── chat_ui.py               # Chat tab — streaming + source cards
│   ├── analytics_ui.py          # Analytics tab — 2D / 3D embedding explorer
│   ├── data_management_ui.py    # Knowledge Base tab
│   ├── playground_ui.py         # Playground tab — 8 interactive stages
│   ├── advanced_ui.py           # Advanced tab — 3 advanced techniques
│   ├── evaluation_ui.py         # Evaluation tab — metrics + per-case detail
│   ├── github_push_ui.py        # GitHub Push tab — commands + buttons
│   ├── settings_ui.py           # Sidebar — model picker + hyperparameter sliders
│   └── styles.py                # Custom CSS
│
├── docs/
│   └── architecture.html        # Interactive diagram embedded in the Home tab
│
├── data/                        # 🚫 git-ignored — created on first run
└── assets/                      # 🚫 git-ignored — FAISS index + embeddings cache
```

---

## ⚙️ Configuration cheat sheet

Everything lives in `backend/config.py` as a `@dataclass`. The sidebar exposes the most useful knobs.

### Chunking
- **`chunk_size`** (default 500) — max characters per chunk
- **`chunk_overlap`** (50) — characters shared between adjacent chunks
- **`min_chunk_length`** (30) — discard chunks shorter than this (in words)

### Retrieval
- **`k`** (5) — number of chunks to feed the LLM per question. More chunks = more context but also more noise + tokens.

### Generation
- **`temperature`** (0.3) — 0.1 = factual, 1.0 = creative, 1.5 = chaotic
- **`max_new_tokens`** (300) — max answer length
- **`top_p`** (0.9), **`repetition_penalty`** (1.15)

### System prompt
The instructions sent to the LLM before every question. Editable in the sidebar.

---

## 🤖 Available models

All models run **locally** — no API keys required.

### Embedding (text → vectors)

| Model | Dim | Size | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 80 MB | Fastest, best for limited RAM |
| `all-mpnet-base-v2` **(default)** | 768 | 420 MB | Balanced general-purpose English |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | Compact + great retrieval scores |
| `BAAI/bge-base-en-v1.5` | 768 | 440 MB | Top of the MTEB leaderboard for its size |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 1.1 GB | Use for non-English documents |

### LLM (generation)

| Model | Params | Size | Notes |
|---|---|---|---|
| `Qwen2.5-0.5B-Instruct` | 0.5 B | ~1 GB | Fastest, but less coherent |
| `Qwen2.5-1.5B-Instruct` **(default)** | 1.5 B | ~3 GB | Good speed / quality tradeoff |
| `Qwen2.5-3B-Instruct` | 3 B | ~6 GB | Noticeably better, needs ~10 GB RAM |
| `SmolLM2-1.7B-Instruct` | 1.7 B | ~3.4 GB | HuggingFace alternative |
| `Phi-3-mini-4k-instruct` | 3.8 B | ~7 GB | Strong reasoning for its size |

> Switching models requires clicking **🔨 Build Pipeline** again — the embedding cache is invalidated automatically when you change the embedding model.

---

## 🆘 Troubleshooting

### `ModuleNotFoundError` for `langchain_core` / `streamlit` / etc.
You forgot to install the requirements or you're not in the right virtualenv.
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### `Tesseract is not installed or it's not in your PATH`
1. Install Tesseract at the OS level (see the [Detailed installation](#-detailed-installation) section)
2. On Windows the installer doesn't add to PATH automatically — RAG Studio auto-probes a few standard locations
3. If yours is in a non-standard folder, set `TESSERACT_CMD` env var **before** starting Streamlit

### `403 Forbidden` when downloading Wikipedia articles
Already handled — RAG Studio sends a proper User-Agent header. If you still see it, your network may be blocking Wikipedia. Try a different connection.

### The LLM echoes the context instead of answering
Make sure you're using a recent version (v6+). The fix is in `backend/rag_chain.py` — it calls `tokenizer.apply_chat_template()` so the model sees its expected role tokens.

### Streaming hangs forever
Some LLMs / transformers versions have flaky streaming on Windows. Toggle **⚡ Stream** off in the Chat tab to use the blocking path instead.

### Pipeline auto-load skipped despite cache being present
The cache is invalidated when you change:
- The embedding model name
- `chunk_size`, `chunk_overlap`, `min_chunk_length`
- The set of files in `data/` (any add / remove / modify)

This is intentional — vectors built with one model are useless for another.

### Out of memory on the LLM
Switch to a smaller model (Qwen 0.5B or 1.5B) from the sidebar, then **🔨 Build Pipeline** again.

### Models download super slowly
HuggingFace's CDN occasionally throttles. Try:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1   # faster transfer protocol
pip install hf_transfer
```

---

## 📤 Pushing your fork to GitHub

The easiest way is the built-in **🚀 GitHub Push** tab — it gives you both copy-paste commands and one-click buttons.

If you'd rather use the terminal directly:

```bash
# 1. Create an empty repo on github.com (the green "New" button)

# 2. From your local rag-studio folder:
git init
git branch -M main
git add .
git commit -m "Initial commit: RAG Studio"

# 3. Connect it to your remote
git remote add origin https://github.com/YOUR_USERNAME/rag-studio.git

# 4. Push
git push -u origin main
```

The `.gitignore` shipped with the project already excludes:
- `venv/`, `__pycache__/`, `*.pyc`
- `data/` (downloaded articles, can be re-fetched)
- `assets/` (FAISS index + embeddings cache, can be re-built)
- `.env`, IDE files, etc.

So your repository stays slim (≈ 100 KB) — the heavy stuff is regenerated on first run.

---

## 📜 License & credits

Educational project — feel free to fork, modify, and use in your own courses.

Built with:
- 🦜 [LangChain](https://github.com/langchain-ai/langchain)
- 📐 [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- 🤗 [HuggingFace Transformers](https://github.com/huggingface/transformers) + [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
- 🎈 [Streamlit](https://streamlit.io/)
- 👁️ [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- 📊 [Plotly](https://plotly.com/python/) + [scikit-learn](https://scikit-learn.org/)
- 📏 [RAGAS](https://github.com/explodinggradients/ragas) (optional)

Default models:
- Embedding: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- LLM: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

---

<p align="center">
  Made with 🧠 for teaching — Prof.ssa Flora Amato · DIETI · UniNa Federico II
</p>
