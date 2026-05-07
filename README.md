# 🤖 RAG Studio

**Retrieval-Augmented Generation** — powered by LangChain, FAISS & HuggingFace

> Prof.ssa Flora Amato — DIETI, Università degli Studi di Napoli Federico II

A complete, fully-local RAG system designed for teaching. Students can swap models,
upload their own PDFs, and visualize the embedding space — all in the browser.

## ✨ Features

- 🏠 **Interactive architecture diagram** — clickable modules in the home tab
- 💬 **Chat with sources** — every answer cites the chunks it used + similarity scores
- 📊 **3D / 2D embedding-space explorer** — PCA/t-SNE with query overlay; retrieved chunks shown in full below the plot
- 📚 **Knowledge-base management** — drag-and-drop PDFs/TXTs/images, delete files
- 🔎 **OCR (Tesseract)** — images and scanned PDFs are auto-routed through OCR; the Playground includes a step-by-step OCR walkthrough with bounding-box visualization
- 🎮 **Pipeline Playground** — step through OCR · Chunking · Embedding · Retrieval · Prompt Builder · Full Input · Full Trace · Persistence, each calling the real backend functions
- 🧪 **Advanced tab** — Query Expansion · ReAct Agent · Memory Compression, each implemented in `advanced/` and visualised step by step
- 🚀 **GitHub Push tab** — guided git init → commit → push with both copy-paste commands and one-click buttons
- 🤖 **Model picker** — choose between 5 embedding models and 5 LLMs from the sidebar
- ⚙️ **Live tuning** — chunk size, k, temperature, system prompt, all editable
- 💾 **Persistent pipeline cache** — embeddings + FAISS index are saved to `assets/` after each build; subsequent restarts auto-load the cache in ~2 seconds. Cache is invalidated automatically if you change the embedding model, chunking parameters, or the document set.

## Quick Start

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

#### Python packages

```bash
pip install -r requirements.txt
```

#### System packages for OCR

OCR (Tesseract + Poppler) lives outside Python and must be installed at the OS level:

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr poppler-utils

# macOS (with Homebrew)
brew install tesseract poppler

# Windows
# Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
```

##### 🪟 Windows users — important

The Tesseract installer for Windows does **not** add itself to your PATH automatically.
The app tries to auto-detect the install in these locations:

```
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe
```

If you installed Tesseract somewhere else, set the `TESSERACT_CMD` environment
variable to the full path of `tesseract.exe` before launching Streamlit:

```powershell
# PowerShell
$env:TESSERACT_CMD = "C:\path\to\tesseract.exe"
streamlit run app.py
```

The same applies to Poppler — set `POPPLER_PATH` to the folder containing `pdftoppm.exe`
if it isn't on PATH (only matters if you want to OCR scanned PDFs; image OCR works without Poppler).

Without these, the app still runs perfectly for plain TXT and born-digital PDFs — you just won't be able to upload images or scanned PDFs.

> **Note**: First run downloads the chosen embedding + LLM models from HuggingFace
> (defaults: ~420 MB embedding + ~3 GB LLM). They are cached afterwards.

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Workflow

1. Pick an **embedding model** and **LLM** in the sidebar (defaults are fine).
2. Either click **📥 Download AI Articles** to get 10 Wikipedia articles,
   **or** open the **📚 Knowledge Base** tab and drag-and-drop your own PDFs/TXTs.
3. Click **🔨 Build Pipeline** to process everything.
4. Switch to the **💬 Chat** tab and ask questions.
5. Explore the **📊 Analytics** tab to see your documents in the 768-d embedding space.

## Available Models

### Embedding (text → vectors)

| Model | Dim | Size | Best for |
|-------|-----|------|----------|
| MiniLM-L6 | 384 | 80 MB | Fast experiments |
| MPNet-base **(default)** | 768 | 420 MB | Balanced general use |
| BGE-small | 384 | 130 MB | High quality, small |
| BGE-base | 768 | 440 MB | Top retrieval scores |
| Multilingual MPNet | 768 | 1.1 GB | Non-English documents |

### LLM (answer generation)

| Model | Params | Size | Notes |
|-------|--------|------|-------|
| Qwen 0.5B | 0.5B | ~1 GB | Fastest, low RAM |
| Qwen 1.5B **(default)** | 1.5B | ~3 GB | Recommended starting point |
| Qwen 3B | 3B | ~6 GB | Higher quality |
| SmolLM2 1.7B | 1.7B | ~3.4 GB | HuggingFace alternative |
| Phi-3 mini | 3.8B | ~7 GB | Strong reasoning |

## Project Structure

```
rag_app/
├── app.py                       # Entry point + sidebar + 4-tab router
├── requirements.txt
├── README.md
├── .gitignore
│
├── backend/                     # Main RAG pipeline (and only the pipeline)
│   ├── config.py                # @dataclass + EMBEDDING_MODELS + LLM_MODELS
│   ├── data_loader.py           # Wikipedia API + PDF/TXT upload + delete
│   ├── chunker.py               # RecursiveCharacterTextSplitter
│   ├── embedder.py              # any HF sentence-transformer → vectors
│   ├── vector_store.py          # FAISS index build + search + save/load
│   ├── llm.py                   # any HF causal LM (Qwen/SmolLM/Phi-3/...)
│   └── rag_chain.py             # RAGPipeline.ask() — chat-template aware
│
├── utilities/                   # External to pipeline (analytics, helpers)
│   ├── analytics.py             # PCA/t-SNE 2D & 3D + Plotly + topic detection
│   ├── ocr.py                   # Tesseract OCR — images + scanned PDFs
│   └── pipeline_cache.py        # Persist & restore embeddings + FAISS index
│
├── advanced/                    # Optional RAG enhancements
│   ├── query_expansion.py       # LLM-rewrites query → multi-query retrieval
│   ├── react_agent.py           # ReAct loop (Thought→Action→Observation)
│   └── memory_compression.py    # Summarize old turns to keep prompts bounded
│
├── frontend/                    # Streamlit UI
│   ├── landing.py               # Home tab — embeds docs/architecture.html
│   ├── chat_ui.py               # Chat tab — messages + source cards
│   ├── analytics_ui.py          # Analytics tab — 2D/3D embedding explorer
│   ├── data_management_ui.py    # Knowledge Base tab — upload + delete
│   ├── playground_ui.py         # Playground tab — 8 interactive stages
│   ├── advanced_ui.py           # Advanced tab — query expansion, ReAct, memory
│   ├── github_push_ui.py        # GitHub Push tab — copy commands + buttons
│   ├── settings_ui.py           # Sidebar — model + chunking + generation
│   └── styles.py                # Custom CSS
│
├── docs/
│   └── architecture.html        # Interactive architecture diagram
│
├── data/                        # Auto-created — PDFs/TXTs of the KB
└── assets/                      # Auto-created — FAISS index cache
```

## Pipeline Flow

```
PDF / TXT / Wikipedia → Load → Chunk → Embed (e.g. 768d) → FAISS index
                                                                  ↓
User question → Embed query → Search FAISS → Top-K chunks → LLM → Answer
```

## Requirements

- Python 3.9+
- 4–8 GB RAM (depends on chosen LLM)
- GPU optional — CUDA is auto-detected
- Internet (only for first-time model download + Wikipedia)

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG Studio"
git remote add origin https://github.com/YOUR_USERNAME/rag-studio.git
git branch -M main
git push -u origin main
```

The included `.gitignore` already excludes `data/`, `assets/`, `venv/` and `__pycache__/`.
#   R A G _ D e m o _ L e s s o n s  
 #   R A G _ D e m o _ L e s s o n s  
 