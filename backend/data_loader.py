"""
backend/data_loader.py — Document Acquisition
===============================================
This module is responsible for getting RAW TEXT into the system.
There are two complementary sources:

  1. WIKIPEDIA  — Download articles via the official REST API
                  (no scraping, no HTML parsing needed)
  2. UPLOAD     — Drag-and-drop PDF / TXT files in the Streamlit UI

Whichever source is used, the files end up as plain text on disk
inside data/, and `load_documents()` turns them into LangChain
Document objects with .page_content + .metadata. From that point on,
every downstream module (chunker, embedder, ...) treats them
identically — that's the power of having one canonical Document type.

WHY WIKIPEDIA AS DEFAULT?
  - Free, legally distributable, well-structured prose.
  - Easy to verify the LLM's answer against the source.
  - Wide coverage of AI/ML topics → a meaningful demo.
"""

# ── Standard library ──────────────────────────────────────
import requests                  # HTTP client for Wikipedia REST calls
import shutil                    # not strictly used here but kept for future
from pathlib import Path         # path manipulation
from typing import List, Optional, BinaryIO  # type hints for clarity

# ── LangChain imports ─────────────────────────────────────
# TextLoader      — reads .txt files into Document objects
# DirectoryLoader — applies a loader recursively over a folder
# PyPDFLoader     — uses pypdf under the hood, yields one Document per page
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
# Document is a tiny class with `.page_content` (str) and `.metadata` (dict)
from langchain_core.documents import Document

# ── Project imports ───────────────────────────────────────
from backend.config import DATA_DIR, WIKI_TOPICS

# ── OCR fallback (utilities/ocr.py) ───────────────────────
# Lazy import inside functions to keep startup fast — Tesseract isn't
# always installed on every system.


# ────────────────────────────────────────────────────────────
# Wikipedia API requirements
# ────────────────────────────────────────────────────────────
# Since 2024, Wikipedia REQUIRES a User-Agent header on every request.
# Without it the API returns 403 Forbidden. The UA must identify the
# tool, version and (ideally) a contact for abuse reports.
WIKIPEDIA_HEADERS = {
    "User-Agent": "RAGStudio/1.0 (Educational Project; Python/requests)",
}


# ════════════════════════════════════════════════════════════
# 1. WIKIPEDIA DOWNLOAD
# ════════════════════════════════════════════════════════════

def download_wikipedia_article(topic: str, save_dir: Path = DATA_DIR) -> str:
    """
    Download a single Wikipedia article and save it as plain text.

    Uses the MediaWiki "extracts" API endpoint. The crucial parameter
    is `explaintext=True`, which tells Wikipedia to strip all HTML and
    return clean prose — much easier to chunk than raw wiki markup.

    Args:
        topic    : Article title with underscores (e.g. "Deep_learning")
        save_dir : Where to write the .txt file (default: data/)

    Returns:
        Path to the saved file as a string, or "" if download failed.
    """
    # Where the .txt for this topic will live.
    filepath = save_dir / f"{topic}.txt"

    # Skip re-downloading if we already have a non-trivial copy.
    # The 100-byte threshold filters out empty / error pages.
    if filepath.exists() and filepath.stat().st_size > 100:
        return str(filepath)

    # Wikipedia's MediaWiki action API endpoint.
    url = "https://en.wikipedia.org/w/api.php"

    # Build the query parameters. Wikipedia's API takes them in the URL.
    params = {
        "action": "query",                       # we're querying
        "titles": topic.replace("_", " "),       # human-readable title
        "prop": "extracts",                      # we want the article text
        "explaintext": True,                     # → plain text, no HTML
        "exlimit": 1,                            # only one article
        "format": "json",                        # JSON response, not XML
    }

    try:
        # GET request with our identifying User-Agent.
        resp = requests.get(url, params=params, headers=WIKIPEDIA_HEADERS, timeout=15)
        resp.raise_for_status()  # raises if status code is 4xx/5xx
        data = resp.json()       # parse JSON body

        # The response shape is: {"query": {"pages": {"<page_id>": {...}}}}
        pages = data.get("query", {}).get("pages", {})

        # Iterate (there's normally only one page, but the structure is a dict).
        for page_id, page_data in pages.items():
            text = page_data.get("extract", "")

            # Sanity check: skip if the article is empty or extremely short.
            if text and len(text) > 200:
                # Add a markdown-style header so the topic name is visible
                # in the chunk text — useful for the RAG citations later.
                full_text = f"# {topic.replace('_', ' ')}\n\n{text}"
                filepath.write_text(full_text, encoding="utf-8")
                return str(filepath)

    except Exception as e:
        # Network errors, timeouts, JSON decoding errors etc.
        # We DON'T crash the app: just log and return empty string.
        print(f"  ⚠️ Failed to download '{topic}': {e}")

    return ""


def download_all_articles(
    topics: List[str] = WIKI_TOPICS,
    progress_callback=None,
) -> List[str]:
    """
    Download every article in WIKI_TOPICS (or a custom list).

    Args:
        topics            : List of article titles (default: WIKI_TOPICS)
        progress_callback : Optional function(current, total, topic).
                            The Streamlit sidebar uses this to update a
                            progress bar in real time.

    Returns:
        List of file paths actually downloaded (may be < len(topics)
        if some failed).
    """
    paths = []
    for i, topic in enumerate(topics):
        # Update the UI before starting each download.
        if progress_callback:
            progress_callback(i, len(topics), topic)

        path = download_wikipedia_article(topic)
        if path:
            paths.append(path)

    return paths


# ════════════════════════════════════════════════════════════
# 2. FILE UPLOAD (drag-and-drop)
# ════════════════════════════════════════════════════════════

# Whitelist of file types we know how to load. Anything else is rejected
# at upload time so we never have to handle weird formats downstream.
#
#   .pdf / .txt → loaded directly (PyPDFLoader / TextLoader)
#   .jpg / .png / .tiff / .bmp → routed through the OCR pipeline
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def save_uploaded_file(uploaded_file, save_dir: Path = DATA_DIR) -> Optional[Path]:
    """
    Persist a Streamlit-uploaded file (PDF or TXT) to data/.

    Streamlit's `UploadedFile` is an in-memory file-like object. We
    write its bytes to disk so the rest of the pipeline (which reads
    from data/) can pick it up.

    Args:
        uploaded_file : a `streamlit.runtime.uploaded_file_manager.UploadedFile`
        save_dir      : destination directory (default: data/)

    Returns:
        Path object of the saved file, or None if the extension is
        not allowed.
    """
    name = uploaded_file.name
    ext = Path(name).suffix.lower()

    # Reject unknown extensions — DON'T try to load .doc, .epub, .html, etc.
    if ext not in ALLOWED_EXTENSIONS:
        return None

    target = save_dir / name
    # `getbuffer()` gives us bytes-like access to the in-memory file.
    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


def list_documents(directory: Path = DATA_DIR) -> List[dict]:
    """
    Scan data/ and return one metadata dict per file.

    Used by the "Knowledge Base" tab to display what's currently
    indexed. Returns one row of info per file; the UI then renders
    each row as a card with a delete button.

    Returns:
        A list of dicts: {name, path, size_kb, ext, source}.
        `source` is a heuristic: .txt → "Wikipedia", .pdf → "Upload".
    """
    if not directory.exists():
        return []

    items = []
    for f in sorted(directory.iterdir()):
        # Skip subdirectories, dotfiles, etc.
        if not f.is_file():
            continue

        ext = f.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        size_kb = f.stat().st_size / 1024

        # Best-effort guess at where the file came from / how it'll be loaded.
        if ext == ".txt":
            source = "Wikipedia"
        elif ext == ".pdf":
            source = "PDF"
        elif ext in IMAGE_EXTENSIONS:
            source = "Image (OCR)"
        else:
            source = "Upload"

        items.append({
            "name": f.name,
            "path": str(f),
            "size_kb": size_kb,
            "ext": ext,
            "source": source,
        })
    return items


def delete_document(filename: str, directory: Path = DATA_DIR) -> bool:
    """
    Delete a single file from data/.

    We accept just the file NAME (not a full path) for safety —
    a user can never accidentally delete something outside data/.

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    target = directory / filename
    if target.exists() and target.is_file():
        target.unlink()
        return True
    return False


def clear_all_documents(directory: Path = DATA_DIR) -> int:
    """
    Wipe every supported file in data/. Used by the "Delete ALL" button.

    Returns:
        Number of files actually removed.
    """
    count = 0
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            f.unlink()
            count += 1
    return count


# ════════════════════════════════════════════════════════════
# 3. LOAD documents (TXT + PDF) into LangChain Documents
# ════════════════════════════════════════════════════════════

def load_documents(directory: Path = DATA_DIR) -> List[Document]:
    """
    Load every supported file in data/ as LangChain Documents.

    Three input types are handled:

      .txt → TextLoader. One Document per file.

      .pdf → PyPDFLoader, with OCR fallback. We first sniff the PDF: if
             the embedded text layer looks empty (i.e. it's a scanned
             document), we route it through Tesseract via utilities.ocr.
             For born-digital PDFs the fast PyPDFLoader path is used.

      .jpg / .png / ... → utilities.ocr.ocr_image_file. One Document
             per image, page=1.

    Each resulting Document has:
        .page_content : str  — the text extracted (or OCR'd) from the file
        .metadata     : dict — at minimum {"source": "<filename>"}
                               PDFs add {"page": <int>}
                               OCR'd files add {"ocr": True, "ocr_conf": <float>}

    A single corrupt file never crashes the whole indexing run — every
    file is loaded inside its own try/except.
    """
    docs: List[Document] = []
    if not directory.exists():
        return docs

    # ─── 1. Plain-text files ─────────────────────────────
    for txt_path in sorted(directory.glob("**/*.txt")):
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"  ⚠️ Failed to load TXT {txt_path.name}: {e}")

    # ─── 2. PDF files (PyPDF first, OCR fallback) ───────
    # Lazy import OCR so users without Tesseract can still use the app
    # for born-digital PDFs and TXTs. We also check the actual binary
    # is callable — pytesseract imports fine even without the binary,
    # so a try/import alone isn't enough.
    ocr_available = False
    try:
        from utilities.ocr import (
            pdf_needs_ocr, ocr_pdf, tesseract_available,
        )
        is_ok, msg = tesseract_available()
        if is_ok:
            ocr_available = True
        else:
            print(f"  ℹ️ OCR disabled: {msg.splitlines()[0]}")
    except ImportError as e:
        print(f"  ℹ️ OCR libraries not installed ({e}); scanned PDFs/images will be skipped.")

    for pdf_path in sorted(directory.glob("**/*.pdf")):
        try:
            # Sniff: does this PDF have a usable text layer?
            if ocr_available and pdf_needs_ocr(pdf_path):
                # Scanned PDF — route to OCR.
                print(f"  🔎 OCR-ing scanned PDF: {pdf_path.name}")
                pages = ocr_pdf(pdf_path)
                for page_idx, text, conf in pages:
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "page": page_idx,
                                "ocr": True,
                                "ocr_conf": conf,
                            },
                        ))
            else:
                # Born-digital PDF — fast PyPDF path.
                loader = PyPDFLoader(str(pdf_path))
                docs.extend(loader.load())
        except Exception as e:
            print(f"  ⚠️ Failed to load PDF {pdf_path.name}: {e}")

    # ─── 3. Images (always OCR) ─────────────────────────
    if ocr_available:
        try:
            from utilities.ocr import ocr_image_file
        except ImportError:
            ocr_image_file = None

        if ocr_image_file is not None:
            for ext in IMAGE_EXTENSIONS:
                for img_path in sorted(directory.glob(f"**/*{ext}")):
                    try:
                        text, conf = ocr_image_file(img_path)
                        if text.strip():
                            docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source": str(img_path),
                                    "page": 1,
                                    "ocr": True,
                                    "ocr_conf": conf,
                                },
                            ))
                    except Exception as e:
                        print(f"  ⚠️ Failed to OCR {img_path.name}: {e}")

    return docs
