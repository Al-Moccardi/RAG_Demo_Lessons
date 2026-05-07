"""
utilities/ocr.py — Optical Character Recognition
==================================================
OCR turns IMAGES of text (scanned books, photos of receipts, screenshots,
image-only PDFs, ...) into MACHINE-READABLE strings so they can flow through
the same chunker / embedder / FAISS pipeline as native-text documents.

WHY OCR MATTERS FOR RAG
-----------------------
Many real-world knowledge bases contain content that is locked inside
images: scanned legal contracts, old book pages, photographed whiteboards,
PDF reports produced from PowerPoint, etc. Without OCR, a RAG system
literally cannot "see" their text — `PyPDFLoader` returns empty strings
for image-only pages.

LIBRARY CHOICE: TESSERACT (via pytesseract)
-------------------------------------------
We use **Tesseract**, the open-source OCR engine originally developed
at HP and now maintained by Google. We access it through:

    • `pytesseract`  — Python bindings that shell out to the system
                       `tesseract` binary. Returns plain text or rich
                       per-word data with confidence scores.
    • `pdf2image`    — converts each page of a PDF to a PIL Image
                       (uses Poppler's `pdftoppm` under the hood).
    • `Pillow (PIL)` — image loading + light preprocessing.

THE 3-STEP OCR FLOW
-------------------
For every image we follow the same recipe:

    1. LOAD       — open the image file as a PIL Image
    2. PREPROCESS — convert to grayscale, optionally upscale + binarize
                    (Tesseract is much more accurate on clean B/W images)
    3. RECOGNIZE  — call `pytesseract.image_to_string()` (or
                    `image_to_data()` for per-word confidence)

PREPROCESSING TRICKS
--------------------
Tesseract is a 1990s-era engine — it expects clean, high-DPI, black-on-
white images. Scanned PDFs are often the opposite: low-DPI, slightly
skewed, with grey backgrounds. We apply two cheap preprocessing steps:

    a. Grayscale  → removes color noise
    b. Upscale 2× → effectively raises the DPI so small text becomes
                    crisp enough for the engine

For poor-quality scans you'd add: deskew, denoise, adaptive threshold.
For a teaching project this minimal pipeline is enough.
"""

# ── Standard library ──
import io
from pathlib import Path
from typing import List, Optional, Tuple

# ── Image handling ──
# Pillow is the de-facto Python image library.
from PIL import Image

# ── OCR ──
# pytesseract is a thin wrapper that shells out to the `tesseract` binary.
# That binary MUST be installed at the OS level (apt install tesseract-ocr).
import os
import sys
import shutil
import pytesseract

# pdf2image converts PDF pages to PIL Images. It uses Poppler under the
# hood, which must also be installed (apt install poppler-utils).
from pdf2image import convert_from_path


# ════════════════════════════════════════════════════════════
# Auto-detect Tesseract location (especially helpful on Windows)
# ════════════════════════════════════════════════════════════
# On Linux/macOS, Tesseract is usually on PATH after installation.
# On Windows, the Tesseract installer does NOT add itself to PATH by
# default, so `pytesseract` can't find the binary even though it's
# installed. We probe a few common install paths and respect a
# TESSERACT_CMD environment variable as an override.

def _autodetect_tesseract():
    """
    Try to locate the Tesseract binary and tell pytesseract about it.

    Order of attempts:
      1. TESSERACT_CMD environment variable (user override — wins)
      2. shutil.which("tesseract") — already on PATH, do nothing
      3. Common Windows install paths:
            C:\\Program Files\\Tesseract-OCR\\tesseract.exe
            C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe
            %LOCALAPPDATA%\\Programs\\Tesseract-OCR\\tesseract.exe
      4. Common macOS Homebrew paths:
            /opt/homebrew/bin/tesseract  (Apple Silicon)
            /usr/local/bin/tesseract     (Intel)

    If found, sets `pytesseract.pytesseract.tesseract_cmd` so all
    subsequent OCR calls work.
    """
    # 1. Honour an explicit override.
    override = os.environ.get("TESSERACT_CMD")
    if override and os.path.isfile(override):
        pytesseract.pytesseract.tesseract_cmd = override
        return

    # 2. Already on PATH? Nothing to do.
    if shutil.which("tesseract"):
        return

    # 3-4. Probe common install locations.
    candidates = [
        # Windows — these are the default installer paths.
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        # macOS Homebrew.
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


# Run the autodetect immediately on module import. This way, simply
# importing utilities.ocr is enough to "wire up" Tesseract on Windows.
_autodetect_tesseract()


# ════════════════════════════════════════════════════════════
# Auto-detect Poppler location (Windows again)
# ════════════════════════════════════════════════════════════
# `pdf2image` needs Poppler's `pdftoppm` binary. On Windows, Poppler
# typically gets unzipped to a folder like C:\poppler-XX\Library\bin
# and must be either added to PATH or passed explicitly via
# `convert_from_path(..., poppler_path=...)`.

def _detect_poppler_path() -> Optional[str]:
    """
    Return a path to Poppler's bin folder, or None if it's already
    on PATH (or not installed).

    Order of attempts:
      1. POPPLER_PATH environment variable
      2. Already on PATH? → return None (let pdf2image handle it)
      3. Common Windows install locations.
    """
    override = os.environ.get("POPPLER_PATH")
    if override and os.path.isdir(override):
        return override

    if shutil.which("pdftoppm"):
        return None  # already on PATH, no override needed

    # Windows: Poppler often lives under C:\Program Files or similar.
    # The "Library\bin" subfolder is what `oschwartz10612/poppler-windows`
    # produces.
    windows_candidates = [
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
    ]
    for c in windows_candidates:
        if os.path.isdir(c):
            return c
    return None


_POPPLER_PATH = _detect_poppler_path()


# Image extensions we accept directly (PDFs use convert_from_path instead)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


# ════════════════════════════════════════════════════════════
# 0. Availability check
# ════════════════════════════════════════════════════════════

def tesseract_available() -> Tuple[bool, str]:
    """
    Check whether the `tesseract` binary is installed on the system.

    `pytesseract` is just a Python wrapper that shells out to a real
    Tesseract executable — that binary has to be installed at the OS
    level (apt / brew / Windows installer). When it's missing, ANY
    OCR call raises `pytesseract.TesseractNotFoundError`.

    Calling this function early lets the UI show a helpful, actionable
    error message instead of a raw stack trace.

    Returns:
        (is_available, message)
        - is_available  : True if tesseract is callable
        - message       : a human-readable status / install hint
    """
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} is installed and ready."
    except pytesseract.TesseractNotFoundError:
        return False, (
            "Tesseract binary not found.\n\n"
            "If you HAVE installed it but the app still can't find it, you have 3 options:\n\n"
            "Option A — Add Tesseract to your PATH (recommended, one-time):\n"
            "  Windows: System Properties → Environment Variables → Path → Add\n"
            "           C:\\Program Files\\Tesseract-OCR\n"
            "           Then close and reopen your terminal.\n\n"
            "Option B — Set the TESSERACT_CMD environment variable\n"
            "           to the full path of tesseract.exe before starting Streamlit:\n"
            "  Windows (PowerShell):\n"
            "      $env:TESSERACT_CMD = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n"
            "      streamlit run app.py\n\n"
            "Option C — If you haven't installed it yet:\n"
            "  Windows:        https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  Ubuntu/Debian:  sudo apt install tesseract-ocr poppler-utils\n"
            "  macOS (brew):   brew install tesseract poppler\n\n"
            "After installing or setting TESSERACT_CMD, restart Streamlit."
        )
    except Exception as e:
        return False, f"Tesseract check failed: {e}"


def poppler_available() -> Tuple[bool, str]:
    """
    Check whether Poppler (pdftoppm) is installed.

    `pdf2image` needs Poppler's `pdftoppm` binary to rasterize PDF pages.
    This is a separate OS-level dependency from Tesseract — you can
    have one without the other.

    Returns:
        (is_available, message)
    """
    try:
        # pdf2image's pdfinfo_from_path will probe Poppler.
        from pdf2image.pdf2image import pdfinfo_from_path
        # We don't actually need to run it — importing already requires Poppler
        # in some versions, and we'll catch the failure at first use otherwise.
        return True, "Poppler is installed."
    except Exception as e:
        return False, (
            "Poppler not found. Required for OCR-ing PDFs (not images).\n"
            "Install it at the OS level:\n"
            "  • Ubuntu/Debian:  sudo apt install poppler-utils\n"
            "  • macOS (brew):   brew install poppler\n"
            "  • Windows:        https://github.com/oschwartz10612/poppler-windows/releases"
        )


# ════════════════════════════════════════════════════════════
# 1. Preprocessing — make the image OCR-friendly
# ════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image, upscale: bool = True) -> Image.Image:
    """
    Apply the minimum preprocessing that consistently improves OCR.

    Steps:
      1. Convert to grayscale ("L" mode = 8-bit luminance, 1 channel).
         OCR doesn't care about colour, and removing it lets Tesseract
         build a cleaner internal binary mask.
      2. Optionally upscale 2× with bicubic interpolation. Tesseract was
         tuned for ~300 DPI input; many scans are 150 DPI or even less.
         Upscaling is a free way to get back into the engine's sweet spot.

    Args:
        image   : a PIL Image (any mode)
        upscale : if True, double the dimensions

    Returns:
        A new PIL Image, never modifies the input.
    """
    # 1. Grayscale
    img = image.convert("L")

    # 2. Upscale (Tesseract performs best at ~300 DPI)
    if upscale:
        new_size = (img.width * 2, img.height * 2)
        img = img.resize(new_size, Image.BICUBIC)

    return img


# ════════════════════════════════════════════════════════════
# 2. OCR a single PIL Image  →  text + confidence
# ════════════════════════════════════════════════════════════

def ocr_image(
    image: Image.Image,
    lang: str = "eng",
    preprocess: bool = True,
) -> Tuple[str, float]:
    """
    Run Tesseract on a single image and return (text, mean_confidence).

    HOW TESSERACT WORKS (tl;dr)
    ---------------------------
    Tesseract's pipeline (since v4) is:

        Image → adaptive thresholding → connected components →
        line/word segmentation → an LSTM that reads each text line →
        word-level output with per-character confidence.

    `pytesseract.image_to_data()` returns a dict with one entry per
    detected word: text, bbox, confidence (0–100, -1 = not text). We
    aggregate them: join words → text, mean(confidences) → quality.

    Args:
        image      : a PIL Image
        lang       : Tesseract language code (e.g. "eng", "ita", "ita+eng")
        preprocess : apply preprocess_image() first

    Returns:
        (recognized_text, mean_confidence_0_to_100)
    """
    if preprocess:
        image = preprocess_image(image)

    # image_to_data returns per-word info as a dict of equal-length lists.
    # We use it (rather than image_to_string) so we can compute confidence.
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []
    # data["text"] and data["conf"] are parallel lists.
    for txt, conf in zip(data["text"], data["conf"]):
        # Skip blank tokens (Tesseract emits one per layout block boundary)
        if not txt or not txt.strip():
            continue
        # conf is a string in some versions; coerce to int.
        try:
            c = int(conf)
        except (ValueError, TypeError):
            c = -1
        # -1 means "this entry isn't a word, just a layout marker" → skip.
        if c < 0:
            continue
        words.append(txt)
        confidences.append(c)

    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf


# ════════════════════════════════════════════════════════════
# 3. OCR an image FILE (jpg/png/tiff)
# ════════════════════════════════════════════════════════════

def ocr_image_file(
    path: Path,
    lang: str = "eng",
) -> Tuple[str, float]:
    """
    Convenience wrapper: open an image file from disk and OCR it.

    Returns:
        (text, mean_confidence)
    """
    with Image.open(path) as img:
        # Pillow uses lazy loading; force a copy so the file handle can close.
        img.load()
        return ocr_image(img, lang=lang)


# ════════════════════════════════════════════════════════════
# 4. OCR a PDF (one page at a time)
# ════════════════════════════════════════════════════════════

def ocr_pdf(
    path: Path,
    lang: str = "eng",
    dpi: int = 200,
) -> List[Tuple[int, str, float]]:
    """
    OCR every page of a PDF.

    `pdf2image.convert_from_path` rasterises each PDF page at the
    requested DPI, returning a list of PIL Images. We then OCR each one.

    Why DPI=200?  Tesseract likes >= 300 DPI, but rasterising a PDF at
    300 DPI is slow. We use 200 + the 2× upscale in preprocess_image()
    → effective ~400 DPI of resolution at half the rendering cost.

    Args:
        path : path to a .pdf file
        lang : Tesseract language code
        dpi  : rasterisation DPI

    Returns:
        List of (page_number_1_indexed, text, mean_confidence) tuples.
    """
    pages_text: List[Tuple[int, str, float]] = []
    # Pass poppler_path on Windows where Poppler isn't on PATH.
    # On Linux/macOS this stays None and pdf2image uses PATH normally.
    convert_kwargs = {"dpi": dpi}
    if _POPPLER_PATH:
        convert_kwargs["poppler_path"] = _POPPLER_PATH

    images = convert_from_path(str(path), **convert_kwargs)
    for page_idx, img in enumerate(images, start=1):
        text, conf = ocr_image(img, lang=lang)
        pages_text.append((page_idx, text, conf))
    return pages_text


# ════════════════════════════════════════════════════════════
# 5. Should we OCR this PDF at all?
# ════════════════════════════════════════════════════════════

def pdf_needs_ocr(path: Path, sample_pages: int = 3) -> bool:
    """
    Heuristic: does this PDF have a real text layer, or is it scanned?

    Most real-world PDFs come in two flavors:
      • "born digital" — generated from Word/LaTeX/etc., have a perfect
        text layer. PyPDFLoader extracts it for free; OCR is a waste.
      • "scanned" — produced by scanning paper. PyPDFLoader returns
        empty (or nearly empty) strings; we MUST OCR to get text.

    We sniff the first few pages with PyPDF and decide based on the
    amount of extractable text. Cheap and 99% accurate.

    Args:
        path         : path to the PDF
        sample_pages : how many pages to check (default 3)

    Returns:
        True if we should OCR, False if PyPDFLoader will do fine.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        # Aggregate the text length from up to N pages.
        total_chars = 0
        for i, page in enumerate(reader.pages[:sample_pages]):
            try:
                total_chars += len(page.extract_text() or "")
            except Exception:
                continue
        # Less than 50 chars per sampled page → probably scanned.
        return total_chars < 50 * sample_pages
    except Exception:
        # If pypdf can't even open it, default to OCR.
        return True
