# pyrefly: ignore [missing-import]
import os
import shutil
import fitz
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from pdf2image import convert_from_path
import pytesseract
from ingest.models import ChunkMeta

logger = logging.getLogger(__name__)

# Fallback paths for Windows environments
_POPPLER_FALLBACKS = [
    r"D:\Software\poppler\poppler-26.02.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\bin",
]

_TESSERACT_FALLBACKS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\ankit\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"D:\Software\Tesseract-OCR\tesseract.exe",
    r"D:\Software\tesseract\tesseract.exe",
]

def find_poppler_path() -> str | None:
    if shutil.which("pdftoppm"):
        return None
    for folder in _POPPLER_FALLBACKS:
        candidate = os.path.join(folder, "pdftoppm.exe")
        if os.path.isfile(candidate):
            return folder
    return None

def setup_tesseract() -> bool:
    if shutil.which("tesseract"):
        return True
    for candidate in _TESSERACT_FALLBACKS:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return True
    return False

POPPLER_PATH = find_poppler_path()
TESSERACT_AVAILABLE = setup_tesseract()

# ---- Tier 1 ----
def extract_with_pymupdf(pdf_path: Path) -> str:
    """
    Tier 1: fast text extraction using PyMuPDF.
    Returns the full text of the PDF as a single string.
    Raises an exception if extraction fails entirely.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
        doc.close()
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from {pdf_path.name}")
        return full_text
    except Exception as e:
        logger.error(f"PyMuPDF failed on {pdf_path}: {e}")
        raise RuntimeError(f"PyMuPDF extraction failed: {e}") from e

def extract_with_docling(pdf_path: Path) -> str:
    """
    Tier 2: layout aware extraction using Docling.
    Handles tables, columns, and mixed layouts.
    Returns text as a single string.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        logger.info(f"Docling extracted {len(markdown_text)} chars from {pdf_path.name}")
        return markdown_text
    except Exception as e:
        logger.error(f"Docling failed on {pdf_path}: {e}")
        raise RuntimeError(f"Docling extraction failed: {e}") from e

def is_extraction_poor(text: str, min_chars: int = 300) -> bool:
    """Heuristic to decide if tier 1 output is too poor to use."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    whitespace_ratio = (len(text) - len(text.replace(" ", "").replace("\n", ""))) / max(1, len(text))
    if whitespace_ratio > 0.8:
        return True
    return False

def extract_pdf(pdf_path: Path, force_tier: str = None) -> dict:
    logger.warning(f"Request Reached")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ---- Tier 1 ----
    try:
        text_tier1 = extract_with_pymupdf(pdf_path)
        if not is_extraction_poor(text_tier1):
            return {
                "text": text_tier1,
                "tier_used": "pymupdf",
                "pages": None,
                "quality_metrics": {"chars": len(text_tier1)}
            }
        else:
            logger.warning(f"PyMuPDF output poor for {pdf_path.name}. Falling back to Docling.")
    except Exception as e:
        logger.warning(f"PyMuPDF crashed on {pdf_path.name}: {e}. Falling back to Docling.")

    # ---- Tier 2 ----
    try:
        text_tier2 = extract_with_docling(pdf_path)
        return {
            "text": text_tier2,
            "tier_used": "docling",
            "pages": None,
            "quality_metrics": {"chars": len(text_tier2)}
        }
    except Exception as e:
        logger.warning(f"Tier 2 crashed on {pdf_path.name}: {e}. Trying Tesseract.")

    # ---- Tier 3 ----
    try:
        text_t3 = extract_with_tesseract(pdf_path)
        return {"text": text_t3, "tier_used": "tesseract"}
    except Exception as e:
        raise RuntimeError(f"All parsing tiers failed for {pdf_path.name}: {e}") from e

def extract_with_tesseract(pdf_path: Path, dpi: int = 300) -> str:
    """
    Tier 3: full OCR using Tesseract.
    Converts PDF pages to images, runs OCR, returns plain text.
    """
    if not TESSERACT_AVAILABLE and not setup_tesseract():
        raise RuntimeError("Tesseract is not installed or not found on PATH.")

    try:
        kwargs = {"dpi": dpi}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
        images = convert_from_path(pdf_path, **kwargs)
        text_parts = []
        
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang='eng')
            text_parts.append(page_text)
            logger.debug(f"OCR done for page {i+1} of {pdf_path.name}")
        
        full_text = "\n\n".join(text_parts)
        logger.info(f"Tesseract extracted {len(full_text)} chars from {pdf_path.name}")
        return full_text
        
    except Exception as e:
        logger.error(f"Tesseract failed on {pdf_path}: {e}")
        raise RuntimeError(f"Tesseract extraction failed: {e}") from e

def extract_page_pymupdf(pdf_path: Path, page_num: int) -> str:
    """Extract text from a specific page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]   # 0‑based index
    text = page.get_text()
    doc.close()
    return text

def extract_all_pages_docling(pdf_path: Path) -> list[str]:
    """Run Docling once and return a list of page texts."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    num_pages = doc.num_pages()
    return [doc.export_to_markdown(page_no=i + 1) for i in range(num_pages)]

def extract_page_tesseract(pdf_path: Path, page_num: int, dpi: int = 300) -> str:
    """OCR a single page using Tesseract."""
    if not TESSERACT_AVAILABLE and not setup_tesseract():
        raise RuntimeError("Tesseract is not installed or not found on PATH.")
    kwargs = {"first_page": page_num + 1, "last_page": page_num + 1, "dpi": dpi}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH
    images = convert_from_path(pdf_path, **kwargs)
    if not images:
        return ""
    text = pytesseract.image_to_string(images[0], lang='eng')
    return text

def extract_pdf_with_page_tiers(pdf_path: Path) -> tuple[list[str], list[str]]:
    """
    Returns:
      - page_texts: list of strings, one per page.
      - page_tiers: list of tier names ('pymupdf', 'docling', 'tesseract'), same length.
    """
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    # ---- Tier 1: PyMuPDF per page ----
    page_texts = [None] * page_count
    page_tiers = [None] * page_count
    poor_indices = []

    for i in range(page_count):
        text = extract_page_pymupdf(pdf_path, i)
        page_texts[i] = text
        if is_extraction_poor(text):
            poor_indices.append(i)
            page_tiers[i] = "pending"
        else:
            page_tiers[i] = "pymupdf"

    if not poor_indices:
        return page_texts, page_tiers

    # ---- Tier 2: Docling (only if we have poor pages) ----
    try:
        docling_page_texts = extract_all_pages_docling(pdf_path)
        if len(docling_page_texts) == page_count:
            for i in poor_indices:
                text = docling_page_texts[i]
                if not is_extraction_poor(text, min_chars=100):
                    page_texts[i] = text
                    page_tiers[i] = "docling"
            poor_indices = [i for i in poor_indices if page_tiers[i] == "pending"]
        else:
            pass
    except Exception as e:
        logger.warning(f"Docling failed entirely for {pdf_path.name}: {e}")

    # ---- Tier 3: Tesseract (for any remaining poor pages) ----
    for i in poor_indices:
        try:
            text = extract_page_tesseract(pdf_path, i)
            page_texts[i] = text
            page_tiers[i] = "tesseract"
        except Exception as e:
            logger.error(f"Tesseract also failed on page {i} of {pdf_path.name}: {e}")
            page_texts[i] = ""
            page_tiers[i] = "failed"

    return page_texts, page_tiers

def create_basic_meta(pdf_name: str, page_number: int, **kwargs) -> ChunkMeta:
    """Create a ChunkMeta instance with only the bare essentials."""
    return ChunkMeta(pdf_name=pdf_name, page_number=page_number, **kwargs)


