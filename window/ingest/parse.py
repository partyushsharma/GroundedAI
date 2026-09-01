# pyrefly: ignore [missing-import]
import fitz
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from pdf2image import convert_from_path
import pytesseract
from ingest.models import ChunkMeta



logger = logging.getLogger(__name__)

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

# Implement a quality check to decide when to fallback to docling if PyMuPDF output is poor.
# This can be based on text length, whitespace ratio, or presence of key terms.

def is_extraction_poor(text: str, min_chars: int = 300) -> bool:
    """Heuristic to decide if tier 1 output is too poor to use."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    # Optional: check whitespace ratio (if > 80% whitespace, bad)
    whitespace_ratio = (len(text) - len(text.replace(" ", "").replace("\n", ""))) / max(1, len(text))
    if whitespace_ratio > 0.8:
        return True
    return False

# Single orchestrator function that tries tier 1, checks quality, and escalates to tier 2 if needed.
def extract_pdf(pdf_path: Path, force_tier: str = None) -> dict:
    logger.warning(f"Request Reached")
    """
    Ladder extraction: tries PyMuPDF first, falls back to Docling if quality is poor.
    Returns a dict with:
        'text': extracted text,
        'tier_used': 'pymupdf' or 'docling',
        'pages': page count (approx),
        'quality_metrics': {...}
    """
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
        # We'll let the caller handle the crash (task 1.4 will add tier 3)
        logger.warning(f"Tier 2 crashed on {pdf_path.name}: {e}. Trying Tesseract.")

    # ---- Tier 3 ----
    try:
        text_t3 = extract_with_tesseract(pdf_path)
        # Tesseract may still be messy, but we accept it as the final attempt
        return {"text": text_t3, "tier_used": "tesseract"}
    except Exception as e:
        # All tiers failed – raise a clear error
        raise RuntimeError(f"All parsing tiers failed for {pdf_path.name}: {e}") from e

# Write the Tesseract extraction function
# In ingest/parse.py, add a function that:
# Converts every page of the PDF to an image (using pdf2image).
# Runs Tesseract on each image.
# Combines the results into a single string.
def extract_with_tesseract(pdf_path: Path, dpi: int = 300) -> str:
    """
    Tier 3: full OCR using Tesseract.
    Converts PDF pages to images, runs OCR, returns plain text.
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
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

# --- STEP 1.5 Logging --- 
def extract_page_pymupdf(pdf_path: Path, page_num: int) -> str:
    """Extract text from a specific page using PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]   # 0‑based index
    text = page.get_text()
    doc.close()
    return text

# Page‑level Docling (Tier 2)
# Docling normally processes the whole document. To avoid re‑processing the whole PDF for each page,
# we process the PDF once and cache the page‑wise results. Write a function that returns a list of page texts (one per page):
def extract_all_pages_docling(pdf_path: Path) -> list[str]:
    """Run Docling once and return a list of page texts."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    # Docling stores pages as a dict; we sort by page number
    pages = sorted(result.document.pages.items(), key=lambda x: x[0])
    return [page.text for _, page in pages]

# Page‑level Tesseract (Tier 3)
# Use pdf2image to convert only one page at a time to an image, then OCR it:
def extract_page_tesseract(pdf_path: Path, page_num: int, dpi: int = 300) -> str:
    """OCR a single page using Tesseract."""
    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=dpi)
    if not images:
        return ""
    text = pytesseract.image_to_string(images[0], lang='eng')
    return text

# Write a page‑aware orchestrator
# This function:
# Gets the total page count (using PyMuPDF).
# Tier 1 – extracts all pages with PyMuPDF and checks each page’s quality.
# Tier 2 – if any page is poor, runs Docling once and overwrites those pages (mapping by index).
# Tier 3 – if any page is still poor, runs Tesseract only on those remaining poor pages.
def extract_pdf_with_page_tiers(pdf_path: Path) -> tuple[list[str], list[str]]:
    """
    Returns:
      - page_texts: list of strings, one per page.
      - page_tiers: list of tier names ('pymupdf', 'docling', 'tesseract'), same length.
    """
    # Get page count
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

    # If all pages are fine, return early
    if not poor_indices:
        return page_texts, page_tiers

    # ---- Tier 2: Docling (only if we have poor pages) ----
    try:
        docling_page_texts = extract_all_pages_docling(pdf_path)
        # Docling should have the same number of pages
        if len(docling_page_texts) == page_count:
            for i in poor_indices:
                text = docling_page_texts[i]
                if not is_extraction_poor(text, min_chars=100):  # Docling may return empty on scans
                    page_texts[i] = text
                    page_tiers[i] = "docling"
            # Update poor_indices to keep only those still bad
            still_poor = [i for i in poor_indices if page_tiers[i] == "pending"]
            poor_indices = still_poor
        else:
            # Docling returned different page count – fallback to Tesseract for all poor pages
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
            # Last resort: leave as empty text, mark tier as 'failed'
            logger.error(f"Tesseract also failed on page {i} of {pdf_path.name}: {e}")
            page_texts[i] = ""
            page_tiers[i] = "failed"

    return page_texts, page_tiers


def create_basic_meta(pdf_name: str, page_number: int, **kwargs) -> ChunkMeta:
    """Create a ChunkMeta instance with only the bare essentials."""
    return ChunkMeta(pdf_name=pdf_name, page_number=page_number, **kwargs)

