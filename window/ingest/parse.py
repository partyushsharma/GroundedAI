import fitz
from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
from pdf2image import convert_from_path
import pytesseract


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