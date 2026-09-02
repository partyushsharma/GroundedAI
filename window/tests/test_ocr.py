import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.parse import extract_pdf

def test_ocr():
    pdf_path = Path("data/raw/06MDE170516F633150EBCFE438084174F7DECCDC20C.PDF")
    if not pdf_path.exists():
        return
    result = extract_pdf(pdf_path)
    assert result is not None
    assert "text" in result
    assert "tier_used" in result