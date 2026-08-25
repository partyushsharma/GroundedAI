# scripts/test_ocr.py
from pathlib import Path
from ingest.parse import extract_pdf

pdf_path = Path("data/raw/29MD8B47C911ECA14450AACBAEF6D7981EF8.pdf")
result = extract_pdf(pdf_path)
print(f"Tier used: {result['tier_used']}")
print(f"Text preview (first 500 chars):\n{result['text'][:500]}")