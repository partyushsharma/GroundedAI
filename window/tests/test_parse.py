from pathlib import Path
from ingest.parse import extract_with_pymupdf

def test_extract_clean_pdf():
    sample = Path("data/raw/02MD787706E669D641C090B415DBE22DEE29.PDF")
    if not sample.exists():
        print("Skipping test: no sample PDF found.")
        return
    text = extract_with_pymupdf(sample)
    assert len(text) > 100
    print(f"Extracted {len(text)} characters.")