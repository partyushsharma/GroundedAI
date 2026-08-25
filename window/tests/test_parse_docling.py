from pathlib import Path
from ingest.parse import extract_pdf


def test_parse_docling():
    pdf_path = Path("data/raw/263MD.pdf") #table format 
    result = extract_pdf(pdf_path)
    print(f"Tier used: {result['tier_used']}")
    print(f"Text preview:\n{result['text'][:100000]}")
    print(f"Characters: {result['quality_metrics']['chars']}")
    assert result is not None
    assert "tier_used" in result
    assert "text" in result
    assert "quality_metrics" in result
