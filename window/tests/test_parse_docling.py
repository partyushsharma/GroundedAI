from pathlib import Path
from ingest.parse import extract_pdf


def test_parse_docling():
    pdf_path = Path("data/raw/263MD.pdf")
    if not pdf_path.exists():
        samples = [p for p in Path("data/raw").iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        if not samples:
            return
        pdf_path = samples[0]
    result = extract_pdf(pdf_path)
    print(f"Tier used: {result['tier_used']}")
    print(f"Text preview:\n{result['text'][:1000]}")
    print(f"Characters: {result['quality_metrics']['chars']}")
    assert result is not None
    assert "tier_used" in result
    assert "text" in result
    assert "quality_metrics" in result
