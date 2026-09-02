# scripts/build_quality_log.py
import json
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest.parse import extract_pdf_with_page_tiers

logging.basicConfig(level=logging.INFO)

raw_dir = Path("data/raw")
output_file = Path("data/processed/parse_quality.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

tier_counter = {"pymupdf": 0, "docling": 0, "tesseract": 0, "failed": 0}

with open(output_file, "w") as f_out:
    for pdf_path in [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]:
        try:
            page_texts, page_tiers = extract_pdf_with_page_tiers(pdf_path)
            for i, (text, tier) in enumerate(zip(page_texts, page_tiers)):
                record = {
                    "pdf": pdf_path.name,
                    "page": i,
                    "tier": tier,
                    "char_count": len(text.strip())
                }
                f_out.write(json.dumps(record) + "\n")
                tier_counter[tier] = tier_counter.get(tier, 0) + 1
            logging.info(f"Processed {pdf_path.name}: {len(page_tiers)} pages")
        except Exception as e:
            logging.error(f"Failed to process {pdf_path.name}: {e}")

# Print the split
total = sum(tier_counter.values())
print("\n===== PARSE QUALITY SPLIT =====")
for tier, count in sorted(tier_counter.items(), key=lambda x: -x[1]):
    pct = (count / total * 100) if total > 0 else 0
    print(f"{tier:>10}: {count:>6} pages ({pct:>5.1f}%)")