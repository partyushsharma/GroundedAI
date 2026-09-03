# Create scripts/run_naive_chunking.py. This script will:
# Read all PDFs from data/raw/.
# Use your tiered parser (extract_pdf_with_page_tiers) to get page texts.
# Chunk each page.
# Collect chunk count and token stats.
# Save all chunks to data/processed/chunks.jsonl.

import json
import tiktoken
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.parse import extract_pdf_with_page_tiers
from ingest.models import ChunkMeta
from ingest.chunking import chunk_text
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 512
OVERLAP = int(CHUNK_SIZE * 0.1)  # 51 tokens
tokenizer = tiktoken.get_encoding("cl100k_base")
# ----------------------------

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def main():
    all_pdfs = [p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    logger.info(f"Found {len(all_pdfs)} PDFs to process.")
    
    total_chunks = 0
    total_tokens = 0
    token_lengths = []  # list of token counts per chunk

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for pdf_path in all_pdfs:
            try:
                # 1. Extract per-page text and tiers
                page_texts, page_tiers = extract_pdf_with_page_tiers(pdf_path)
                
                for page_num, (text, tier) in enumerate(zip(page_texts, page_tiers)):
                    if not text.strip():
                        continue  # skip empty pages
                    
                    # 2. Build metadata using your ChunkMeta schema
                    meta = ChunkMeta(
                        pdf_name=pdf_path.name,
                        page_number=page_num,
                        # We'll add extra info later – for now, just basic
                    ).model_dump()  # convert to dict for easier JSON storage
                    
                    # 3. Chunk the page
                    chunk_records = chunk_text(text, meta)
                    
                    # 4. Write each chunk to JSONL and accumulate stats
                    for record in chunk_records:
                        # Count tokens
                        token_count = count_tokens(record["text"])
                        token_lengths.append(token_count)
                        total_tokens += token_count
                        
                        # Write to file
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1
                
                logger.info(f"Processed {pdf_path.name} -> {sum(1 for t in page_texts if t.strip())} non‑empty pages")
                
            except Exception as e:
                logger.error(f"Failed on {pdf_path.name}: {e}")
                continue

    # ---------- Print final statistics ----------
    if total_chunks == 0:
        logger.warning("No chunks were created.")
        return

    avg_tokens = total_tokens / total_chunks
    min_tokens = min(token_lengths)
    max_tokens = max(token_lengths)
    
    print("\n" + "="*50)
    print("NAIVE CHUNKING STATISTICS")
    print("="*50)
    print(f"Total chunks created:   {total_chunks:,}")
    print(f"Total tokens (approx):  {total_tokens:,}")
    print(f"Average tokens/chunk:   {avg_tokens:.1f}")
    print(f"Min tokens/chunk:       {min_tokens}")
    print(f"Max tokens/chunk:       {max_tokens}")
    print(f"Output saved to:        {OUTPUT_FILE}")
    print("="*50)

    # Optional: write stats to a file
    with open("data/processed/chunk_stats.txt", "w") as f:
        f.write(f"Total chunks: {total_chunks}\n")
        f.write(f"Average tokens: {avg_tokens:.1f}\n")
        f.write(f"Min: {min_tokens}, Max: {max_tokens}\n")

if __name__ == "__main__":
    main()