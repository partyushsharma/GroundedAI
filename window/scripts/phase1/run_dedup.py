import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.dedup import deduplicate_chunks
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INPUT_CHUNKS = Path("data/processed/chunks.jsonl")
OUTPUT_CHUNKS = Path("data/processed/chunks_dedup.jsonl")

def main():
    if not INPUT_CHUNKS.exists():
        print(f"❌ Input file not found: {INPUT_CHUNKS}")
        print("   Run naive chunking (task 1.7) first.")
        return
    
    print("🔍 Starting MinHash deduplication...")
    original, unique = deduplicate_chunks(INPUT_CHUNKS, OUTPUT_CHUNKS)
    removed = original - unique
    
    print("\n" + "="*50)
    print("DEDUPLICATION RESULTS")
    print("="*50)
    print(f"Original chunks:      {original:,}")
    print(f"Unique chunks kept:   {unique:,}")
    print(f"Duplicates removed:   {removed:,}")
    print(f"Reduction:            {removed/original*100:.1f}%")
    print(f"Output saved to:      {OUTPUT_CHUNKS}")
    print("="*50)

if __name__ == "__main__":
    main()