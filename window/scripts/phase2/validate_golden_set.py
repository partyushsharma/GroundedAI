import json
import re
from pathlib import Path

GOLDEN_FILE = Path("data/golden/golden_set.jsonl")
CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")

def is_quote_in_text(quote: str, text: str) -> bool:
    quote_clean = quote.strip()
    if not quote_clean:
        return False
    if quote_clean in text:
        return True
    quote_norm = " ".join(quote_clean.split())
    text_norm = " ".join(text.split())
    if quote_norm in text_norm:
        return True
    def normalize(t):
        return re.sub(r'[^\w\s]', '', t.lower())
    return normalize(quote_clean) in normalize(text)

# Load chunks
chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

empty_count = 0
invalid_count = 0
mismatch_count = 0
valid_count = 0

with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        q = json.loads(line)
        ids = q.get("gold_chunk_ids", [])
        if not ids:
            print(f"[EMPTY] Question {idx} (PDF: {q.get('source_pdf', 'N/A')}): empty gold chunk IDs")
            empty_count += 1
            continue
        
        for cid in ids:
            if cid >= len(chunks):
                print(f"[INVALID] Question {idx}: invalid chunk ID {cid} (max {len(chunks)-1})")
                invalid_count += 1
            else:
                quote = q["supporting_quote"]
                if not is_quote_in_text(quote, chunks[cid]["text"]):
                    print(f"[MISMATCH] Question {idx}: quote not found in chunk {cid}")
                    mismatch_count += 1
                else:
                    valid_count += 1

print("\n" + "="*50)
print("VALIDATION SUMMARY")
print("="*50)
print(f"Valid Mappings: {valid_count}")
print(f"Mismatches: {mismatch_count}")
print(f"Empty Gold IDs: {empty_count}")
print(f"Invalid IDs: {invalid_count}")
print("="*50)