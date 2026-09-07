import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")
VERIFIED_FILE = Path("data/golden/verified_candidates.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.jsonl")

# Load chunks with their IDs
chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        chunk = json.loads(line)
        chunks.append({
            "id": idx,
            "text": chunk["text"],
            "meta": chunk["meta"]
        })
logger.info(f"Loaded {len(chunks)} chunks")


def find_chunk_ids_for_quote(quote: str, chunks: list, source_pdf: str = None) -> list[int]:
    """
    Return all chunk IDs where the quote appears as a substring.
    First filters chunks by source_pdf if provided.
    If none found, try looser matches (whitespace, punctuation).
    """
    quote_clean = quote.strip()
    if not quote_clean:
        return []

    # Filter candidate chunks by PDF if specified
    if source_pdf:
        target_chunks = [c for c in chunks if c["meta"].get("pdf_name") == source_pdf]
        if not target_chunks:
            target_chunks = chunks
    else:
        target_chunks = chunks

    # First pass: exact substring match
    matched_ids = [c["id"] for c in target_chunks if quote_clean in c["text"]]
    if matched_ids:
        return matched_ids
    
    # Second pass: normalise whitespace (multiple spaces/newlines -> single space)
    quote_norm = " ".join(quote_clean.split())
    matched_ids = [c["id"] for c in target_chunks if quote_norm in " ".join(c["text"].split())]
    if matched_ids:
        return matched_ids
    
    # Third pass: ignore case and punctuation (aggressive fallback)
    import re
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower())
    
    quote_agg = normalize(quote_clean)
    matched_ids = [c["id"] for c in target_chunks if quote_agg in normalize(c["text"])]
    if matched_ids:
        return matched_ids

    # Fallback to all chunks if pdf_name filter was restrictive
    if source_pdf and target_chunks != chunks:
        return find_chunk_ids_for_quote(quote, chunks, source_pdf=None)
    
    return []


def main():
    verified = []
    with open(VERIFIED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            verified.append(json.loads(line))
    
    logger.info(f"Loaded {len(verified)} verified candidates")
    
    golden_set = []
    empty_count = 0
    
    for candidate in verified:
        quote = candidate.get("supporting_quote", "")
        source_pdf = candidate.get("source_pdf", "")
        if not quote:
            logger.warning(f"Question has no supporting quote: {candidate['question'][:50]}...")
            candidate["gold_chunk_ids"] = []
            empty_count += 1
            golden_set.append(candidate)
            continue
        
        # Find chunks scoped to source_pdf
        ids = find_chunk_ids_for_quote(quote, chunks, source_pdf=source_pdf)
        candidate["gold_chunk_ids"] = ids
        
        if not ids:
            empty_count += 1
            logger.warning(f"No chunk found for quote: '{quote[:80]}...' in candidate: {candidate['question'][:50]}")
        else:
            logger.info(f"Found {len(ids)} chunk(s) for question: {candidate['question'][:50]}...")
        
        golden_set.append(candidate)
    
    # Save the final golden set
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in golden_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n" + "="*50)
    print("GOLD CHUNK MAPPING RESULTS")
    print("="*50)
    print(f"Total verified questions: {len(verified)}")
    print(f"Questions with gold chunks: {len(verified) - empty_count}")
    print(f"Questions with empty gold list: {empty_count}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("="*50)
    
    if empty_count > 0:
        print("\n[WARNING] Some questions have no gold chunk IDs!")
        print("   You need to manually fix these before proceeding.")
    else:
        print("\n[SUCCESS] All questions have at least one gold chunk ID!")

if __name__ == "__main__":
    main()