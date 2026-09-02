# This file will contain:
# A tokeniser (convert text into a set of shingles or words).
# A MinHash generator.
# An LSH index that groups near‑duplicates.
# A function that reads your chunks.jsonl, marks duplicates, and writes a cleaned version.

import json
import re
from pathlib import Path
from typing import List, Set, Tuple
from datasketch import MinHash, MinHashLSH
import logging

logger = logging.getLogger(__name__)

# ---------- Configuration ----------
NUM_PERM = 128          # Number of hash functions – higher = more accurate, slower
THRESHOLD = 0.85        # Jaccard similarity threshold (0.85 = 85% similar)
# -----------------------------------

def tokenize(text: str) -> Set[str]:
    """
    Convert text into a set of tokens (shingles).
    We use word-level shingles (5-grams) to capture phrasing, not just exact words.
    This is more robust than character shingles for regulatory text.
    """
    # Clean text: lower case, remove punctuation, split into words
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    # Use 5-gram shingles (order matters)
    shingles = set()
    for i in range(len(words) - 4):
        shingle = " ".join(words[i:i+5])
        shingles.add(shingle)
    # If the text is very short, fallback to word set
    if not shingles:
        shingles = set(words)
    return shingles

def get_minhash(text: str) -> MinHash:
    """Generate a MinHash signature for a given text."""
    tokens = tokenize(text)
    m = MinHash(num_perm=NUM_PERM)
    for token in tokens:
        m.update(token.encode('utf8'))
    return m

def deduplicate_chunks(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Reads chunks from input_path, finds near-duplicates using MinHash LSH,
    writes only the unique chunks to output_path.
    Returns (original_count, unique_count).
    """
    # 1. Load all chunks
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    original_count = len(chunks)
    logger.info(f"Loaded {original_count} chunks.")
    
    # 2. Build LSH index
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    # We'll store a mapping from chunk index to MinHash
    minhashes = []
    
    for idx, chunk in enumerate(chunks):
        text = chunk['text']
        if not text.strip():
            minhashes.append(None)
            continue
        m = get_minhash(text)
        minhashes.append(m)
        # Insert into LSH (using string ID)
        lsh.insert(str(idx), m)
    
    # 3. Find duplicate groups
    # We keep the first occurrence of each duplicate group; mark others for removal.
    keep_indices = set()
    duplicate_count = 0
    processed = set()
    
    for idx, m in enumerate(minhashes):
        if m is None or idx in processed:
            continue
        # Query LSH for near-duplicates (including itself)
        result_ids = lsh.query(m)
        # Convert string IDs back to integers
        result_indices = [int(rid) for rid in result_ids if int(rid) < len(chunks)]
        # Filter out any that are actually below threshold (LSH is approximate)
        # We can do a quick exact Jaccard check for confidence
        exact_dupes = []
        for rid in result_indices:
            if rid == idx:
                exact_dupes.append(rid)
                continue
            # Compute exact Jaccard
            jaccard = m.jaccard(minhashes[rid])
            if jaccard >= THRESHOLD:
                exact_dupes.append(rid)
        
        # Keep the first one in the group, discard the rest
        if exact_dupes:
            keep_indices.add(exact_dupes[0])
            duplicate_count += len(exact_dupes) - 1
            processed.update(exact_dupes)
    
    # 4. Write out only the kept chunks
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for idx in sorted(keep_indices):
            json.dump(chunks[idx], f_out, ensure_ascii=False)
            f_out.write('\n')
    
    unique_count = len(keep_indices)
    logger.info(f"Kept {unique_count} unique chunks, removed {duplicate_count} duplicates.")
    return original_count, unique_count