# scripts/validate_unanswerable.py
import json
from pathlib import Path

UNANSWERABLE_FILE = Path("data/golden/unanswerable.jsonl")
CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")

# Load all chunk texts
chunk_texts = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunk = json.loads(line)
        chunk_texts.append(chunk["text"].lower())

# Load unanswerable questions
with open(UNANSWERABLE_FILE, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        q = json.loads(line)
        question = q["question"].lower()
        
        # Extract key terms (words longer than 4 chars, excluding common words)
        stopwords = {'what', 'under', 'for', 'the', 'of', 'to', 'from', 'with', 'for', 'its', 'are', 'can', 'how', 'under', 'upon'}
        keywords = [w for w in question.split() if len(w) > 4 and w not in stopwords]
        
        # Check if any chunk contains ALL keywords
        found = False
        for text in chunk_texts:
            if all(kw in text for kw in keywords[:3]):  # Check top 3 keywords
                found = True
                break
        
        if found:
            print(f"⚠️  Question {idx} MIGHT be answerable: {q['question'][:80]}...")
            print(f"   Keywords found: {[kw for kw in keywords[:3] if any(kw in text for text in chunk_texts)]}")
        else:
            print(f"✅ Question {idx} appears unanswerable.")