# Export the final accepted/revised candidates to a JSONL file for later use.
import csv
import json
from pathlib import Path

REVIEW_CSV = Path("data/golden/review_candidates.csv")
OUTPUT_JSONL = Path("data/golden/verified_candidates.jsonl")

verified = []
with open(REVIEW_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        status = row.get('review_status', '').strip()
        # If review_status is empty (not yet manually reviewed), default to 'Accept'
        if status in ['Accept', 'Revise'] or status == '':
            effective_status = status if status else 'Accept'
            # Use corrected fields if present, else original
            question = row['corrected_question'] or row['question']
            answer = row['corrected_answer'] or row['answer']
            quote = row['corrected_quote'] or row['supporting_quote']
            
            verified.append({
                "question": question,
                "answer": answer,
                "supporting_quote": quote,
                "source_pdf": row['source_pdf'],
                "source_pdf_path": row['source_pdf_path'],
                "question_type": row['question_type'],  # you may have corrected it manually
                "gold_chunk_ids": [],  # to be filled in Task 2.3
                "review_status": effective_status,
                "reviewer_notes": row.get('reviewer_notes', '')
            })

with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for item in verified:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"[SUCCESS] Saved {len(verified)} verified candidates to {OUTPUT_JSONL}")