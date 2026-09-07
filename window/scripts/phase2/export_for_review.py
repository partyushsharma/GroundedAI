# Your candidates are in data/golden/raw_candidates.jsonl. 
# Convert them to a CSV that can be opened in Excel, Google Sheets, or any spreadsheet tool.

import json
import csv
from pathlib import Path

INPUT_FILE = Path("data/golden/raw_candidates.jsonl")
OUTPUT_CSV = Path("data/golden/review_candidates.csv")

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    # Write header
    writer.writerow([
        "id", "question", "answer", "supporting_quote", 
        "source_pdf", "source_pdf_path", "question_type",
        "review_status", "reviewer_notes", 
        "corrected_question", "corrected_answer", "corrected_quote",
        "gold_chunk_ids"  # will be filled later in Task 2.3
    ])
    
    # Read candidates and write rows
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            candidate = json.loads(line)
            writer.writerow([
                idx, 
                candidate.get('question', ''),
                candidate.get('answer', ''),
                candidate.get('supporting_quote', ''),
                candidate.get('source_pdf', ''),
                candidate.get('source_pdf_path', ''),
                candidate.get('question_type', 'unknown'),
                "",  # review_status – to fill
                "",  # reviewer_notes
                "",  # corrected_question
                "",  # corrected_answer
                "",  # corrected_quote
                ""   # gold_chunk_ids
            ])

print(f"✅ Exported {idx+1} candidates to {OUTPUT_CSV}")