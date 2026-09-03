# scripts/inspect_candidates.py
import json
from pathlib import Path

candidates_file = Path("data/golden/raw_candidates.jsonl")

with open(candidates_file, "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        candidate = json.loads(line)
        print(f"--- Candidate {i+1} ---")
        print(f"Question: {candidate['question']}")
        print(f"Answer: {candidate['answer']}")
        print(f"Quote: {candidate['supporting_quote']}")
        print(f"Type: {candidate.get('question_type', 'unknown')}")
        print(f"Source: {candidate['source_pdf']}")
        print()