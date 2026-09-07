# scripts/combine_golden_set.py
import json
from pathlib import Path

GOLDEN_FILE = Path("data/golden/golden_set.jsonl")
UNANSWERABLE_FILE = Path("data/golden/unanswerable.jsonl")
COMBINED_FILE = Path("data/golden/golden_set_with_unanswerable.jsonl")

with open(COMBINED_FILE, 'w', encoding='utf-8') as f_out:
    # Write answerable questions
    with open(GOLDEN_FILE, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            q = json.loads(line)
            q["is_unanswerable"] = False
            f_out.write(json.dumps(q, ensure_ascii=False) + '\n')
    # Write unanswerable questions
    with open(UNANSWERABLE_FILE, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            q = json.loads(line)
            q["is_unanswerable"] = True
            f_out.write(json.dumps(q, ensure_ascii=False) + '\n')

print("Combined golden set saved.")