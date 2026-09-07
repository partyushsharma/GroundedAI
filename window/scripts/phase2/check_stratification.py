# scripts/check_stratification.py
import json
from pathlib import Path
from collections import Counter

GOLDEN_FILE = Path("data/golden/golden_set_with_unanswerable.jsonl")

# Load and count
type_counts = Counter()
total = 0

with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        q = json.loads(line)
        qtype = q.get("question_type", "unknown")
        type_counts[qtype] += 1
        total += 1

# Define targets from Golden Set Tracker
targets = {
    "factoid": 80,
    "exact identifier": 20,
    "multi-hop": 35,
    "comparative": 15,
    "temporal": 10,
    "unanswerable": 20,
}

print("\n" + "="*50)
print("GOLDEN SET STRATIFICATION CHECK")
print("="*50)
print(f"Total questions: {total}")
print("\nType breakdown:")
print(f"{'Type':<20} {'Target':<10} {'Actual':<10} {'Status':<10}")
print("-"*50)

all_met = True
for qtype, target in targets.items():
    actual = type_counts.get(qtype, 0)
    status = "✅" if actual >= target else f"❌ need {target - actual} more"
    if actual < target:
        all_met = False
    print(f"{qtype:<20} {target:<10} {actual:<10} {status}")

print("="*50)

if all_met:
    print("\n✅ All stratification targets met! Proceed to Task 2.6.")
else:
    print("\n⚠️  Some types are below target. See instructions below.")

# Optional: show any extra types
for qtype, count in type_counts.items():
    if qtype not in targets:
        print(f"ℹ️  Extra type found: {qtype} ({count} questions)")