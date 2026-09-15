import pandas as pd
from pathlib import Path

CSV = Path("data/ablation_results.csv")
df = pd.read_csv(CSV)

# Filter the sweep rows
sweep = df[df['experiment_name'].str.startswith('rerank_sweep_')].copy()

# Extract the k value from the experiment name (or add a column to the config if you prefer)
sweep['k'] = sweep['experiment_name'].str.extract(r'k(\d+)').astype(int)
sweep = sweep.sort_values('k')

# Select columns we care about
cols = ['k', 'recall@10', 'mrr', 'ndcg@10', 'p95_latency_ms']
# Only keep columns that exist (recall@10 may be 'recall@10' in your CSV)
available = [c for c in cols if c in sweep.columns]
print(sweep[available].to_string(index=False))

# Save a clean table for the README
sweep[available].to_csv("data/rerank_sweep_table.csv", index=False)
print("\nSaved to data/rerank_sweep_table.csv")

# ---- Compute marginal gains ----
sweep = sweep.reset_index(drop=True)
if 'recall@10' in sweep.columns:
    print("\nMarginal Recall@10 gain per step:")
    for i in range(1, len(sweep)):
        prev = sweep.loc[i-1]
        curr = sweep.loc[i]
        recall_gain = curr['recall@10'] - prev['recall@10']
        latency_cost = curr['p95_latency_ms'] - prev['p95_latency_ms']
        print(
            f"  k={int(prev['k'])} → k={int(curr['k'])}: "
            f"+{recall_gain:.4f} recall, +{latency_cost:.0f} ms"
        )