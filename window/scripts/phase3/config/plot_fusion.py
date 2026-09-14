import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ablation_results.csv")
experiments = [
    "baseline_dense_flat",
    "bm25_only",
    "hybrid_rrf_k60",
    "hybrid_weighted_alpha0.3",
    "hybrid_weighted_alpha0.5",
    "hybrid_weighted_alpha0.7",
]
df_filtered = df[df['experiment_name'].isin(experiments)]

plt.figure(figsize=(10, 5))
plt.bar(df_filtered['experiment_name'], df_filtered['recall@10'], color='steelblue')
plt.xticks(rotation=30, ha='right')
plt.ylabel("Recall@10")
plt.title("Fusion Method Comparison – Recall@10")
plt.tight_layout()
plt.savefig("data/fusion_comparison.png")
print("Chart saved to data/fusion_comparison.png")