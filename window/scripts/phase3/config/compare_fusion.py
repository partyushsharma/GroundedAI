import pandas as pd

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
df_filtered = df_filtered.sort_values('recall@10', ascending=False)

print("\n" + "="*80)
print("FUSION METHOD COMPARISON")
print("="*80)
print(df_filtered[['experiment_name', 'recall@10', 'mrr', 'ndcg@10']].to_string(index=False))
print("="*80)