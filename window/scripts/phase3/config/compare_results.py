# scripts/compare_results.py
import pandas as pd
df = pd.read_csv("data/ablation_results.csv")
# Filter for the three experiments
experiments = ["baseline_dense_flat", "bm25_only", "hybrid_rrf_k60"]
df_filtered = df[df['experiment_name'].isin(experiments)]
print(df_filtered[['experiment_name', 'recall@10', 'mrr', 'ndcg@10']])