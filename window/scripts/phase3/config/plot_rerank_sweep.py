import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/rerank_sweep_table.csv")

fig, ax1 = plt.subplots(figsize=(9, 5))

# Left axis: Recall@10
color = 'steelblue'
ax1.set_xlabel('Rerank candidate depth (k)')
ax1.set_ylabel('Recall@10', color=color)
ax1.plot(df['k'], df['recall@10'], marker='o', color=color, linewidth=2, label='Recall@10')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.set_xticks(df['k'])
ax1.set_xticklabels([str(k) for k in df['k']])

# Right axis: p95 latency
ax2 = ax1.twinx()
color2 = 'firebrick'
ax2.set_ylabel('p95 latency (ms)', color=color2)
ax2.plot(df['k'], df['p95_latency_ms'], marker='s', color=color2, linewidth=2, linestyle='--', label='p95 latency')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Rerank Depth Sweep – Quality vs Latency')
fig.tight_layout()
plt.savefig("data/rerank_sweep.png", dpi=150)
print("Saved chart to data/rerank_sweep.png")