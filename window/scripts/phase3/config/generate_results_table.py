import pandas as pd
from pathlib import Path

CSV = Path("data/ablation_results.csv")
MD_FILE = Path("README.md")

def main():
    df = pd.read_csv(CSV)
    # Select relevant columns
    cols = ['experiment_name', 'recall@10', 'mrr', 'ndcg@10']
    table_df = df[cols]
    # Sort by recall@10 descending
    table_df = table_df.sort_values('recall@10', ascending=False)
    # Convert to markdown
    markdown = table_df.to_markdown(index=False)
    print(markdown)
    # Optionally insert into README or save to file
    with open("data/results_table.md", 'w') as f:
        f.write(markdown)

if __name__ == "__main__":
    main()