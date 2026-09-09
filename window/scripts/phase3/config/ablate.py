# This script will:
# Take a config file path (or run all configs in configs/).
# Load the golden set.
# For each config:
# - Instantiate the pipeline.
# - For each question, run retrieve() to get top‑k IDs.
# - Compute metrics using your retrieval_metrics module.
# - Store results.
# - Append to a master results CSV.
# - Optionally generate a Markdown table. 

#!/usr/bin/env python
import json
import sys
import argparse
from pathlib import Path
import pandas as pd
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(SCRIPT_DIR.parent))
sys.path.append(str(PROJECT_ROOT))

from config_loader import load_config
from retrieval_pipeline import RetrievalPipeline
from scripts.phase2.retrieval_eval.retrieval_metrics import evaluate_retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_FILE = Path("data/golden/golden_set_with_unanswerable.jsonl")
RESULTS_CSV = Path("data/ablation_results.csv")
RESULTS_MD = Path("README.md")  # or separate table file

def run_experiment(config_path: Path):
    config = load_config(config_path)
    logger.info(f"Running experiment: {config.experiment_name}")
    
    # 1. Load golden set (skip unanswerable for retrieval metrics)
    questions = []
    with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            if not q.get('is_unanswerable', False):
                questions.append(q)
    logger.info(f"Loaded {len(questions)} answerable questions")
    
    # 2. Initialize pipeline
    pipeline = RetrievalPipeline(config)
    
    # 3. Run retrieval for each question
    results = []
    golds = []
    for idx, q in enumerate(questions):
        gold_ids = q.get('gold_chunk_ids', [])
        if not gold_ids:
            continue
        retrieved_ids = pipeline.retrieve(q['question'], k=config.retrieve_k)
        results.append({'query_id': idx, 'retrieved_ids': retrieved_ids})
        golds.append({'query_id': idx, 'gold_ids': gold_ids})
    
    # 4. Compute metrics
    metrics = evaluate_retrieval(results, golds, config.k_values)
    metrics['experiment_name'] = config.experiment_name
    metrics['description'] = config.description
    
    # 5. Record in CSV
    df = pd.DataFrame([metrics])
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        # Remove old entry with same experiment name (if re-running)
        existing = existing[existing['experiment_name'] != config.experiment_name]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    logger.info(f"Results appended to {RESULTS_CSV}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--config', type=str, help="Path to a single YAML config file")
    parser.add_argument('--all', action='store_true', help="Run all configs in configs/")
    args = parser.parse_args()
    
    if args.config:
        config_path = Path(args.config)
        # If the path doesn't exist as given (relative to cwd), try a series
        # of fallback base directories so common invocation styles all work:
        #   hybrid_rrf.yaml              → SCRIPT_DIR/hybrid_rrf.yaml
        #   phase3/config/hybrid_rrf.yaml → PROJECT_ROOT/scripts/phase3/...
        #   scripts/phase3/config/...    → PROJECT_ROOT/scripts/phase3/...
        if not config_path.exists():
            fallbacks = [
                SCRIPT_DIR / config_path,
                PROJECT_ROOT / "scripts" / config_path,
                PROJECT_ROOT / config_path,
            ]
            for candidate in fallbacks:
                if candidate.exists():
                    config_path = candidate
                    break
        run_experiment(config_path)
    elif args.all:
        for config_path in SCRIPT_DIR.glob("*.yaml"):
            run_experiment(config_path)
    else:
        print("Please specify --config or --all")
        sys.exit(1)

if __name__ == "__main__":
    main()