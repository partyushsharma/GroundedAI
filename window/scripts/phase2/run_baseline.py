# What baseline means here:
# - Dense retrieval only (FAISS Flat – exact cosine similarity).
# - No hybrid search, no reranking, no query transforms.
# - Retrieve the top‑k documents (we’ll compute metrics at k=1,5,10,20).
# - Use the default embedding model (BGE‑base‑en‑v1.5).

"""
Run the naive dense retrieval baseline on the answerable golden set.
Save the results as a JSON file for later reference and print a summary.
"""

import json
import numpy as np
from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
import faiss

from retrieval_eval import compute_all_metrics, evaluate_retrieval
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
GOLDEN_FILE = Path("data/golden/golden_set_with_unanswerable.jsonl")
FAISS_INDEX_PATH = Path("data/indexes/faiss.index")
METADATA_PATH = Path("data/indexes/metadata.json")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 20  # we'll compute metrics for k up to 20
OUTPUT_FILE = Path("data/baseline_results.json")
# -----------------------------

def main():
    # 1. Load golden set (only answerable questions)
    questions = []
    with open(GOLDEN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            # Skip unanswerable (they have no gold IDs)
            if q.get('is_unanswerable', False):
                continue
            questions.append(q)
    logger.info(f"Loaded {len(questions)} answerable questions")

    # 2. Load FAISS index and metadata
    logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index has {index.ntotal} vectors")

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)  # list of dicts, each with 'text', 'meta'
    logger.info(f"Loaded metadata for {len(all_chunks)} chunks")

    # 3. Load embedding model
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Prepare results list for evaluate_retrieval
    results = []  # list of dicts with query_id, retrieved_ids
    golds = []    # list of dicts with query_id, gold_ids

    for idx, q in enumerate(questions):
        # Get gold chunk IDs
        gold_ids = q.get('gold_chunk_ids', [])
        if not gold_ids:
            # Skip questions with no gold IDs (shouldn't happen)
            logger.warning(f"Question {idx} has no gold chunk IDs, skipping")
            continue

        # Embed query
        query_text = q['question']
        query_emb = model.encode([query_text], normalize_embeddings=True).astype('float32')

        # Search FAISS
        distances, indices = index.search(query_emb, TOP_K)
        retrieved_ids = indices[0].tolist()

        # Store
        results.append({
            'query_id': idx,
            'retrieved_ids': retrieved_ids
        })
        golds.append({
            'query_id': idx,
            'gold_ids': gold_ids
        })

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx+1} queries")

    # 5. Compute metrics
    logger.info("Computing metrics...")
    k_values = [1, 5, 10, 20]
    avg_metrics = evaluate_retrieval(results, golds, k_values)

    # 6. Print and save results
    print("\n" + "="*50)
    print("NAIVE BASELINE RETRIEVAL RESULTS")
    print("="*50)
    for k in k_values:
        print(f"Recall@{k:2d}:  {avg_metrics.get(f'recall@{k}', 0):.4f}")
        print(f"Precision@{k:2d}: {avg_metrics.get(f'precision@{k}', 0):.4f}")
        print(f"nDCG@{k:2d}:    {avg_metrics.get(f'ndcg@{k}', 0):.4f}")
    print(f"MRR:            {avg_metrics.get('mrr', 0):.4f}")
    print("="*50)

    # Save to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'configuration': 'Dense only - FAISS Flat (exact baseline)',
            'model': MODEL_NAME,
            'top_k': TOP_K,
            'metrics': avg_metrics,
            'num_queries': len(results)
        }, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()