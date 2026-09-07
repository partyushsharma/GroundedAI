import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from scripts.phase2.retrieval_eval.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    compute_all_metrics,
    evaluate_retrieval
)

def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    gold = [3, 6, 7]
    # k=1: none -> 0
    assert recall_at_k(retrieved, gold, 1) == 0.0
    # k=3: hits {3} -> 1/3 = 0.333...
    assert recall_at_k(retrieved, gold, 3) == 1/3
    # k=5: hits {3} -> 1/3
    assert recall_at_k(retrieved, gold, 5) == 1/3

def test_precision_at_k():
    retrieved = [1, 2, 3, 4, 5]
    gold = [3, 6, 7]
    assert precision_at_k(retrieved, gold, 1) == 0.0
    assert precision_at_k(retrieved, gold, 3) == 1/3
    assert precision_at_k(retrieved, gold, 5) == 1/5

def test_mrr():
    retrieved = [1, 2, 3, 4, 5]
    gold = [3, 6, 7]
    # first relevant is at position 3 -> 1/3
    assert mrr(retrieved, gold) == 1/3
    retrieved2 = [6, 7, 1, 2, 3]
    gold2 = [3, 6, 7]
    # first relevant at pos 1 -> 1
    assert mrr(retrieved2, gold2) == 1.0
    retrieved3 = [1, 2, 4, 5, 8]
    gold3 = [3, 6, 7]
    # no relevant -> 0
    assert mrr(retrieved3, gold3) == 0.0

def test_ndcg_at_k():
    # Example from literature: NDCG@k for binary relevance
    retrieved = [1, 2, 3, 4, 5]
    gold = [3, 6, 7]
    # k=1: DCG = 0, ideal = 1/log2(2)=1, so NDCG=0
    assert ndcg_at_k(retrieved, gold, 1) == 0.0
    # k=3: DCG = 1/log2(3+1) = 1/2, ideal = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.1309. DCG = 0.5. NDCG = 0.5/2.1309 ≈ 0.2346.
    ndcg = ndcg_at_k(retrieved, gold, 3)
    assert 0 <= ndcg <= 1
    # k=5: 1 gold at rank 3 (DCG=0.5), ideal for 3 golds = 2.1309 -> NDCG ≈ 0.2346
    import numpy as np
    ideal_5 = 1.0 + 1.0/np.log2(3) + 1.0/np.log2(4)
    assert abs(ndcg_at_k(retrieved, gold, 5) - (0.5 / ideal_5)) < 1e-6

def test_compute_all_metrics():
    retrieved = [1, 2, 3, 4, 5]
    gold = [3, 6, 7]
    metrics = compute_all_metrics(retrieved, gold, [1, 3, 5])
    assert metrics['recall@1'] == 0.0
    assert metrics['recall@3'] == 1/3
    assert metrics['recall@5'] == 1/3
    assert metrics['precision@1'] == 0.0
    assert metrics['precision@3'] == 1/3
    assert metrics['precision@5'] == 1/5
    assert metrics['mrr'] == 1/3

def test_evaluate_retrieval():
    results = [
        {'query_id': 0, 'retrieved_ids': [1, 2, 3, 4, 5]},
        {'query_id': 1, 'retrieved_ids': [3, 1, 2, 4, 5]},
    ]
    golds = [
        {'query_id': 0, 'gold_ids': [3, 6, 7]},
        {'query_id': 1, 'gold_ids': [1, 2]},
    ]
    avg = evaluate_retrieval(results, golds, [1, 3])
    # Query0: recall@3=1/3, MRR=1/3; Query1: recall@3=2/2=1, MRR=1/2 (first gold '1' at pos 2)
    # avg recall@3 = (1/3+1)/2 = 2/3
    # avg MRR = (1/3 + 1/2)/2 = 5/12
    assert abs(avg['recall@3'] - 2/3) < 1e-9
    assert abs(avg['mrr'] - (1/3 + 1/2)/2) < 1e-9