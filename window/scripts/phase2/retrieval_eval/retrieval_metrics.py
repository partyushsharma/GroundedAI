# Steps for Task 2.6 – Build retrieval metrics module (no LLM)
# This module will compute the core retrieval metrics for your golden set:
# - Recall@k: fraction of relevant chunks found in top‑k results.
# - Precision@k: fraction of retrieved chunks that are relevant.
# - MRR: mean reciprocal rank of the first relevant result.
# - nDCG@k: normalised discounted cumulative gain (handles multiple relevances).

"""
Retrieval metrics for RAG evaluation.
All functions compute scores without calling any LLM.
"""

from typing import List, Set, Union, Optional
import numpy as np


def recall_at_k(
    retrieved_ids: List[int],
    gold_ids: List[int],
    k: int
) -> float:
    """
    Compute Recall@k: fraction of gold IDs that appear in the first k retrieved IDs.
    If gold_ids is empty, returns 0.0 (or could be undefined; we choose 0.0).
    """
    if not gold_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(retrieved_set.intersection(gold_set))
    return hits / len(gold_set)


def precision_at_k(
    retrieved_ids: List[int],
    gold_ids: List[int],
    k: int
) -> float:
    """
    Compute Precision@k: fraction of retrieved IDs in top-k that are relevant.
    If k==0, returns 0.0.
    """
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(retrieved_set.intersection(gold_set))
    return hits / k


def mrr(
    retrieved_ids: List[int],
    gold_ids: List[int]
) -> float:
    """
    Mean Reciprocal Rank: 1 / (position of first relevant result).
    Returns 0.0 if no relevant result found.
    """
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved_ids: List[int],
    gold_ids: List[int],
    k: int,
    gain_function: Optional[callable] = None
) -> float:
    """
    Discounted Cumulative Gain at k.
    By default, gain = 1 if doc is relevant (binary relevance), else 0.
    """
    if gain_function is None:
        # binary relevance
        gold_set = set(gold_ids)
        gain_function = lambda doc_id: 1.0 if doc_id in gold_set else 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        gain = gain_function(doc_id)
        dcg += gain / np.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved_ids: List[int],
    gold_ids: List[int],
    k: int,
    gain_function: Optional[callable] = None
) -> float:
    """
    Normalised DCG at k.
    Returns 0.0 if ideal DCG is 0 (no relevant docs or k=0).
    """
    if k == 0 or not gold_ids:
        return 0.0

    # Compute DCG for actual ranking
    dcg = dcg_at_k(retrieved_ids, gold_ids, k, gain_function)

    # Compute ideal DCG: sort by gain descending
    if gain_function is None:
        # binary: put all golds first, then non-golds
        gold_set = set(gold_ids)
        gain_func = lambda doc_id: 1.0 if doc_id in gold_set else 0.0
    else:
        gain_func = gain_function

    # Create a list of all docs in retrieved_ids
    if gain_function is None:
        ideal = 0.0
        for i in range(1, min(len(gold_ids), k) + 1):
            ideal += 1.0 / np.log2(i + 1)
    else:
        ideal = 0.0
        for i in range(1, min(len(gold_ids), k) + 1):
            ideal += 1.0 / np.log2(i + 1)

    return dcg / ideal if ideal > 0 else 0.0


def compute_all_metrics(
    retrieved_ids: List[int],
    gold_ids: List[int],
    k_values: List[int] = [1, 5, 10, 20]
) -> dict:
    """
    Compute a dictionary of all metrics for a single query.
    Returns dict with keys: recall@k, precision@k, mrr, ndcg@k.
    """
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, gold_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, gold_ids, k)
    metrics["mrr"] = mrr(retrieved_ids, gold_ids)
    return metrics


def evaluate_retrieval(
    results: List[dict],
    golds: List[dict],
    k_values: List[int] = [1, 5, 10, 20]
) -> dict:
    """
    Evaluate retrieval results against gold chunk IDs.
    results: list of dicts, each with 'query_id' and 'retrieved_ids' (list of chunk IDs).
    golds: list of dicts, each with 'query_id' and 'gold_ids' (list of chunk IDs).
    Returns dict of averaged metrics.
    """
    # Build mapping from query_id to gold_ids
    gold_map = {item['query_id']: item['gold_ids'] for item in golds}

    all_metrics = []
    for res in results:
        qid = res['query_id']
        if qid not in gold_map:
            continue
        gold_ids = gold_map[qid]
        retrieved = res['retrieved_ids']
        metrics = compute_all_metrics(retrieved, gold_ids, k_values)
        all_metrics.append(metrics)

    # Average across queries
    if not all_metrics:
        return {}
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    return avg_metrics