# Steps for Task 2.6 – Build retrieval metrics module (no LLM)
# This module will compute the core retrieval metrics for your golden set:
# - Recall@k: fraction of relevant chunks found in top‑k results.
# - Precision@k: fraction of retrieved chunks that are relevant.
# - MRR: mean reciprocal rank of the first relevant result.
# - nDCG@k: normalised discounted cumulative gain (handles multiple relevances).