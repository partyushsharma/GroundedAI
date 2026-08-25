"""Retrieval: question -> the best 8 passages.

Architecture layers 11-14. Built in Phase 3.
  query_transform.py -- expansion, decomposition, HyDE
  hybrid.py          -- dense + sparse run in parallel, merged with RRF
  rerank.py          -- bge-reranker-v2-m3 re-scores the top 100, keeps 8
  diversity.py       -- MMR plus a per-source cap"""
