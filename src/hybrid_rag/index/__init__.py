"""Indexing: text -> vectors -> searchable indexes.

Architecture layers 6-10. Built in Phase 1, tuned in Phase 3.
  embed.py         -- sentence-transformers wrapper over BGE / E5 / nomic
  faiss_index.py   -- FAISS Flat / HNSW / IVF, parameter sweeps live here
  chroma_index.py  -- ChromaDB, kept alongside FAISS to compare metadata filtering
  bm25_index.py    -- bm25s keyword index, the sparse half of hybrid search
  lifecycle.py     -- version / rebuild / swap indexes safely (Phase 5)"""
