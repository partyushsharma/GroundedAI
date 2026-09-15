# Task 3.3 ####################################################
# Now you'll combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF). This is a simple but powerful method that merges two ranked lists 
# based on their positions, ignoring the raw scores.
# The key insight: dense and sparse scores are on different scales – averaging them is wrong.
#  RRF sidesteps this by using only the rank positions.
################################################################

import json
import faiss
import time
import bm25s
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from config_loader import RetrievalConfig
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
class RetrievalPipeline:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.faiss_index = faiss.read_index(config.faiss_index_path)
        with open("data/indexes/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # BM25 components
        self.bm25 = None
        self.bm25_corpus = None
        self.bm25_tokenizer = None
        if config.retrieval_type in ["sparse", "hybrid"] and config.sparse_index_path:
            self._load_bm25()
        
        # Reranker (lazy-loaded)
        self.reranker = None
        if config.reranker:
            self._load_reranker()
    
    def _load_bm25(self):
        index_path = Path(self.config.sparse_index_path)
        
        # Load BM25 index
        self.bm25 = bm25s.BM25.load(str(index_path))
        
        # Load tokenizer config
        tok_config_file = index_path / "tokenizer_config.json"
        stem = True
        if tok_config_file.exists():
            with open(tok_config_file, 'r', encoding='utf-8') as f:
                tok_config = json.load(f)
                stem = tok_config.get("stem", True)
        
        self.stemmer = PorterStemmer() if stem else None
        self.bm25_tokenizer = self._tokenize_query

    def _tokenize_query(self, query: str):
        stemmer_fn = (lambda tokens: [self.stemmer.stem(t) for t in tokens]) if self.stemmer else None
        return bm25s.tokenize([query], stopwords="english", stemmer=stemmer_fn)
    
    def _reciprocal_rank_fusion(self, dense_ids: list[int], sparse_ids: list[int], k: int = 60) -> list[int]:
        """
        Reciprocal Rank Fusion: merge two ranked lists.
        RRF score = sum(1 / (k + rank)) for each document.
        """
        # Build score dictionary
        scores = {}
        
        # Dense ranks
        for rank, doc_id in enumerate(dense_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Sparse ranks
        for rank, doc_id in enumerate(sparse_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids
    
    def _normalise_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Min-max normalisation to [0, 1].
        If all scores are equal, return all zeros (avoid divide-by-zero).
        """
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _weighted_score_fusion(
        self,
        dense_ids: list[int],
        dense_scores: np.ndarray,
        sparse_ids: list[int],
        sparse_scores: np.ndarray,
        alpha: float = 0.5
    ) -> list[int]:
        """
        Weighted score fusion:
            score(doc) = alpha * dense_norm(doc) + (1 - alpha) * sparse_norm(doc)
        Documents present in only one list get a score of 0 from the other list.
        """
        # Normalise both score arrays to [0, 1]
        dense_norm = self._normalise_scores(np.array(dense_scores, dtype=np.float32))
        sparse_norm = self._normalise_scores(np.array(sparse_scores, dtype=np.float32))
        
        # Build score dictionaries
        dense_score_map = {doc_id: dense_norm[i] for i, doc_id in enumerate(dense_ids)}
        sparse_score_map = {doc_id: sparse_norm[i] for i, doc_id in enumerate(sparse_ids)}
        
        # Union of all doc IDs
        all_doc_ids = set(dense_score_map.keys()) | set(sparse_score_map.keys())
        
        # Blend scores
        fused_scores = {}
        for doc_id in all_doc_ids:
            d_score = dense_score_map.get(doc_id, 0.0)
            s_score = sparse_score_map.get(doc_id, 0.0)
            fused_scores[doc_id] = alpha * d_score + (1 - alpha) * s_score
        
        # Sort descending
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return sorted_ids
        
    def retrieve(self, query: str, k: int) -> list[int]:
        """
        Retrieve top-k chunks. If reranker is configured, retrieves a larger
        candidate set first, then reranks down to k.
        """
        # Determine how many candidates to retrieve before reranking
        if self.config.reranker and self.config.retrieve_k:
            # Reranking: retrieve more candidates, then rerank
            candidate_k = self.config.retrieve_k  # e.g., 100
            final_k = k  # e.g., 8
        else:
            candidate_k = k
            final_k = k
        
        # ---- 1. Base retrieval (dense / sparse / hybrid) ----
        if self.config.retrieval_type == "dense":
            emb = self.model.encode(
                [query],
                normalize_embeddings=self.config.normalize_embeddings
            ).astype('float32')
            _, indices = self.faiss_index.search(emb, candidate_k)
            candidate_ids = indices[0].tolist()
        
        elif self.config.retrieval_type == "sparse":
            query_tokens = self.bm25_tokenizer(query)
            results = self.bm25.retrieve(query_tokens, k=candidate_k)
            candidate_ids = results[0].tolist()
        
        elif self.config.retrieval_type == "hybrid":
            dense_k = min(candidate_k * 2, 500)
            sparse_k = min(candidate_k * 2, 500)
            
            emb = self.model.encode(
                [query],
                normalize_embeddings=self.config.normalize_embeddings
            ).astype('float32')
            dense_distances, dense_indices = self.faiss_index.search(emb, dense_k)
            dense_ids = dense_indices[0].tolist()
            dense_scores = dense_distances[0]
            
            query_tokens = self.bm25_tokenizer(query)
            sparse_result = self.bm25.retrieve(query_tokens, k=sparse_k, return_as="tuple")
            sparse_ids_raw, sparse_scores_raw = sparse_result
            sparse_ids = sparse_ids_raw[0].tolist() if isinstance(sparse_ids_raw[0], (list, np.ndarray)) else sparse_ids_raw[0]
            sparse_scores = sparse_scores_raw[0] if isinstance(sparse_scores_raw[0], (list, np.ndarray)) else sparse_scores_raw
            
            fusion_method = self.config.hybrid_fusion or "rrf"
            if fusion_method == "rrf":
                rrf_k = self.config.rrf_k if self.config.rrf_k else 60
                fused = self._reciprocal_rank_fusion(dense_ids, sparse_ids, k=rrf_k)
            elif fusion_method == "weighted":
                alpha = self.config.weighted_alpha if self.config.weighted_alpha is not None else 0.5
                fused = self._weighted_score_fusion(dense_ids, dense_scores, sparse_ids, sparse_scores, alpha=alpha)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            candidate_ids = fused[:candidate_k]
        
        else:
            raise ValueError(f"Unknown retrieval_type: {self.config.retrieval_type}")
        
        # ---- 2. Reranking (if configured) ----
        if self.config.reranker:
            top_n = self.config.rerank_top_n or final_k
            reranked_ids, _, _ = self.rerank(query, candidate_ids, top_n=top_n)
            return reranked_ids
        
        return candidate_ids[:final_k]

    def _load_reranker(self):
        model_map = {
            "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
            "bge-reranker-base": "BAAI/bge-reranker-base",
            "bge-reranker-large": "BAAI/bge-reranker-large",
        }
        full_name = model_map.get(self.config.reranker, self.config.reranker)
        
        # Auto-detect best device
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        logging.info(f"Loading reranker {full_name} on device={device}")
        self.reranker = CrossEncoder(full_name, max_length=512, device=device)

    def rerank(self, query: str, candidate_ids: list[int], top_n: int) -> tuple[list[int], list[float], float]:
        """
        Rerank candidate chunks using a cross-encoder.
        
        Returns:
            - reranked_ids: top_n chunk IDs after reranking
            - reranked_scores: corresponding scores
            - latency_ms: reranking time in milliseconds
        """
        if self.reranker is None:
            raise ValueError("Reranker not loaded. Set 'reranker' in config.")
        
        # Prepare (query, passage) pairs
        pairs = []
        for cid in candidate_ids:
            passage = self.metadata[cid]['text']
            pairs.append((query, passage))
        
        # Score all pairs
        start = time.perf_counter()
        scores = self.reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Sort by score descending
        sorted_pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        top_pairs = sorted_pairs[:top_n]
        
        reranked_ids = [p[0] for p in top_pairs]
        reranked_scores = [float(p[1]) for p in top_pairs]
        
        return reranked_ids, reranked_scores, latency_ms

    def retrieve_with_latency(self, query: str, k: int) -> dict:
        """
        Same as retrieve() but returns detailed latency breakdown.
        Useful for ablation runs that need per-stage timings.
        """
        import time
        result = {"retrieved_ids": [], "retrieval_ms": 0.0, "rerank_ms": 0.0}
        
        # Base retrieval
        start = time.perf_counter()
        if self.config.reranker:
            candidate_k = self.config.retrieve_k or 100
        else:
            candidate_k = k
        
        # (Reuse the base retrieval logic – simplest is to call retrieve() with reranker disabled internally)
        # For simplicity, we call retrieve() with a monkey-patched config; cleaner approach below:
        original_reranker = self.config.reranker
        self.config.reranker = None
        base_ids = self.retrieve(query, k=candidate_k)
        self.config.reranker = original_reranker
        result["retrieval_ms"] = (time.perf_counter() - start) * 1000
        
        # Reranking
        if self.config.reranker:
            start = time.perf_counter()
            top_n = self.config.rerank_top_n or k
            reranked_ids, _, _ = self.rerank(query, base_ids, top_n=top_n)
            result["rerank_ms"] = (time.perf_counter() - start) * 1000
            result["retrieved_ids"] = reranked_ids
        else:
            result["retrieved_ids"] = base_ids[:k]
        
        return result