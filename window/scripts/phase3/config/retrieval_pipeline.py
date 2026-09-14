# Task 3.3 ####################################################
# Now you'll combine dense and sparse retrieval using Reciprocal Rank Fusion (RRF). This is a simple but powerful method that merges two ranked lists 
# based on their positions, ignoring the raw scores.
# The key insight: dense and sparse scores are on different scales – averaging them is wrong.
#  RRF sidesteps this by using only the rank positions.
################################################################

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import bm25s
from nltk.stem import PorterStemmer

from config_loader import RetrievalConfig


class RetrievalPipeline:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.faiss_index = faiss.read_index(config.faiss_index_path)
        with open("data/indexes/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # BM25 components (lazy-loaded)
        self.bm25 = None
        self.stemmer = None
        
        # Load BM25 if needed
        if config.retrieval_type in ["sparse", "hybrid"] and config.sparse_index_path:
            self._load_bm25()
    
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
        if self.config.retrieval_type == "dense":
            emb = self.model.encode(
                [query], 
                normalize_embeddings=self.config.normalize_embeddings
            ).astype('float32')
            distances, indices = self.faiss_index.search(emb, k)
            return indices[0].tolist()
        
        elif self.config.retrieval_type == "sparse":
            if self.bm25 is None:
                raise ValueError("BM25 index not loaded. Check sparse_index_path.")
            query_tokens = self._tokenize_query(query)
            results = self.bm25.retrieve(query_tokens, k=k, return_as="tuple")
            # bm25s returns (doc_ids, scores) when return_as="tuple"
            doc_ids, _ = results
            return doc_ids[0].tolist()
        
        elif self.config.retrieval_type == "hybrid":
            if self.bm25 is None:
                raise ValueError("BM25 index not loaded for hybrid retrieval.")
            
            dense_k = min(k * 2, 200)
            sparse_k = min(k * 2, 200)
            
            # ---- Dense retrieval (with scores) ----
            emb = self.model.encode(
                [query], 
                normalize_embeddings=self.config.normalize_embeddings
            ).astype('float32')
            dense_distances, dense_indices = self.faiss_index.search(emb, dense_k)
            dense_ids = dense_indices[0].tolist()
            dense_scores = dense_distances[0]  # cosine similarity scores
            
            # ---- Sparse retrieval (with scores) ----
            query_tokens = self._tokenize_query(query)
            sparse_result = self.bm25.retrieve(query_tokens, k=sparse_k, return_as="tuple")
            sparse_ids_raw, sparse_scores_raw = sparse_result
            sparse_ids = sparse_ids_raw[0].tolist() if isinstance(sparse_ids_raw[0], (list, np.ndarray)) else sparse_ids_raw[0]
            sparse_scores = sparse_scores_raw[0] if isinstance(sparse_scores_raw[0], (list, np.ndarray)) else sparse_scores_raw
            
            # ---- Fusion ----
            fusion_method = self.config.hybrid_fusion or "rrf"
            
            if fusion_method == "rrf":
                rrf_k = self.config.rrf_k if self.config.rrf_k else 60
                fused_ids = self._reciprocal_rank_fusion(dense_ids, sparse_ids, k=rrf_k)
            elif fusion_method == "weighted":
                alpha = self.config.weighted_alpha if self.config.weighted_alpha is not None else 0.5
                fused_ids = self._weighted_score_fusion(
                    dense_ids, dense_scores,
                    sparse_ids, sparse_scores,
                    alpha=alpha
                )
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            return fused_ids[:k]
        
        else:
            raise ValueError(f"Unknown retrieval_type: {self.config.retrieval_type}")