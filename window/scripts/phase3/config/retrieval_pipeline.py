# src/retrieval_pipeline.py
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config_loader import RetrievalConfig

class RetrievalPipeline:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.index = faiss.read_index(config.faiss_index_path)
        with open("data/indexes/metadata.json", 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)   # list of chunks with 'text' and 'meta'
        
    def retrieve(self, query: str, k: int) -> list[int]:
        # 1. Embed query
        emb = self.model.encode([query], normalize_embeddings=self.config.normalize_embeddings).astype('float32')
        # 2. Search (dense only for now)
        distances, indices = self.index.search(emb, k)
        return indices[0].tolist()