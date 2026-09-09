from pydantic import BaseModel, Field
from typing import Optional, Literal
import yaml
from pathlib import Path

class RetrievalConfig(BaseModel):
    experiment_name: str
    description: str
    
    embedding_model: str
    normalize_embeddings: bool = True
    
    index_type: Literal["flat", "hnsw", "ivf"] = "flat"
    faiss_index_path: str
    chroma_collection: Optional[str] = None
    
    retrieval_type: Literal["dense", "sparse", "hybrid"] = "dense"
    hybrid_fusion: Optional[Literal["rrf", "weighted"]] = None
    rrf_k: Optional[int] = 60
    weighted_alpha: Optional[float] = 0.5
    sparse_index_path: Optional[str] = None
    
    retrieve_k: int = 100
    reranker: Optional[Literal["bge-reranker-v2-m3"]] = None
    rerank_top_n: Optional[int] = 8
    
    query_transform: Optional[Literal["hyde", "multi_query", "decomposition"]] = None
    hyde_temperature: Optional[float] = 0.3
    num_query_variants: Optional[int] = 3
    
    diversity_filter: Optional[Literal["mmr"]] = None
    mmr_lambda: Optional[float] = 0.5
    
    k_values: list[int] = [1, 5, 10, 20]

def load_config(path: Path) -> RetrievalConfig:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return RetrievalConfig(**data)