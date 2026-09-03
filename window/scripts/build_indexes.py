#  This script will:
# Load the deduplicated chunks.
# Embed them using a chosen model (BGE‑base‑en‑v1.5 is the default from your tech stack).
# Build a FAISS index (Flat IP for cosine similarity) with metadata mapping.
# Build a Chroma collection with the same data.
# Optionally save the index and metadata to disk for later use.
# Provide a test query function.

import json
from pathlib import Path
import numpy as np
import faiss
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")
FAISS_INDEX_PATH = Path("data/indexes/faiss.index")
METADATA_PATH = Path("data/indexes/metadata.json")
CHROMA_PATH = Path("data/indexes/chroma_db")

# 1. Load chunks
chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))
logger.info(f"Loaded {len(chunks)} chunks")

MODEL_NAME = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(MODEL_NAME)
# BGE models work best when you add a query prefix for queries, but for indexing we don't need it.


# 2. Embed all chunks (batch processing)
BATCH_SIZE = 128
embeddings = []
for i in range(0, len(chunks), BATCH_SIZE):
    batch = [chunk['text'] for chunk in chunks[i:i+BATCH_SIZE]]
    batch_emb = model.encode(batch, normalize_embeddings=True)  # normalize for cosine similarity
    embeddings.append(batch_emb)
    logger.info(f"Embedded batch {i//BATCH_SIZE + 1}")
embeddings = np.vstack(embeddings).astype('float32')
logger.info(f"Embedding shape: {embeddings.shape}")


# 3. Build FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)   # Inner product
faiss_index.add(embeddings)
logger.info(f"FAISS index built with {faiss_index.ntotal} vectors")

# Save FAISS index and metadata (metadata is a list of dicts, one per chunk)
FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
with open(METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")


# 4. Build Chroma collection
# Create persistent Chroma client
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

# Create or get collection. Use the same embedding function.
class ChromaEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model):
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, normalize_embeddings=True).tolist()

    def name(self) -> str:
        return "bge-base-en-v1.5"

chroma_ef = ChromaEmbeddingFunction(model)

collection = chroma_client.get_or_create_collection(
    name="rbi_chunks",
    embedding_function=chroma_ef,
    metadata={"hnsw:space": "cosine"}   # use cosine similarity
)

# Chroma requires ids, documents, and metadata (optional)
ids = [str(i) for i in range(len(chunks))]
documents = [chunk['text'] for chunk in chunks]
# We can also store chunk['meta'] as metadata, but Chroma metadata must be flat dicts (not nested).
# We'll flatten the meta dict (or keep only some fields)
metadatas = []
for chunk in chunks:
    meta = chunk['meta'].copy()
    # Remove any nested dicts if present; or keep as strings.
    # For simplicity, we convert all values to strings.
    flat_meta = {k: str(v) for k, v in meta.items() if not isinstance(v, (dict, list))}
    metadatas.append(flat_meta)

# Add in batches to avoid timeout
BATCH_SIZE_CHROMA = 1000
for i in range(0, len(chunks), BATCH_SIZE_CHROMA):
    collection.add(
        ids=ids[i:i+BATCH_SIZE_CHROMA],
        documents=documents[i:i+BATCH_SIZE_CHROMA],
        metadatas=metadatas[i:i+BATCH_SIZE_CHROMA],
    )
    logger.info(f"Added batch {i//BATCH_SIZE_CHROMA + 1} to Chroma")

logger.info(f"Chroma collection ready with {collection.count()} documents")


# Add a function at the end of the script (or a separate test script) that:
# Takes a query string.
# Embeds it using the same model (with normalize_embeddings=True).
# Searches FAISS (top‑k) and retrieves corresponding metadata.
# Queries Chroma (top‑k) and retrieves results.
# Prints the top results from both, side‑by‑side.
def test_query(query: str, k: int = 5):
    # Embed query
    query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    
    # FAISS search
    distances, indices = faiss_index.search(query_emb, k)
    print("\n=== FAISS Results ===")
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        print(f"{i+1}. Score: {distances[0][i]:.4f}")
        print(f"   {chunk['text'][:200]}...")
        print(f"   Source: {chunk['meta']['pdf_name']}, page {chunk['meta']['page_number']}\n")
    
    # Chroma query
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    print("\n=== Chroma Results ===")
    for i in range(k):
        print(f"{i+1}. Score: {1 - results['distances'][0][i]:.4f}")  # Chroma returns distance, we convert to similarity
        print(f"   {results['documents'][0][i][:200]}...")
        meta = results['metadatas'][0][i]
        print(f"   Source: {meta.get('pdf_name', 'unknown')}, page {meta.get('page_number', 'unknown')}\n")

# TESTING
if __name__ == "__main__":
    test_query("What are the guidelines for foreign exchange remittances?")