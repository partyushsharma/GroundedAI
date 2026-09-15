import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import bm25s
from bm25s.tokenization import Tokenizer
from nltk.stem import PorterStemmer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT = Path("data/processed/chunks_contextual.jsonl")
FAISS_OUT = Path("data/indexes/faiss_contextual.index")
META_OUT = Path("data/indexes/metadata_contextual.json")
BM25_OUT_DIR = Path("data/indexes/bm25_contextual")
BM25_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ---- Load ----
texts, metas = [], []
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        texts.append(c["text"])
        metas.append({"text": c["text"], "original_text": c["original_text"],
                      "context": c["context"], "meta": c["meta"]})
logger.info(f"Loaded {len(texts)} contextual chunks")

# ---- FAISS ----
model = SentenceTransformer(MODEL_NAME)
batch_size = 128
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    emb = model.encode(batch, normalize_embeddings=True).astype("float32")
    embeddings.append(emb)
embeddings = np.vstack(embeddings)
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, str(FAISS_OUT))

with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False)
logger.info(f"FAISS written: {index.ntotal} vectors")

# ---- BM25 ----
stemmer = PorterStemmer()
stemmer_fn = lambda tokens: [stemmer.stem(t) for t in tokens]
corpus_tokens = bm25s.tokenize(texts, stopwords="english", stemmer=stemmer_fn)

bm25 = bm25s.BM25(corpus=texts)
bm25.index(corpus_tokens)
bm25.save(str(BM25_OUT_DIR))
with open(BM25_OUT_DIR / "tokenizer_config.json", "w") as f:
    json.dump({"stem": True}, f)
logger.info("BM25 written")