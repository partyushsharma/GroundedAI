import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch, faiss, chromadb, bm25s, fitz          # fitz = pymupdf
import sentence_transformers, ragas, litellm, pydantic
import numpy as np

print("torch     ", torch.__version__)
print("faiss     ", faiss.__version__)
print("chromadb  ", chromadb.__version__)

# The real test: faiss and torch coexisting without an OpenMP crash
idx = faiss.IndexFlatIP(64)
idx.add(np.random.rand(10, 64).astype("float32"))
print("faiss+torch OK —", idx.ntotal, "vectors")

dev = ("mps" if torch.backends.mps.is_available()
       else "cuda" if torch.cuda.is_available() else "cpu")
print("device    ", dev)
print("\nALL PACKAGES OK")