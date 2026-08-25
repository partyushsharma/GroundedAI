"""Ingestion: PDF -> clean, chunked, deduplicated text with metadata.

Architecture layers 2-5. Built in Phase 1.
  parse.py   -- the 3-tier parse ladder: PyMuPDF -> Docling -> Tesseract (1.2-1.5)
  schema.py  -- ChunkMeta Pydantic model, incl. tenant_id / acl placeholders (1.6)
  chunk.py   -- 512-token chunking with 10% overlap (1.7)
  dedupe.py  -- MinHash near-duplicate removal (1.8)"""
