import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from context_generator import ContextGenerator
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")
OUTPUT_FILE = Path("data/processed/chunks_contextual.jsonl")
# Cache the raw document text so we don't re-parse the PDF for every chunk
DOC_CACHE = Path("data/cache/doc_texts")
DOC_CACHE.mkdir(parents=True, exist_ok=True)


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            c = json.loads(line)
            c["chunk_id"] = idx  # match the IDs used in FAISS/BM25 indexes
            chunks.append(c)
    return chunks


def load_doc_text(pdf_name: str) -> str:
    """Load (or extract) the full document text for a given PDF."""
    safe = pdf_name.replace("/", "_")
    cache_file = DOC_CACHE / f"{safe}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    from ingest.parse import extract_pdf_with_page_tiers
    pdf_path = Path("data/raw") / pdf_name
    page_texts, _ = extract_pdf_with_page_tiers(pdf_path)
    full = "\n\n".join(t for t in page_texts if t.strip())
    cache_file.write_text(full, encoding="utf-8")
    return full


def main():
    chunks = load_chunks()
    logger.info(f"Loaded {len(chunks)} chunks")

    # Group chunks by document — this avoids re-parsing PDFs
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["meta"]["pdf_name"], []).append(c)

    logger.info(f"Chunks span {len(by_doc)} documents")

    generator = ContextGenerator()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        processed = 0
        for doc_name, doc_chunks in by_doc.items():
            logger.info(f"Processing {doc_name}: {len(doc_chunks)} chunks")
            doc_text = load_doc_text(doc_name)

            for c in doc_chunks:
                context = generator.generate_context(
                    doc_id=doc_name,
                    chunk_index=c["chunk_id"],
                    chunk_text=c["text"],
                    document_text=doc_text,
                )

                # Prepend context to text — this is what gets embedded
                enriched_text = f"{context}\n\n{c['text']}" if context else c["text"]

                record = {
                    "text": enriched_text,
                    "original_text": c["text"],
                    "context": context,
                    "meta": {**c["meta"], "chunk_id": c["chunk_id"]},
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                processed += 1
                if processed % 500 == 0:
                    logger.info(f"Progress: {processed}/{len(chunks)} | {generator.stats()}")

    logger.info(f"Done. Final stats: {generator.stats()}")
    logger.info(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()