import json
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_FILE = Path("data/processed/chunks_dedup.jsonl")
OUTPUT_FILE = Path("data/indexes/parent_child_map.json")

WINDOW = 2  # children on each side (parent = 1 + 2*WINDOW = 5 children)


def main():
    # Load chunks in order — chunk_id == line number (matches FAISS/BM25 indexes)
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            c = json.loads(line)
            c["chunk_id"] = idx
            chunks.append(c)

    logger.info(f"Loaded {len(chunks)} chunks")

    # Group by (pdf_name, page_number) so parents don't cross documents
    # Actually better: group by pdf_name only — parents can span pages within the same doc
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["meta"]["pdf_name"]].append(c)

    # Assign parent_id to every child
    mapping = {}  # child_id -> {parent_id, parent_text, parent_span: [start, end]}
    parents = {}  # parent_id -> {text, child_ids, pdf_name}

    parent_counter = 0

    for pdf_name, doc_chunks in by_doc.items():
        # Sort by chunk_id to preserve document order
        doc_chunks = sorted(doc_chunks, key=lambda x: x["chunk_id"])
        n = len(doc_chunks)

        # Sliding window over children
        # Parent i covers children [i - WINDOW, i + WINDOW]
        for i, child in enumerate(doc_chunks):
            start = max(0, i - WINDOW)
            end = min(n, i + WINDOW + 1)
            window_children = doc_chunks[start:end]

            # Parent text = concatenation of window children's text
            parent_text = "\n\n".join(wc["text"] for wc in window_children)
            child_ids = [wc["chunk_id"] for wc in window_children]

            # De-duplicate parents: use a key from the first+last child
            parent_key = f"{pdf_name}::{child_ids[0]}::{child_ids[-1]}"
            if parent_key not in parents:
                parents[parent_key] = {
                    "parent_id": parent_counter,
                    "text": parent_text,
                    "child_ids": child_ids,
                    "pdf_name": pdf_name,
                    "span": [child_ids[0], child_ids[-1]],
                }
                parent_counter += 1
            else:
                # Reuse existing parent
                pass

            mapping[child["chunk_id"]] = {
                "parent_key": parent_key,
                "parent_id": parents[parent_key]["parent_id"],
            }

    # Freeze parents into a list, re-index by integer ID
    parent_list = sorted(parents.values(), key=lambda p: p["parent_id"])

    logger.info(f"Built {len(parent_list)} unique parents for {len(chunks)} children")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "window": WINDOW,
            "child_to_parent": mapping,
            "parents": parent_list,
        }, f, ensure_ascii=False)
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()