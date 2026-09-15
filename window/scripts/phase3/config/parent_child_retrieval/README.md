The problem: Small chunks (~512 tokens) are great for searching (precise matching), but bad for generation — the LLM often gets a fragment that lacks surrounding context. Large chunks are the opposite.

The solution: Search over small child chunks but hand the LLM the larger parent passage that contains them. You get precision at search time and rich context at generation time.

Query → search over child chunks (512 tok) → map to parent passage (~2048 tok) → rerank on child text → send parent to LLM

This is a retrieval‑only task for now (generation comes in Phase 4). For Phase 3, "parent" just means the returned chunk IDs point to the parent text, and reranking/scoring uses whatever text you choose.

1. Design the parent-child mapping
Two common parent definitions:

Variant	Parent size	When to use
Fixed window	3–5 consecutive children merged (e.g., 2048–2560 tokens)	Simple, works well for regulatory text
Section-based	The whole section the child belongs to	Better semantically, but requires section detection
For this task, use fixed window: each child’s parent = itself + N neighbours on each side.

Parameters:

Child chunk size: 512 tokens (from Task 1.7)

Overlap: 51 tokens (10%)

Parent window: 3 children → ~1500 tokens (with overlaps) or 5 children → ~2500 tokens

Start with 3 children — enough context without blowing the LLM context window.