# GroundedAI

**Hybrid retrieval RAG over Indian financial regulation — built to be measured, not demoed.**

A question-answering system over a few hundred RBI circulars and Master Directions.
Every design choice in it is an experiment with a number attached: dense vs sparse vs
hybrid retrieval, four embedding models, three FAISS index types, reranking on and off,
chunk sizes, and Matryoshka embedding truncation. The measurements are the point. The
chatbot is a side effect.

Regulatory text was chosen deliberately because it is hostile to naive RAG:

- **Exact strings matter.** "Circular DBR.No.BP.BC.45/21.04.048/2018-19" is a token soup
  that vector search handles badly and keyword search nails. This is why the retriever is
  hybrid rather than purely dense.
- **Documents supersede each other.** An answer that is correct as of 2019 and withdrawn
  in 2021 is a wrong answer. Metadata filtering is a correctness requirement, not a feature.
- **The PDFs are a mess.** Multi-column layouts, tables that carry the actual rule, and
  scanned pages with no text layer. Hence a three-tier parse ladder instead of one parser.
- **Confident wrong answers are the worst outcome.** A compliance answer that is invented
  is more dangerous than no answer, so there is an explicit abstention gate and every
  citation span is verified against its source chunk.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Environment, model cache, `doctor.py`, skeleton | **Complete** |
| 1 | Ingestion: parse ladder, chunking, dedup, indexes | Not started |
| 2 | Golden set: 180 hand-verified questions | Not started |
| 3 | Retrieval: hybrid, rerank, and the ablation matrix | Not started |
| 4 | Generation: prompting, validation, guardrails | Not started |
| 5 | Evaluation: RAGAS, LLM judge, tracing, CI gate | Not started |

## Results

Populated by the ablation runner (`src/hybrid_rag/evaluate/ablation.py`) against the
180-question golden set. Empty until Phase 3 — an empty table is more honest than a
plausible one.

| Configuration | Recall@10 | nDCG@10 | MRR | Faithfulness | p95 latency |
|---|---|---|---|---|---|
| _pending Phase 3_ | | | | | |

## Quickstart

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), and Tesseract
(`brew install tesseract`).

```bash
git clone https://github.com/partyushsharma/GroundedAI.git
cd GroundedAI

uv sync                                   # resolves from uv.lock
uv run python -m spacy download en_core_web_sm   # needed by Presidio

cp .env.example .env                      # then add your two free API keys

uv run python scripts/fetch_models.py     # ~4.6 GB of models into the HF cache
uv run python doctor.py                   # must print PASS
```

`doctor.py` is the contract. It verifies the interpreter, Tesseract and its language
data, that `.env` exists and is **not** tracked by git, that all twenty packages import,
that FAISS and torch coexist without the macOS OpenMP abort, that every model loads from
cache *and produces sane similarities*, and that both LLM providers answer. It exits
non-zero on any failure or any warning. Run it before you trust a result.

## Architecture

Data flows top to bottom.

| # | Layer | Component | What it does | Tech | Phase |
|---|---|---|---|---|---|
| 1 | Source | PDF corpus | A few hundred RBI circulars and Master Directions downloaded to disk. Deliberately messy: multi-column, tables, some scanned. | requests / manual download | Phase 1 |
| 2 | Ingestion | Parse ladder | Turns each PDF into clean text. Tries the fast parser first; falls back to a smarter one, then to OCR, only when needed. | PyMuPDF -> Docling -> Tesseract | Phase 1 |
| 3 | Ingestion | Chunker | Cuts long documents into small passages. Too small loses meaning, too big blurs the search. You test several strategies. | LangChain splitters, custom | Phase 1 & 3 |
| 4 | Ingestion | Metadata builder | Attaches facts to each chunk: page, section, date, circular number, whether it has been superseded. Decides what you can filter on later. | Pydantic | Phase 1 |
| 5 | Ingestion | Deduplicator | Removes near-identical chunks. Regulatory documents repeat boilerplate constantly and duplicates poison your results. | MinHash (datasketch) | Phase 1 |
| 6 | Indexing | Embedder | Converts each chunk of text into a list of numbers (a vector) that captures its meaning. | sentence-transformers, BGE | Phase 1 |
| 7 | Indexing | Dense index (FAISS) | Stores vectors and finds the closest ones fast. Library, not a database — you manage metadata yourself. | FAISS (Flat / HNSW / IVF) | Phase 1 & 3 |
| 8 | Indexing | Dense index (Chroma) | Same job, but a real database with metadata filtering built in. Kept alongside FAISS so you can compare filtering behaviour. | ChromaDB | Phase 1 & 3 |
| 9 | Indexing | Sparse index | Old-fashioned keyword search. Beats vectors on exact strings like circular numbers and section references. | bm25s | Phase 3 |
| 10 | Indexing | Index lifecycle | Version, rebuild, and swap indexes safely when you change the embedding model. Rarely built — big interview point. | Custom + JSON manifests | Phase 5 |
| 11 | Query | Query transformer | Rewrites the user's question before searching: expand it, split it up, or generate a fake answer to search with. | LiteLLM + prompts | Phase 3 |
| 12 | Retrieval | Hybrid retriever | Runs vector search and keyword search at the same time, then merges the two ranked lists into one. | FAISS + bm25s + RRF | Phase 3 |
| 13 | Retrieval | Reranker | Takes the top 100 results and re-scores them with a slower, much more accurate model. Keeps the best 8. | bge-reranker-v2-m3 | Phase 3 |
| 14 | Retrieval | Diversity filter | Stops all 8 results being near-copies of each other or all from one document. | MMR + source cap | Phase 3 |
| 15 | Generation | Prompt builder | Assembles the final prompt. Order matters — models pay less attention to the middle of a long prompt. | Custom templates | Phase 4 |
| 16 | Generation | LLM caller | Sends the prompt to a free model, with automatic fallback to a second provider if the first is rate-limited. | LiteLLM -> Groq / Gemini | Phase 4 |
| 17 | Generation | Output validator | Forces the answer into a fixed JSON shape and checks every quoted citation really appears in the source chunk. | Pydantic + span check | Phase 4 |
| 18 | Generation | Abstention gate | Makes the system say 'I don't know' when the retrieved passages are weak, instead of inventing an answer. | Score threshold | Phase 4 |
| 19 | Guardrails | Injection defence | Protects against a document in the corpus containing text that tries to hijack the model. | Delimiters + code-level rules | Phase 4 |
| 20 | Guardrails | PII scanner | Detects and redacts personal data on the way in and on the way out. | Presidio | Phase 4 |
| 21 | Evaluation | Golden set | 180 questions with known correct answers and known correct source chunks. The ruler you measure everything against. | JSONL, hand-reviewed | Phase 2 |
| 22 | Evaluation | Retrieval metrics | Scores whether the right passages were found. Needs no AI model, so it is free and unlimited. | Custom Python | Phase 2 |
| 23 | Evaluation | Generation metrics | Scores whether the written answer is faithful to the passages. Costs API calls, so used sparingly. | RAGAS | Phase 5 |
| 24 | Evaluation | LLM judge | An AI grading the answers — then you check the grader against your own labels before trusting it. | Gemini + Cohen's kappa | Phase 5 |
| 25 | Evaluation | Failure classifier | For each wrong answer, decides: did search fail, or did the model ignore what it was given? Different bugs, different fixes. | Custom Python | Phase 4 |
| 26 | Evaluation | Ablation runner | Runs every configuration automatically and writes the results table into your README. | Custom + YAML configs | Phase 3 |
| 27 | Ops | Tracing | Records every step of every query so you can see where time and money go. | Langfuse (Docker) | Phase 5 |
| 28 | Ops | CI gate | Runs a short test suite on every code change and blocks the change if quality drops. | GitHub Actions | Phase 5 |
| 29 | Serving | API | A small web service wrapping the pipeline so it can be demoed. | FastAPI (.NET in Project 2) | Phase 5 |

## Repository layout

```
doctor.py                  environment health check -- run this first
scripts/fetch_models.py    downloads the ~4.6 GB model cache
configs/                   YAML ablation configs (layer 26)
data/
  raw/                     source PDFs (git-ignored)
  processed/               parsed text + per-page parse quality (git-ignored)
  manifest.jsonl           tracked: URL, date and hash of every source file
indexes/                   FAISS + Chroma artefacts (git-ignored)
eval/
  golden/                  the 180-question golden set (tracked)
  results/                 ablation output tables (tracked)
src/hybrid_rag/
  ingest/                  parse ladder, chunking, dedup, ChunkMeta  (layers 2-5)
  index/                   embedder, FAISS, Chroma, BM25, lifecycle  (layers 6-10)
  retrieve/                query transform, hybrid + RRF, rerank, MMR (layers 11-14)
  generate/                prompt, LLM call, output validation, abstention (15-18)
  guardrails/              injection defence, PII scanning            (layers 19-20)
  evaluate/                metrics, LLM judge, failure classifier, ablation (21-26)
  api/                     FastAPI serving layer                      (layer 29)
tests/

`.github/workflows/` is intentionally absent until Phase 5: GitHub rejects any push
that touches that path unless the token carries the `workflow` scope, and an empty
placeholder there is not worth the friction.

The corpus and the indexes stay out of git deliberately: `data/manifest.jsonl` records the
URL, date and hash of every source document, so the corpus is reproducible from a tracked
file rather than committed as a few hundred megabytes of binary.

## Tech stack

Everything is free, and everything local-first where the workload is large enough for API
pricing to matter. The "why this one" column is the part worth reading.

| Tool | Category | Purpose here | Why this one |
|---|---|---|---|
| Python 3.12 | Language | All ingestion, retrieval and evaluation code. | The entire AI ecosystem is Python-first. |
| uv | Package manager | Installs packages and manages the virtual environment. | Resolves this large dependency tree in seconds; pip takes minutes. |
| PyMuPDF | PDF parsing | First-attempt text extraction from PDFs. Very fast. | Fastest option; handles the ~70% of pages that are clean. |
| Docling | PDF parsing | Second attempt for hard pages — understands columns, tables and reading order. | Layout-aware and open source; LlamaParse is paid. |
| Tesseract | OCR | Last resort for scanned pages with no text layer. | The standard open-source OCR engine. |
| sentence-transformers | Embeddings | Turns text into vectors. Runs the BGE / E5 models locally. | Local means no API cost and no rate limit on the biggest workload. |
| BAAI/bge-base-en-v1.5 | Embedding model | Your default text-to-vector model. 768 dimensions. | Strong quality-to-size ratio; runs comfortably on a laptop. |
| intfloat/e5-large-v2 | Embedding model | Larger comparison model for the embedding ablation. | Tests whether a bigger model is worth the extra storage. |
| nomic-embed-text-v1.5 | Embedding model | Supports Matryoshka truncation (768 -> 256 -> 128 dims). | Lets you measure the accuracy-vs-storage trade-off directly. |
| bge-reranker-v2-m3 | Reranker | Re-scores the top 100 search results much more accurately. | Best open cross-encoder; Cohere Rerank is paid. |
| FAISS | Vector index | Fast similarity search. You tune HNSW and IVF parameters on it. | Named in most job descriptions; exposes the low-level knobs you want to demonstrate. |
| ChromaDB | Vector database | Second vector store, with metadata filtering built in. | Enables the pre-filter vs post-filter comparison against FAISS. |
| bm25s | Keyword search | Classic keyword ranking. Half of your hybrid search. | Much faster than rank-bm25, same interface. |
| datasketch | Deduplication | MinHash near-duplicate detection before indexing. | Standard, fast, well documented. |
| LiteLLM | LLM gateway | One interface to all model providers, with automatic fallback. | Free tiers get deprecated without notice; this stops a run dying. |
| Groq | LLM provider | Fast generation with open models. ~1,000 requests/day free. | Fastest free inference available. |
| Google AI Studio | LLM provider | Long-context model for drafting the golden set and judging answers. | 1M token context reads a whole document at once. No card required. |
| OpenRouter | LLM provider | Third fallback lane when the first two are rate-limited. | Wide free model selection. |
| Pydantic | Validation | Defines the metadata schema and forces answers into a fixed JSON shape. | Standard in Python; gives type safety like C# classes. |
| RAGAS | Evaluation | Scores faithfulness, answer relevancy, context precision and recall. | The most-named RAG eval framework in job postings. |
| DeepEval | Evaluation | Secondary eval library; useful for test-style assertions. | Adds pytest-style eval assertions. |
| Langfuse | Observability | Records a trace of every query so you can see time and cost per stage. | Self-hostable in Docker; LangSmith is paid. |
| Presidio | PII / guardrails | Finds and redacts personal data in and out. | Microsoft-maintained; carries DPDP-compliance vocabulary. |
| FastAPI | Serving | Wraps the pipeline in a small web API for demoing. | Minimal boilerplate; auto-generates API docs. |
| pytest | Testing | Unit tests for metrics and guardrails. | Python standard. |
| GitHub Actions | CI | Runs a short eval suite on every change and blocks quality regressions. | Free for public repositories; the CI gate is a strong portfolio signal. |
| Docker Desktop | Infrastructure | Runs Langfuse and Postgres locally. | Only realistic way to self-host these. |
| Git + GitHub | Version control | Hosts the repo. Recruiters read it. | The portfolio artifact itself. |

## Method notes

**The measuring tool comes before the thing measured.** Phase 2 — writing and hand-verifying
180 questions with known-correct answers *and* known-correct source chunks — happens before
any retrieval tuning. Without it every number produced later is unfalsifiable, and the
numbers are the entire point.

**Retrieval failure and generation failure are different bugs.** When an answer is wrong,
`evaluate/failure_classifier.py` decides whether the right passage was never retrieved or
was retrieved and then ignored. They have different fixes, and conflating them is how
people end up tuning prompts to fix a search problem.

**Free tiers disappear without notice.** Every model call goes through LiteLLM with a
Groq → Gemini → OpenRouter fallback chain, so a provider deprecating a model mid-run
degrades instead of killing a four-hour evaluation.

## License

MIT
