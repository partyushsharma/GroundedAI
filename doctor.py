#!/usr/bin/env python
"""Environment health check for the hybrid-rag project. Task 0.5.

Run this before starting work, and any time a result looks wrong:

    python doctor.py

The point is to make "is my environment broken?" a question you answer in
sixty seconds instead of a suspicion you carry through a whole phase. When a
Phase 3 ablation produces a number that makes no sense, you should already
know the environment is not the reason.

Every check reports one of four states:

    PASS       checked and healthy
    ATTENTION  works, but needs you to do something -- prints an ATTENTION line
    FAIL       broken, blocks work
    SKIP       deliberately deferred to a later phase; not a problem

Exit code is 0 only when there are zero FAIL and zero ATTENTION results.
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Must precede the torch / faiss imports: both vendor their own libomp and
# macOS aborts the process when the second one loads. Also set in .env, but
# doctor.py has to be safe to run before anything sources that.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Model loading is chatty. This report is meant to be read, so silence the
# progress bars and info logging that would otherwise interleave with it.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent

PASS, ATTENTION, FAIL, SKIP = "PASS", "ATTENTION", "FAIL", "SKIP"


@dataclass
class Result:
    state: str
    detail: str
    advice: str = ""
    extra: list[str] = field(default_factory=list)


CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    """Register a check. Order of definition is order of execution."""

    def wrap(fn):
        CHECKS.append((name, fn))
        return fn

    return wrap


def run(*cmd: str) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return p.returncode, (p.stdout + p.stderr).strip()


# --------------------------------------------------------------------------
# 1-3. Interpreter, system tools, secrets
# --------------------------------------------------------------------------


@check("interpreter")
def _interpreter() -> Result:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) != (3, 12):
        return Result(FAIL, f"Python {ver}, expected 3.12.x",
                      "The pinned version in .python-version is 3.12. Run: uv venv --python 3.12")
    venv = Path(sys.prefix).resolve()
    if venv != (PROJECT_ROOT / ".venv").resolve():
        return Result(ATTENTION, f"Python {ver}, but running from {venv}",
                      "You are outside the project venv. Use 'uv run python doctor.py' "
                      "or activate it: source .venv/bin/activate")
    return Result(PASS, f"Python {ver} in .venv")


@check("tesseract")
def _tesseract() -> Result:
    if shutil.which("tesseract") is None:
        return Result(FAIL, "not on PATH", "brew install tesseract")
    rc, out = run("tesseract", "--version")
    ver = out.splitlines()[0].replace("tesseract", "").strip() if rc == 0 else "?"
    rc, langs = run("tesseract", "--list-langs")
    if "eng" not in langs.split():
        return Result(FAIL, f"{ver}, but 'eng' is missing",
                      "brew install tesseract-lang")
    n = len([l for l in langs.splitlines()[1:] if l.strip()])
    return Result(PASS, f"{ver}, 'eng' present ({n} languages)")


@check("secrets")
def _secrets() -> Result:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    missing = [k for k in ("GROQ_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)]
    if missing:
        return Result(FAIL, f"missing from .env: {', '.join(missing)}",
                      "Add them to .env. See .env.example for the expected names.")

    # A tracked .env is the one mistake in this project that cannot be undone
    # by a later commit, so assert it on every run, not just at task 0.4.
    rc, _ = run("git", "ls-files", "--error-unmatch", ".env")
    if rc == 0:
        return Result(FAIL, "both keys present, but .env IS TRACKED BY GIT",
                      "Your keys are in git history. Run: git rm --cached .env "
                      "-- then rotate both keys, because they must be treated as leaked.")
    return Result(PASS, "both keys present, .env untracked")


# --------------------------------------------------------------------------
# 4-6. Packages, the OpenMP landmine, compute device
# --------------------------------------------------------------------------

# Every tool named in the Tech Stack sheet, as import name -> what it is for.
PACKAGES = {
    "torch": "tensors / model runtime",
    "faiss": "dense index",
    "chromadb": "vector database",
    "bm25s": "keyword search",
    "pymupdf": "parse tier 1",
    "pytesseract": "parse tier 3, OCR",
    "docling": "parse tier 2",
    "sentence_transformers": "embeddings",
    "FlagEmbedding": "BGE model helpers",
    "datasketch": "MinHash dedup",
    "langchain_text_splitters": "chunking",
    "litellm": "LLM gateway",
    "pydantic": "schemas / validation",
    "ragas": "generation metrics",
    "deepeval": "eval assertions",
    "langfuse": "tracing",
    "presidio_analyzer": "PII detection",
    "spacy": "NLP models behind Presidio",
    "einops": "required by nomic-embed remote code",
    "fastapi": "serving",
    "pytest": "tests",
}


@check("packages")
def _packages() -> Result:
    import importlib

    broken = []
    for mod, why in PACKAGES.items():
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{mod} ({why}): {type(exc).__name__}: {exc}")
    if broken:
        return Result(FAIL, f"{len(broken)} of {len(PACKAGES)} failed to import",
                      "uv sync", extra=broken)
    return Result(PASS, f"all {len(PACKAGES)} import cleanly")


@check("faiss + torch")
def _faiss_torch() -> Result:
    # If faiss and torch each load their own libomp, macOS aborts the whole
    # process -- not an exception you can catch. Reaching the end of this
    # function at all is the result.
    import faiss
    import numpy as np
    import torch

    idx = faiss.IndexFlatIP(64)
    vecs = np.random.default_rng(0).random((32, 64), dtype="float32")
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    dist, ids = idx.search(vecs[:1], 3)
    if ids[0][0] != 0:
        return Result(FAIL, "index search did not return the query vector first",
                      "faiss is installed but returning wrong results.")
    t = torch.ones(4) @ torch.ones(4)
    if float(t) != 4.0:
        return Result(FAIL, "torch arithmetic is wrong", "Reinstall torch.")
    return Result(PASS, f"coexist without an OpenMP abort "
                        f"(faiss {faiss.__version__}, torch {torch.__version__})")


@check("device")
def _device() -> Result:
    import torch

    if torch.backends.mps.is_available():
        return Result(PASS, "mps (Apple GPU) available")
    if torch.cuda.is_available():
        return Result(PASS, "cuda available")
    return Result(ATTENTION, "cpu only",
                  "MPS is unavailable, so embedding the Phase 1 corpus (~40k chunks) "
                  "will take hours rather than minutes. Check that this is a native "
                  "arm64 Python, not one running under Rosetta.")


# --------------------------------------------------------------------------
# 7. Models: cached, loadable, and behaving sensibly
# --------------------------------------------------------------------------

RELATED = ("A bank must maintain a minimum capital adequacy ratio.",
           "Banks are required to hold a minimum ratio of regulatory capital.")
UNRELATED = "The chef seared the scallops in brown butter."


def _embedder_check(repo: str, dim: int, q_prefix: str = "", d_prefix: str = "",
                    trust_remote_code: bool = False) -> Result:
    """Load an embedder from cache and prove it produces meaningful vectors."""
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(
            repo,
            local_files_only=True,       # never silently download inside doctor
            trust_remote_code=trust_remote_code,
            device="cpu",                # tiny inputs; avoids MPS warm-up cost
        )
    except Exception as exc:  # noqa: BLE001
        return Result(FAIL, f"will not load: {type(exc).__name__}: {str(exc)[:160]}",
                      f"Run: python scripts/fetch_models.py  (repo {repo})")

    emb = model.encode(
        [q_prefix + RELATED[0], d_prefix + RELATED[1], d_prefix + UNRELATED],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if emb.shape[1] != dim:
        return Result(FAIL, f"got {emb.shape[1]} dimensions, expected {dim}",
                      "The cached weights are not the model we think they are.")

    near = float(emb[0] @ emb[1])
    far = float(emb[0] @ emb[2])
    if near <= far:
        return Result(FAIL, f"dim {dim}, but similarity is inverted "
                            f"(related {near:.3f} <= unrelated {far:.3f})",
                      "The model loaded but its embeddings are meaningless. "
                      "Suspect a bad pooling config or a truncated download.")

    detail = f"dim {dim}, related {near:.3f} > unrelated {far:.3f}"

    # nomic's whole reason for being in this project is Matryoshka truncation.
    if "nomic" in repo:
        small = model.encode([q_prefix + RELATED[0]], normalize_embeddings=True,
                             truncate_dim=256, show_progress_bar=False)
        if small.shape[1] != 256:
            return Result(FAIL, f"{detail}, but truncate_dim=256 gave {small.shape[1]}",
                          "Matryoshka truncation is broken; the Phase 3 "
                          "storage-vs-accuracy ablation depends on it.")
        detail += ", truncates to 256"

    return Result(PASS, detail)


@check("bge-base-en-v1.5")
def _bge() -> Result:
    return _embedder_check("BAAI/bge-base-en-v1.5", 768)


@check("e5-large-v2")
def _e5() -> Result:
    # E5 was trained with these exact prefixes; without them quality collapses.
    return _embedder_check("intfloat/e5-large-v2", 1024,
                           q_prefix="query: ", d_prefix="passage: ")


@check("nomic-embed-text-v1.5")
def _nomic() -> Result:
    return _embedder_check("nomic-ai/nomic-embed-text-v1.5", 768,
                           q_prefix="search_query: ", d_prefix="search_document: ",
                           trust_remote_code=True)


@check("bge-reranker-v2-m3")
def _reranker() -> Result:
    from sentence_transformers import CrossEncoder

    try:
        ce = CrossEncoder("BAAI/bge-reranker-v2-m3", local_files_only=True, device="cpu")
    except Exception as exc:  # noqa: BLE001
        return Result(FAIL, f"will not load: {type(exc).__name__}: {str(exc)[:160]}",
                      "Run: python scripts/fetch_models.py")

    query = "What capital adequacy ratio must banks maintain?"
    scores = ce.predict([(query, RELATED[1]), (query, UNRELATED)],
                        show_progress_bar=False)
    good, bad = float(scores[0]), float(scores[1])
    if good <= bad:
        return Result(FAIL, f"ranking is inverted (relevant {good:.2f} <= "
                            f"irrelevant {bad:.2f})",
                      "The reranker loaded but is not discriminating. Re-download it.")
    return Result(PASS, f"relevant {good:.2f} > irrelevant {bad:.2f}")


# --------------------------------------------------------------------------
# 8. Docling
# --------------------------------------------------------------------------


@check("docling")
def _docling() -> Result:
    import docling
    from docling.document_converter import DocumentConverter

    DocumentConverter()  # constructs lazily; does not fetch weights
    ver = getattr(docling, "__version__", "?")

    from huggingface_hub import constants

    hub = Path(constants.HF_HUB_CACHE)
    layout = list(hub.glob("models--ds4sd--docling-*")) + list(hub.glob("models--ds4sd--Docling*"))
    if not layout:
        return Result(SKIP, f"{ver} imports; layout/tableformer weights deferred "
                            f"to Phase 1 (task 1.3)")
    return Result(PASS, f"{ver}, layout/tableformer weights cached")


# --------------------------------------------------------------------------
# 9-10. LLM lanes and disk
# --------------------------------------------------------------------------

LANES = [
    ("groq/openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("gemini/gemini-3.6-flash", {}),
]


@check("llm lanes")
def _llm() -> Result:
    import logging

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    from litellm import completion

    up, down = [], []
    for model, extra in LANES:
        try:
            r = completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=300,  # reasoning models spend tokens before answering
                **extra,
            )
            if (r.choices[0].message.content or "").strip():
                up.append(model.split("/")[0])
            else:
                down.append(f"{model}: empty response")
        except Exception as exc:  # noqa: BLE001
            down.append(f"{model}: {type(exc).__name__}: {str(exc)[:90]}")

    if not up:
        return Result(FAIL, "no provider answered", extra=down,
                      advice="Both lanes are down. Check your keys and free-tier quota; "
                             "generation and the LLM judge cannot run without one.")
    if down:
        return Result(ATTENTION, f"{'/'.join(up)} answering, {len(down)} lane(s) down",
                      extra=down,
                      advice="One provider is unavailable. Fine for now, but LiteLLM has "
                             "no fallback left if the other rate-limits mid-run.")
    return Result(PASS, f"{'/'.join(up)} both answering")


@check("disk")
def _disk() -> Result:
    free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    if free < 20:
        return Result(ATTENTION, f"{free:.0f} GB free",
                      "Phase 1 adds a few hundred PDFs plus two indexes. Under 20 GB "
                      "is tight; free up space before task 1.1.")
    return Result(PASS, f"{free:.0f} GB free")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

WIDTH = 78


def quiet() -> None:
    """Silence library chatter that would break up the report."""
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    for name in ("transformers", "sentence_transformers", "LiteLLM", "httpx",
                 "chromadb", "docling", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    # Importing these emits its own warnings (e.g. the Hub's "unauthenticated
    # requests" notice), so the imports themselves have to be muffled.
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except Exception:  # noqa: BLE001 - cosmetic only, never fail the run
            pass
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    quiet()
    print("=" * WIDTH)
    print("hybrid-rag doctor".center(WIDTH))
    print(f"{PROJECT_ROOT}".center(WIDTH))
    print("=" * WIDTH)

    results: list[tuple[str, Result]] = []
    started = time.monotonic()

    tty = sys.stdout.isatty()

    for name, fn in CHECKS:
        if tty:
            print(f"  ... {name}".ljust(WIDTH), end="\r", flush=True)
        # Loading these models prints deprecation notices, hub warnings and
        # state-dict chatter. Swallow it, and replay it only if the check fails
        # -- at which point it is exactly what you need to see.
        noise = io.StringIO()
        try:
            with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
                res = fn()
        except Exception as exc:  # noqa: BLE001 - an unexpected crash is a FAIL
            import traceback

            res = Result(FAIL, f"{type(exc).__name__}: {str(exc)[:120]}",
                         advice=f"Unexpected error in the '{name}' check.",
                         extra=traceback.format_exc().splitlines()[-6:])
        if res.state == FAIL:
            tail = [l for l in noise.getvalue().splitlines() if l.strip()][-4:]
            res.extra = res.extra + [f"(output) {l}" for l in tail]
        results.append((name, res))
        print(f"[{res.state:<9}] {name:<24} {res.detail}".ljust(WIDTH))
        for line in res.extra:
            print(f"{'':13}  - {line}")

    fails = [(n, r) for n, r in results if r.state == FAIL]
    attn = [(n, r) for n, r in results if r.state == ATTENTION]
    passes = sum(1 for _, r in results if r.state == PASS)
    skips = sum(1 for _, r in results if r.state == SKIP)

    print("-" * WIDTH)
    print(f"{passes} pass, {len(attn)} attention, {len(fails)} fail, {skips} skipped "
          f"({time.monotonic() - started:.0f}s)")

    if attn or fails:
        print()
        for name, r in fails:
            print(f"ATTENTION: [{name}] {r.detail}")
            if r.advice:
                print(f"           {r.advice}")
        for name, r in attn:
            print(f"ATTENTION: [{name}] {r.detail}")
            if r.advice:
                print(f"           {r.advice}")

    print("-" * WIDTH)
    if fails:
        print("FAIL")
        return 1
    if attn:
        print("FAIL -- checks passed but attention is required (see above)")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
