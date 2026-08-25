"""
doctor.py  –  Environment health-check for Hybrid-RAG
Run:  python doctor.py
Expected outcome: every line says PASS; no ATTENTION lines.
"""

import os, sys, subprocess, importlib, textwrap

# ── Load .env first so keys are available ──────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ATTENTION  python-dotenv not installed – run: uv pip install python-dotenv")
    sys.exit(1)

PASS = "PASS      "
ATTN = "ATTENTION "
issues = []

def ok(label, detail=""):
    msg = f"{PASS} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)

def fail(label, detail=""):
    msg = f"{ATTN} {label}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    issues.append(label)

# ── 1. Python version ──────────────────────────────────────────────────────
v = sys.version_info
label = f"Python {v.major}.{v.minor}.{v.micro}"
if v.major == 3 and v.minor in (10, 11, 12):
    ok(label)
else:
    fail(label, f"project requires 3.10-3.12, got {v.major}.{v.minor}.{v.micro}")

# ── 2. Core packages ───────────────────────────────────────────────────────
REQUIRED = [
    ("torch",                 "torch"),
    ("faiss-cpu",             "faiss"),
    ("chromadb",              "chromadb"),
    ("bm25s",                 "bm25s"),
    ("pymupdf",               "fitz"),
    ("sentence-transformers", "sentence_transformers"),
    ("ragas",                 "ragas"),
    ("litellm",               "litellm"),   # version via importlib.metadata
    ("pydantic",              "pydantic"),
    ("python-dotenv",         "dotenv"),
    ("numpy",                 "numpy"),
    ("pillow",                "PIL"),
    ("pytesseract",           "pytesseract"),
]

for pkg_name, import_name in REQUIRED:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            try:
                import importlib.metadata as _meta
                ver = _meta.version(pkg_name)
            except Exception:
                ver = "?"
        ok(f"package  {pkg_name}", ver)
    except ImportError:
        fail(f"package  {pkg_name}", f"not installed – run: uv pip install {pkg_name}")

# ── 3. Tesseract binary ────────────────────────────────────────────────────
try:
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        capture_output=True, text=True, timeout=10
    )
    if "eng" in result.stdout + result.stderr:
        ok("Tesseract binary", "eng language found")
    else:
        fail("Tesseract binary", "installed but 'eng' language pack missing")
except FileNotFoundError:
    fail("Tesseract binary", "not found on PATH – install from https://github.com/UB-Mannheim/tesseract/wiki")
except Exception as e:
    fail("Tesseract binary", str(e))

# ── 4. Poppler binary ─────────────────────────────────────────────────────
_POPPLER_FALLBACKS = [
    r"D:\Software\poppler\poppler-26.02.0\Library\bin",
    r"C:\Program Files\poppler\Library\bin",
    r"C:\poppler\bin",
]

def _find_pdftoppm():
    """Return (exe_path, via_path) or raise FileNotFoundError."""
    import shutil
    exe = shutil.which("pdftoppm")
    if exe:
        return exe, True
    for folder in _POPPLER_FALLBACKS:
        candidate = os.path.join(folder, "pdftoppm.exe")
        if os.path.isfile(candidate):
            return candidate, False
    raise FileNotFoundError("pdftoppm not found on PATH or known fallback locations")

try:
    exe, on_path = _find_pdftoppm()
    result = subprocess.run(
        [exe, "-v"],
        capture_output=True, text=True, timeout=10
    )
    # pdftoppm prints version to stderr
    output = result.stderr + result.stdout
    version_line = next(
        (line for line in output.splitlines() if "version" in line.lower()), output.strip()
    )
    ver_str = version_line.strip()
    detail = ver_str + ("" if on_path else f"  [not on PATH – add {os.path.dirname(exe)} to PATH]")
    if on_path:
        ok("Poppler binary (pdftoppm)", ver_str)
    else:
        # Found but not on PATH – still usable, but warn
        print(f"ATTENTION  Poppler binary (pdftoppm)  ->  {detail}")
        issues.append("Poppler binary not on PATH")
except FileNotFoundError:
    fail(
        "Poppler binary (pdftoppm)",
        "not found – install Poppler and add its bin/ folder to PATH  "
        "(https://github.com/oschwartz10612/poppler-windows/releases)"
    )
except Exception as e:
    fail("Poppler binary (pdftoppm)", str(e))

# ── 5. API Keys ────────────────────────────────────────────────────────────
for key_name in ["GROQ_API_KEY", "GEMINI_API_KEY"]:
    val = os.getenv(key_name)
    if val and len(val) > 8:
        ok(f"env      {key_name}", val[:8] + "…")
    else:
        fail(f"env      {key_name}", "missing or too short – add to .env")

# ── 6. FAISS + Torch coexistence (OpenMP crash test) ──────────────────────
try:
    # pyrefly: ignore [missing-import]
    import numpy as np, faiss, torch
    idx = faiss.IndexFlatIP(64)
    idx.add(np.random.rand(10, 64).astype("float32"))
    _ = torch.zeros(3)
    ok("faiss + torch coexistence", f"{idx.ntotal} vectors, no OpenMP crash")
except Exception as e:
    fail("faiss + torch coexistence", str(e))

# ── 7. Sentence-Transformers model download (fast, tiny model) ────────────
try:
    # pyrefly: ignore [missing-import]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")   # ~80 MB, cached after first run
    emb = model.encode(["hello world"])
    ok("SentenceTransformer model", f"all-MiniLM-L6-v2, dim={emb.shape[1]}")
except Exception as e:
    fail("SentenceTransformer model", str(e))

# ── 8. ChromaDB sanity ─────────────────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import chromadb
    client = chromadb.Client()
    col = client.get_or_create_collection("doctor_test")
    col.add(documents=["test"], ids=["1"])
    ok("ChromaDB in-memory", f"v{chromadb.__version__}")
    client.delete_collection("doctor_test")
except Exception as e:
    fail("ChromaDB in-memory", str(e))

# ── 9. LiteLLM import (no API call) ───────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import litellm
    ok("litellm import", f"v{litellm.__version__}")
except Exception as e:
    fail("litellm import", str(e))

# ── 10. Torch device ──────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        dev = f"cuda ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    ok("torch device", dev)
except Exception as e:
    fail("torch device", str(e))

# ── Summary ────────────────────────────────────────────────────────────────
print()
if not issues:
    print("=" * 60)
    print("  ALL CHECKS PASSED  –  environment is ready!")
    print("=" * 60)
else:
    print("=" * 60)
    print(f"  {len(issues)} issue(s) need fixing:")
    for i in issues:
        print(f"    •  {i}")
    print("=" * 60)
    sys.exit(1)
