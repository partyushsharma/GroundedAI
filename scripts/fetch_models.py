"""Download every model the project needs into the local Hugging Face cache.

Task 0.5. Idempotent: re-running against a warm cache is a no-op.

Why explicit allow/ignore patterns instead of a plain snapshot_download?
Several of these repos ship the same weights three or four times over --
safetensors plus a legacy pytorch_model.bin plus ONNX plus OpenVINO exports.
intfloat/e5-large-v2 alone carries 1.34 GB of duplicate .bin and another
1.67 GB of OpenVINO. We only ever load these through sentence-transformers,
so we take the safetensors copy and the tokeniser files and nothing else.
That is the difference between a ~4.6 GB download and a ~12 GB one.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from huggingface_hub import snapshot_download

# Files sentence-transformers actually reads. Everything else is an export
# format we never touch.
ST_FILES = [
    "*.json",           # config, tokenizer_config, special_tokens_map, modules
    "1_Pooling/*",      # pooling layer config
    "*.txt",            # vocab.txt (BERT-family tokenisers)
    "*.model",          # sentencepiece.bpe.model (XLM-R family)
    "*.safetensors",
]

# Anything matching these is dropped even if it matched above -- fnmatch
# treats "*" as crossing directory separators, so "onnx/model.safetensors"
# would otherwise sneak through on "*.safetensors".
EXPORT_DIRS = ["onnx/*", "openvino/*", "coreml/*", "assets/*", "*.onnx", "*.onnx_data"]

MODELS = [
    {
        "repo": "BAAI/bge-base-en-v1.5",
        "role": "Default embedder, 768-dim",
        "allow": ST_FILES,
        "expected_gb": 0.44,
    },
    {
        "repo": "intfloat/e5-large-v2",
        "role": "Larger-model embedding ablation, 1024-dim",
        "allow": ST_FILES,
        "expected_gb": 1.34,
    },
    {
        "repo": "nomic-ai/nomic-embed-text-v1.5",
        "role": "Matryoshka truncation ablation (768/256/128)",
        "allow": ST_FILES,
        "expected_gb": 0.55,
    },
    {
        # nomic-embed's config.json has an auto_map pointing at THIS repo for
        # its modelling code, so trust_remote_code=True needs it cached too.
        # Code only -- this repo's own weights are not used.
        "repo": "nomic-ai/nomic-bert-2048",
        "role": "Remote modelling code for nomic-embed (no weights)",
        "allow": ["*.py", "config.json"],
        "expected_gb": 0.00,
    },
    {
        "repo": "BAAI/bge-reranker-v2-m3",
        "role": "Cross-encoder reranker",
        "allow": ST_FILES,
        "expected_gb": 2.27,
    },
]


def dir_size_gb(path: str | Path) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / 1e9


def main() -> int:
    print(f"Fetching {len(MODELS)} model repos into the Hugging Face cache.\n")
    rows, failures = [], []

    for m in MODELS:
        repo = m["repo"]
        print(f"--- {repo}")
        print(f"    {m['role']}")
        started = time.monotonic()
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=m["allow"],
                ignore_patterns=EXPORT_DIRS,
            )
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"    FAILED: {type(exc).__name__}: {exc}\n")
            failures.append(repo)
            continue
        elapsed = time.monotonic() - started
        size = dir_size_gb(path)
        rows.append((repo, size, elapsed))
        print(f"    ok  {size:.2f} GB in {elapsed:.0f}s\n")

    print("=" * 72)
    print(f"{'repo':42s} {'size':>9s} {'time':>7s}")
    print("-" * 72)
    for repo, size, elapsed in rows:
        print(f"{repo:42s} {size:8.2f} GB {elapsed:6.0f}s")
    print("-" * 72)
    print(f"{'TOTAL':42s} {sum(r[1] for r in rows):8.2f} GB")

    if failures:
        print(f"\n{len(failures)} repo(s) failed: {', '.join(failures)}")
        return 1

    print("\nAll models cached. Next: python doctor.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
