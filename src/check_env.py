"""Check the environment is ready to index: packages, corpus, chunks, API key.

Run before src/index.py. Indexing spends Nebius credits, so every precondition
that can be checked for free is checked first.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PACKAGES = [
    ("langchain", True),
    ("langchain_openai", True),
    ("langchain_community", True),
    ("langchain_text_splitters", True),
    ("chromadb", True),
    ("rank_bm25", True),
    ("yaml", True),
    ("pandas", True),
    ("dotenv", False),
    ("sentence_transformers", False),  # reranker; only needed for run C
    ("langsmith", False),
]


def main() -> int:
    problems: list[str] = []

    # Load .env if present, matching what src/index.py does.
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    print("=== packages ===")
    for name, required in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            print(f"  ok    {name:26s} {version}")
        except Exception as exc:
            tag = "FAIL " if required else "opt  "
            print(f"  {tag} {name:26s} {type(exc).__name__}")
            if required:
                problems.append(f"missing package: {name}")

    print("\n=== corpus ===")
    raw = sorted(list((ROOT / "data" / "raw").glob("nvda-*.htm"))
                 + list((ROOT / "data" / "raw").glob("nvda-*.html")))
    print(f"  raw filings: {len(raw)}")
    for p in raw:
        print(f"    {p.name}  {p.stat().st_size:,} bytes")
    if len(raw) < 3:
        problems.append("fewer than 3 filings in data/raw -- run src/fetch.py")

    print("\n=== chunks ===")
    total = 0
    for strategy in ("fixed", "semantic"):
        n = 0
        for fy in (2024, 2025, 2026):
            path = ROOT / "data" / "chunks" / strategy / f"FY{fy}.jsonl"
            if path.exists():
                n += sum(1 for _ in path.open(encoding="utf-8"))
        total += n
        print(f"  {strategy:9s}: {n:,} chunks")
        if not n:
            problems.append(f"no {strategy} chunks -- run src/chunk.py")

    print("\n=== credentials ===")
    key = os.environ.get("NEBIUS_API_KEY", "")
    if key:
        print(f"  ok    NEBIUS_API_KEY set ({len(key)} chars, ends ...{key[-4:]})")
    else:
        print("  FAIL  NEBIUS_API_KEY not set")
        problems.append(
            "NEBIUS_API_KEY not set. Get a key from "
            "https://tokenfactory.nebius.com/project/api-keys then:\n"
            '      $env:NEBIUS_API_KEY = "your-key"'
        )

    ua = os.environ.get("SEC_USER_AGENT", "")
    print(f"  {'ok   ' if ua else 'note '} SEC_USER_AGENT {'set' if ua else 'not set (only needed for src/fetch.py)'}")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"ready to index: {total:,} chunks across two strategies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
