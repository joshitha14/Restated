"""Embed chunks into two Chroma collections, one per chunking strategy.

Two collections rather than one, so both strategies are queryable without
re-indexing (handoff §6 step 3). Re-running is cheap: chunk_ids are stable
hashes, so an existing collection with a matching count is skipped unless
--force is passed. Embedding spends Nebius credits; re-embedding by accident
should not be possible.

Nebius Token Factory is OpenAI-compatible, so langchain_openai talks to it
directly with a different base_url.

    python src/index.py --list-models      # what this account can use
    python src/index.py                    # embed both strategies
    python src/index.py --strategy fixed   # one only
    python src/index.py --limit 20         # smoke test, ~20 chunks per strategy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "chunks"
STORE_DIR = ROOT / "data" / "chroma"

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"

FYS = (2024, 2025, 2026)
# "semantic" is the structural chunker; "semantic_embed" is LangChain's
# SemanticChunker. Both are indexed so all three arms are queryable.
STRATEGIES = ("fixed", "semantic", "semantic_embed")

# Nebius rejects very large batches; 64 is comfortably under the limit and
# keeps a failed batch cheap to retry.
BATCH = 64

# Because check_embedding_ctx_length=False disables LangChain's own length
# guard, over-long input would be silently truncated server-side rather than
# erroring -- a chunk would embed as a prefix of itself, with no warning.
# Qwen3-Embedding-8B has a 32k context; the largest chunk in this corpus is
# ~3,600 chars (~900 tokens), so this ceiling is far above anything real and
# exists to catch a future corpus or chunker change.
MAX_CHARS = 24_000


def load_env() -> None:
    """Load .env if present, so the key survives between shell sessions."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def api_key() -> str:
    load_env()
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not key:
        sys.exit(
            "NEBIUS_API_KEY not set.\n"
            "  1. log in at https://tokenfactory.nebius.com (Token Factory, not AI Cloud)\n"
            "  2. API keys -> create key\n"
            '  3. $env:NEBIUS_API_KEY = "your-key"'
        )
    return key


def list_models() -> None:
    """Print the account's available models, split into chat and embedding."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key(), base_url=BASE_URL)
    models = sorted(m.id for m in client.models.list().data)

    def is_embed(name: str) -> bool:
        return any(k in name.lower() for k in ("embed", "bge", "e5", "gte"))

    print("EMBEDDING models:")
    for m in models:
        if is_embed(m):
            mark = "  <- default" if m == EMBED_MODEL else ""
            print(f"  {m}{mark}")

    print("\nCHAT models:")
    for m in models:
        if not is_embed(m):
            print(f"  {m}")


def load_chunks(strategy: str, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    for fy in FYS:
        path = CHUNK_DIR / strategy / f"FY{fy}.jsonl"
        if not path.exists():
            sys.exit(f"missing {path} -- run src/chunk.py first")
        out.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return out[:limit] if limit else out


def flatten_metadata(chunk: dict) -> dict:
    """Chroma accepts only str/int/float/bool values, so lists are serialised.

    `years_present` is a list (DECISIONS.md D7) and is the field metadata-filtered
    retrieval needs. It is stored two ways: a readable CSV string for citations,
    and one boolean per year so Chroma's `where` clause can filter on it.
    """
    meta = {
        k: v
        for k, v in chunk.items()
        if k != "text" and isinstance(v, (str, int, float, bool)) and v != ""
    }
    years = chunk.get("years_present") or []
    meta["years_present"] = ",".join(str(y) for y in years)
    for fy in FYS:
        meta[f"has_fy{fy}"] = fy in years
    return meta


def embed_strategy(
    strategy: str, model: str, limit: int | None, force: bool
) -> tuple[int, float]:
    import chromadb
    from langchain_openai import OpenAIEmbeddings

    chunks = load_chunks(strategy, limit)

    oversized = [c for c in chunks if len(c["text"]) > MAX_CHARS]
    if oversized:
        sys.exit(
            f"{len(oversized)} chunk(s) exceed {MAX_CHARS:,} chars and would be "
            f"silently truncated server-side; largest is "
            f"{max(len(c['text']) for c in oversized):,}. Fix src/chunk.py first."
        )

    client = chromadb.PersistentClient(path=str(STORE_DIR))
    name = f"nvda_{strategy}"

    existing = {c.name for c in client.list_collections()}
    if name in existing:
        coll = client.get_collection(name)
        if coll.count() == len(chunks) and not force:
            print(f"  {strategy}: {coll.count():,} vectors already indexed, skipping (--force to redo)")
            return 0, 0.0
        client.delete_collection(name)

    coll = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    # check_embedding_ctx_length=False is REQUIRED for Nebius.
    #
    # By default langchain_openai tokenizes text locally with tiktoken and sends
    # integer arrays, which OpenAI accepts but Nebius rejects:
    #     400 - {'detail': 'Tokenized input is not supported'}
    # Disabling it sends raw strings instead. The trade-off is that LangChain no
    # longer splits over-long inputs automatically -- see MAX_CHARS below.
    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=BASE_URL,
        api_key=api_key(),
        check_embedding_ctx_length=False,
    )

    start = time.perf_counter()
    done = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        vectors = embeddings.embed_documents([c["text"] for c in batch])
        coll.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[flatten_metadata(c) for c in batch],
        )
        done += len(batch)
        print(f"  {strategy}: {done:,}/{len(chunks):,} embedded", end="\r", flush=True)

    elapsed = time.perf_counter() - start
    print(f"  {strategy}: {done:,} vectors in {elapsed:,.1f}s ({done / elapsed:.1f}/s)      ")
    return done, elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-models", action="store_true", help="show available models and exit")
    ap.add_argument(
        "--strategy",
        choices=[*STRATEGIES, "both", "all"],
        default="both",
        help="'both' = fixed + semantic; 'all' adds semantic_embed",
    )
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--limit", type=int, help="embed only the first N chunks (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-embed even if already indexed")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return

    api_key()
    if args.strategy == "both":
        names = ["fixed", "semantic"]
    elif args.strategy == "all":
        names = list(STRATEGIES)
    else:
        names = [args.strategy]

    print(f"model: {args.model}")
    print(f"store: {STORE_DIR.relative_to(ROOT)}")
    if args.limit:
        print(f"LIMIT: {args.limit} chunks per strategy (smoke test)")
    print()

    total, secs = 0, 0.0
    for strategy in names:
        n, t = embed_strategy(strategy, args.model, args.limit, args.force)
        total += n
        secs += t

    if total:
        print(f"\n{total:,} vectors embedded in {secs:,.1f}s")
    print(f"-> {STORE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
