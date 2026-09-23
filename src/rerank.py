"""Cross-encoder reranking over a retrieved candidate pool.

The last planned component, built after the measurements rather than before,
because the numbers changed what it had to prove.

A bi-encoder (the embedding model) encodes the query and each chunk separately
and compares vectors. A cross-encoder reads the query and chunk TOGETHER and
scores the pair directly. That is far more accurate and far more expensive, so
it is only usable as a second pass over a small candidate pool.

`BAAI/bge-reranker-base` runs locally via sentence-transformers -- no API call,
so the latency it adds is CPU time rather than network time, and it is stable
enough to report. That matters because the brief asks for reranking to be
reported WITH timings.

What this has to prove, given the measurements already in hand:

  - hybrid retrieval already fixed the split trap that dense failed (0% -> 100%)
  - metadata filtering already beat both on numeric accuracy for ~13ms

So a reranker earns its place only if it improves on those, not merely on the
dense baseline. Run F is scored against runs C and D, not against A.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieve import Hit, Retriever, build  # noqa: E402

MODEL = "BAAI/bge-reranker-base"

# Candidates pulled before reranking. The reranker is O(pool), so this is the
# latency knob. 30 is deep enough to contain the answers the depth curve showed
# sitting at ranks 12-35, without paying for a pool of 50.
DEFAULT_POOL = 30

_MODEL_CACHE: dict[str, object] = {}


def load_model(name: str = MODEL):
    """Load once per process; the model is ~1.1GB and loading dominates latency."""
    if name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[name] = CrossEncoder(name, max_length=512)
    return _MODEL_CACHE[name]


class RerankRetriever(Retriever):
    """Retrieve a pool with a base retriever, then re-score it with a cross-encoder.

    The base retriever is configurable because the interesting comparison is not
    "reranking vs nothing" -- it is whether reranking adds anything ON TOP OF the
    hybrid and metadata-filtered retrievers that already scored well.
    """

    name = "rerank"

    def __init__(
        self,
        strategy: str = "fixed",
        embeddings=None,
        base: str = "hybrid",
        pool: int = DEFAULT_POOL,
        model: str = MODEL,
    ):
        self.strategy = strategy
        self.base = build(base, strategy, embeddings)
        self.base_kind = base
        self.pool = pool
        self.model_name = model
        self.model = load_model(model)
        # Populated per search, so the evaluator can report the split between
        # retrieval time and reranking time rather than one opaque number.
        self.last_base_ms = 0.0
        self.last_rerank_ms = 0.0

    def search(self, query: str, k: int = 5) -> list[Hit]:
        t0 = time.perf_counter()
        candidates = self.base.search(query, k=self.pool)
        self.last_base_ms = (time.perf_counter() - t0) * 1000

        if not candidates:
            self.last_rerank_ms = 0.0
            return []

        t1 = time.perf_counter()
        scores = self.model.predict([(query, h.text) for h in candidates])
        self.last_rerank_ms = (time.perf_counter() - t1) * 1000

        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))[:k]
        out = []
        for rank, i in enumerate(order):
            h = candidates[i]
            out.append(
                Hit(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    score=float(scores[i]),
                    rank=rank,
                    citation=h.citation,
                    fiscal_year=h.fiscal_year,
                    item_section=h.item_section,
                    block_type=h.block_type,
                    years_present=h.years_present,
                    table_caption=h.table_caption,
                )
            )
        return out


def _demo() -> None:
    queries = [
        "What was NVIDIA's Data Center revenue in fiscal year 2026?",
        "What was NVIDIA's basic earnings per share in fiscal 2024?",
        "What were NVIDIA's United States revenues in fiscal 2025?",
    ]

    print(f"loading {MODEL} ...")
    t0 = time.perf_counter()
    r = RerankRetriever(base="hybrid")
    print(f"loaded in {time.perf_counter() - t0:.1f}s\n")

    plain = build("hybrid", "fixed")

    for q in queries:
        print("=" * 78)
        print(f"Q: {q}\n")

        t0 = time.perf_counter()
        before = plain.search(q, k=3)
        ms_before = (time.perf_counter() - t0) * 1000
        print(f"  hybrid alone [{ms_before:6.0f}ms]")
        for h in before:
            print(f"    {h.score:8.4f} {h.block_type:6s} {h.citation[:58]}")

        after = r.search(q, k=3)
        print(f"\n  + rerank [{r.last_base_ms:.0f}ms retrieve + "
              f"{r.last_rerank_ms:.0f}ms rerank = "
              f"{r.last_base_ms + r.last_rerank_ms:.0f}ms]")
        for h in after:
            print(f"    {h.score:8.4f} {h.block_type:6s} {h.citation[:58]}")

        moved = [h.chunk_id for h in after] != [h.chunk_id for h in before[:len(after)]]
        print(f"\n  top-3 changed: {moved}\n")


if __name__ == "__main__":
    _demo()
