"""Retrieval strategies over the indexed collections.

Four retrievers, one interface, so the measurement harness can swap them without
knowing how each works:

    dense      embeddings only -- the baseline
    bm25       lexical only -- included because it is the other half of hybrid,
               and because knowing its solo score explains hybrid's result
    hybrid     dense + BM25 fused with reciprocal rank fusion
    filtered   dense, with `years_present` as a hard pre-filter

Dense finds passages that *mean* the same thing; BM25 finds passages that use the
same *words*. Financial queries need both -- "Data Center revenue" is a phrase to
match literally, while "how fast is the business growing" is not.

Reciprocal rank fusion is used rather than score averaging because dense
distances and BM25 scores are on incomparable scales. RRF only uses RANK, so no
normalisation is needed and no tuning constant has to be justified:

    score(d) = sum over retrievers of 1 / (k + rank(d))

k=60 is the value from the original RRF paper (Cormack et al. 2009) and is used
unchanged -- tuning it on this corpus would be fitting the retriever to the eval
set, which is what the eval-before-retrieval rule exists to prevent.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from index import BASE_URL, EMBED_MODEL, STORE_DIR, api_key  # noqa: E402

CHUNK_DIR = ROOT / "data" / "chunks"
FYS = (2024, 2025, 2026)

# Cormack et al. 2009. Left at the paper's value deliberately; see module docstring.
RRF_K = 60


# One cached client. Per-collection clients were tried as a workaround for the
# filtered-query bug described in FilteredRetriever.search and made no
# difference, so the simpler form stands.
_CLIENTS: dict[str, object] = {}


def open_collection(name: str):
    import chromadb

    if "client" not in _CLIENTS:
        _CLIENTS["client"] = chromadb.PersistentClient(path=str(STORE_DIR))
    return _CLIENTS["client"].get_collection(name)


@dataclass
class Hit:
    """One retrieved chunk, with everything scoring needs."""

    chunk_id: str
    text: str
    score: float
    rank: int
    citation: str
    fiscal_year: int
    item_section: str
    block_type: str
    years_present: list[int] = field(default_factory=list)
    table_caption: str = ""

    @classmethod
    def from_chroma(cls, cid: str, doc: str, meta: dict, score: float, rank: int) -> "Hit":
        years = [int(y) for y in str(meta.get("years_present", "")).split(",") if y.strip()]
        return cls(
            chunk_id=cid,
            text=doc,
            score=score,
            rank=rank,
            citation=meta.get("citation", ""),
            fiscal_year=int(meta.get("fiscal_year", 0)),
            item_section=meta.get("item_section", ""),
            block_type=meta.get("block_type", ""),
            years_present=years,
            table_caption=meta.get("table_caption", ""),
        )


# --------------------------------------------------------------------------
# query-time fiscal year extraction (for the metadata-filtered arm)
# --------------------------------------------------------------------------

FY_QUERY_RE = re.compile(r"fiscal(?:\s+year)?s?\s+((?:20\d{2}[\s,and]*)+)", re.I)
BARE_YEAR_RE = re.compile(r"\b(20[12]\d)\b")

# NVIDIA's fiscal year ends in late January, so FY2026 covers ~calendar 2025.
# A user asking about "2025" most likely means FY2026. This mapping is applied
# ONLY when the query says "calendar", because guessing silently would make the
# disambiguation question (q_temporal_03) untestable.
CALENDAR_OFFSET = 1


def years_in_query(query: str) -> list[int]:
    """Fiscal years named in a query, for use as a hard filter.

    Returns [] when no year is named, which callers treat as "no filter" rather
    than "match nothing" -- a query with no year should search everything.
    """
    years: set[int] = set()
    for m in FY_QUERY_RE.finditer(query):
        years.update(int(y) for y in BARE_YEAR_RE.findall(m.group(1)))
    if not years:
        # A bare year with no "fiscal" qualifier, e.g. "revenue in 2025".
        years.update(int(y) for y in BARE_YEAR_RE.findall(query))
        if "calendar" in query.lower():
            years = {y + CALENDAR_OFFSET for y in years}
    return sorted(y for y in years if 2020 <= y <= 2030)


# --------------------------------------------------------------------------
# retrievers
# --------------------------------------------------------------------------

class Retriever:
    """Base interface: .search(query, k) -> list[Hit]."""

    name = "base"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, strategy: str, embeddings=None):
        from langchain_openai import OpenAIEmbeddings

        self.strategy = strategy
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=BASE_URL,
            api_key=api_key(),
            check_embedding_ctx_length=False,
        )
        self.coll = open_collection(f"nvda_{strategy}")

    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        vec = self.embeddings.embed_query(query)
        res = self.coll.query(
            query_embeddings=[vec],
            n_results=k,
            **({"where": where} if where else {}),
        )
        if not res["ids"][0]:
            return []
        return [
            # Chroma returns cosine DISTANCE; convert so higher = better.
            Hit.from_chroma(cid, doc, meta, 1.0 - dist, rank)
            for rank, (cid, doc, meta, dist) in enumerate(
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
            )
        ]


class BM25Retriever(Retriever):
    """Lexical retrieval over the same chunks, built in memory from the JSONL.

    Reads from data/chunks/ rather than Chroma so the tokenised corpus matches
    the indexed corpus exactly -- if the two ever diverge, hybrid would fuse
    rankings over different document sets.
    """

    name = "bm25"

    TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.,%-]*")

    def __init__(self, strategy: str):
        from rank_bm25 import BM25Okapi

        self.strategy = strategy
        self.chunks: list[dict] = []
        for fy in FYS:
            path = CHUNK_DIR / strategy / f"FY{fy}.jsonl"
            if not path.exists():
                raise SystemExit(f"missing {path} -- run src/chunk.py first")
            self.chunks.extend(
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            )
        self.bm25 = BM25Okapi([self._tokenise(c["text"]) for c in self.chunks])

    @classmethod
    def _tokenise(cls, text: str) -> list[str]:
        """Keep digits, commas and decimals together so '193,737' stays one token.

        A default word tokeniser splits that into ['193', '737'], which destroys
        exactly the signal BM25 is here to provide on numeric queries.
        """
        return cls.TOKEN_RE.findall(text.lower())

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.bm25.get_scores(self._tokenise(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for rank, i in enumerate(order):
            c = self.chunks[i]
            out.append(
                Hit(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=float(scores[i]),
                    rank=rank,
                    citation=c["citation"],
                    fiscal_year=c["fiscal_year"],
                    item_section=c["item_section"],
                    block_type=c["block_type"],
                    years_present=c["years_present"],
                    table_caption=c.get("table_caption", ""),
                )
            )
        return out


class HybridRetriever(Retriever):
    """Dense + BM25 fused by reciprocal rank fusion."""

    name = "hybrid"

    def __init__(self, strategy: str, embeddings=None, pool: int = 20):
        self.strategy = strategy
        self.dense = DenseRetriever(strategy, embeddings)
        self.lexical = BM25Retriever(strategy)
        # Fuse over a deeper pool than we return: a chunk ranked 15th by one
        # retriever and 3rd by the other should be able to surface.
        self.pool = pool

    def search(self, query: str, k: int = 5) -> list[Hit]:
        runs = [self.dense.search(query, self.pool), self.lexical.search(query, self.pool)]

        fused: dict[str, float] = {}
        best: dict[str, Hit] = {}
        for run in runs:
            for hit in run:
                fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + hit.rank + 1)
                # Keep whichever copy ranked higher, for its text/metadata.
                if hit.chunk_id not in best or hit.rank < best[hit.chunk_id].rank:
                    best[hit.chunk_id] = hit

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out = []
        for rank, (cid, score) in enumerate(ranked):
            hit = best[cid]
            out.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    score=score,
                    rank=rank,
                    citation=hit.citation,
                    fiscal_year=hit.fiscal_year,
                    item_section=hit.item_section,
                    block_type=hit.block_type,
                    years_present=hit.years_present,
                    table_caption=hit.table_caption,
                )
            )
        return out


class FilteredRetriever(Retriever):
    """Dense, with fiscal years from the query applied as a hard pre-filter.

    Filters on `years_present`, not `fiscal_year` (DECISIONS.md D7): a held-whole
    table spans three fiscal years, so filtering on the filing's own year would
    wrongly exclude every multi-year table -- including the one holding the
    answer to most numeric questions.

    Falls back to unfiltered search when the query names no year, and when the
    filter returns nothing (better a wrong-year answer than no answer at all,
    and the scorer records which happened).
    """

    name = "filtered"

    def __init__(self, strategy: str, embeddings=None):
        self.strategy = strategy
        self.dense = DenseRetriever(strategy, embeddings)

    # Over-fetch factor for the post-filter. The filter keeps roughly a third of
    # the corpus, so 4x reliably yields k survivors without a second round trip.
    OVERFETCH = 4

    def search(self, query: str, k: int = 5) -> list[Hit]:
        years = years_in_query(query)
        wanted = [y for y in years if y in FYS]
        if not wanted:
            return self.dense.search(query, k)

        # Filtering happens in Python, not in Chroma's `where` clause.
        #
        # Chroma 1.5.9 raises `InternalError: Error executing plan: Internal
        # error: Error finding id` on metadata-filtered queries once several
        # collections have been opened in the same process. It is selective and
        # silent: in a combined scorecard it removed exactly the two
        # metadata-filtered runs while the others completed, so the result looked
        # like a valid comparison with two arms missing.
        #
        # Reproduced in src/debug_chroma*.py. Ruled out: interleaved access,
        # collection identity, collection size, .get(where=...), and client
        # count. The trigger is in Chroma's Rust query layer, not in this code.
        #
        # Post-filtering is equivalent for this corpus -- same predicate, same
        # ranking, one extra round of scoring -- and does not depend on a
        # library path that fails unpredictably.
        pool = self.dense.search(query, k * self.OVERFETCH)
        kept = [
            h for h in pool
            if any(y in h.years_present for y in wanted) or h.fiscal_year in wanted
        ]
        hits = kept[:k]
        # Re-rank positions after filtering so `rank` stays contiguous.
        for i, h in enumerate(hits):
            h.rank = i
        return hits or self.dense.search(query, k)


# --------------------------------------------------------------------------

RETRIEVERS = {
    "dense": DenseRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "filtered": FilteredRetriever,
}


def build(kind: str, strategy: str, embeddings=None) -> Retriever:
    if kind == "bm25":
        return BM25Retriever(strategy)
    # Reranking lives in its own module: it wraps a base retriever rather than
    # being one, and it pulls in sentence-transformers, which the other paths
    # should not require.
    if kind.startswith("rerank"):
        from rerank import RerankRetriever

        # "rerank" wraps hybrid; "rerank:filtered" wraps the filtered retriever.
        base = kind.split(":", 1)[1] if ":" in kind else "hybrid"
        return RerankRetriever(strategy, embeddings, base=base)
    return RETRIEVERS[kind](strategy, embeddings)


def _demo() -> None:
    queries = [
        "What was NVIDIA's Data Center revenue in fiscal 2026?",
        "What were NVIDIA's United States revenues in fiscal 2025?",
        "What was NVIDIA's basic earnings per share in fiscal 2024?",
    ]
    for kind in RETRIEVERS:
        print(f"\n{'=' * 72}\n{kind}  (strategy=fixed)\n{'=' * 72}")
        r = build(kind, "fixed")
        for q in queries:
            t0 = time.perf_counter()
            hits = r.search(q, k=3)
            ms = (time.perf_counter() - t0) * 1000
            print(f"\n  Q: {q}   [{ms:.0f}ms]")
            for h in hits:
                print(f"    {h.score:7.4f} {h.block_type:6s} {h.citation[:62]}")


if __name__ == "__main__":
    _demo()
