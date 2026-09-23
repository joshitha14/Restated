"""Run the eval question set against each configuration and score it.

This is the deliverable. The pipeline exists to produce this table.

Scored per query category, never only in aggregate -- aggregate scores hide
where systems break, which is the one thing the report needs to show.

Metrics, and why each is here:

    numeric_accuracy   the expected figure appears in a retrieved chunk. The
                       sharpest discriminator: a figure is either there or it
                       is not, with no judgement call.
    recall@k           any legitimate source chunk was retrieved. Looser than
                       numeric accuracy; catches "found the right table but the
                       figure was in a column we did not check".
    year_precision     fraction of retrieved chunks belonging to a fiscal year
                       the question actually asked about. Directly measures
                       period confusion -- the thing this corpus was chosen for.
    both_sides         restatement/split only: did retrieval surface BOTH
                       conflicting figures? A system that returns one of them
                       looks correct while hiding that the other exists.
    p50/p95 latency    reranking is a tradeoff, not a free win, so quality is
                       always reported next to what it cost.

Refusal accuracy and faithfulness need generation and are scored in the
generation stage; this module measures retrieval alone, which is what runs A-D
are about.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieve import build, years_in_query  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
REPORTS_DIR = ROOT / "reports"

TOP_K = 5


# --------------------------------------------------------------------------
# expected-figure extraction
# --------------------------------------------------------------------------

# Figures embedded in a prose `expected:` string, e.g.
#   "193737 - 115186 = 78551 (68.2% growth)"
#   "Decelerated: FY2024 47,525 -> FY2025 115,186 (+142%)"
# Percentages and small integers are excluded: they are commentary, not the
# figure being retrieved, and matching "68" against a filing would hit anything.
PROSE_FIGURE_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d{4,})\b")


def expected_figures(q: dict) -> list[str]:
    """Every figure that would count as correct, formatted as it appears.

    A question may have one expected value, several when filings disagree
    (`per_fy2026_filing: 77482` / `per_fy2025_filing: 61257`), or a prose
    explanation with figures embedded in it.

    The prose case was originally missed entirely, which left three questions
    (q_temporal_01, q_trend_01, q_trend_03) permanently unscoreable -- they could
    never register a hit regardless of what retrieval returned. A metric that
    cannot be satisfied is worse than no metric: it silently depresses every
    run's score by the same amount, which looks like a valid comparison.
    """
    exp = q.get("expected")
    out: list[str] = []

    def add_number(v) -> None:
        out.append(f"{v:,}")
        out.append(str(v))

    def add(v) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            add_number(v)
        elif isinstance(v, str):
            for raw in PROSE_FIGURE_RE.findall(v):
                bare = raw.replace(",", "")
                out.append(raw)
                out.append(bare)
                if bare.isdigit():
                    out.append(f"{int(bare):,}")

    if isinstance(exp, dict):
        for v in exp.values():
            add(v)
    else:
        add(exp)
    return [s for s in dict.fromkeys(out) if s]


def expected_by_source(q: dict) -> dict[str, str]:
    """For restatement/split questions: {source_label: figure_as_printed}."""
    exp = q.get("expected")
    if not isinstance(exp, dict):
        return {}
    out = {}
    for key, val in exp.items():
        if isinstance(val, (int, float)):
            out[key] = f"{val:,}"
    return out


def asked_years(q: dict) -> list[int]:
    """Fiscal years the question is about, for year_precision."""
    years = years_in_query(q["question"])
    if years:
        return years
    # Fall back to the declared sources, e.g. ["FY2026", "FY2025"].
    return sorted(
        {int(s[2:]) for s in q.get("source", []) if str(s).startswith("FY")},
        reverse=True,
    )


# --------------------------------------------------------------------------
# scoring one question
# --------------------------------------------------------------------------

@dataclass
class QResult:
    qid: str
    category: str
    question: str
    retriever: str
    strategy: str
    latency_ms: float
    n_hits: int
    numeric_hit: bool = False
    numeric_rank: int | None = None  # 1-based rank of the first chunk with the figure
    recall_hit: bool = False
    year_precision: float = 0.0
    both_sides: bool | None = None
    found_figures: list[str] = field(default_factory=list)
    top_citation: str = ""
    citations: list[str] = field(default_factory=list)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first chunk containing the answer, 0 if absent.

        Binary top-k accuracy cannot separate a configuration that puts the
        answer first from one that buries it at rank 5, even though only the
        first is usable by a generator with a limited context budget.
        """
        return 1.0 / self.numeric_rank if self.numeric_rank else 0.0


def score_question(q: dict, hits: list, retriever: str, strategy: str, ms: float) -> QResult:
    texts = [h.text for h in hits]
    blob = "\n".join(texts)

    res = QResult(
        qid=q["id"],
        category=q["category"],
        question=q["question"],
        retriever=retriever,
        strategy=strategy,
        latency_ms=ms,
        n_hits=len(hits),
        top_citation=hits[0].citation if hits else "",
        citations=[h.citation for h in hits],
    )

    if q["category"] == "unanswerable":
        # Retrieval cannot "fail" an unanswerable question -- the refusal happens
        # at generation. Recorded so the category still appears in the table.
        res.recall_hit = True
        return res

    figures = expected_figures(q)
    found = [f for f in figures if f in blob]
    res.found_figures = found
    res.numeric_hit = bool(found)

    # Rank of the first chunk carrying any expected figure.
    for i, text in enumerate(texts, start=1):
        if any(f in text for f in figures):
            res.numeric_rank = i
            break

    # recall@k: a chunk from a filing the question says can answer it.
    want = asked_years(q)
    if want:
        res.recall_hit = any(
            any(y in h.years_present for y in want) or h.fiscal_year in want
            for h in hits
        )
        covered = sum(
            1 for h in hits
            if any(y in h.years_present for y in want) or h.fiscal_year in want
        )
        res.year_precision = covered / len(hits) if hits else 0.0
    else:
        res.recall_hit = bool(found)
        res.year_precision = 1.0 if hits else 0.0

    # both_sides: for the traps, did we surface BOTH conflicting figures?
    by_source = expected_by_source(q)
    if len(by_source) >= 2:
        res.both_sides = all(fig in blob for fig in by_source.values())

    return res


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------

def run_config(
    questions: list[dict], retriever_kind: str, strategy: str, k: int, embeddings=None
) -> list[QResult]:
    r = build(retriever_kind, strategy, embeddings)

    # One throwaway query before timing starts. The cross-encoder loads a ~1.1GB
    # model lazily, so without this the first question absorbs several seconds of
    # model-loading and the reported latency measures the wrong thing. Harmless
    # for the other retrievers.
    r.search("warmup", k=1)

    out = []
    for q in questions:
        t0 = time.perf_counter()
        hits = r.search(q["question"], k=k)
        ms = (time.perf_counter() - t0) * 1000
        out.append(score_question(q, hits, retriever_kind, strategy, ms))
    return out


def summarise(results: list[QResult]) -> dict:
    lats = sorted(r.latency_ms for r in results)
    scored = [r for r in results if r.category != "unanswerable"]
    traps = [r for r in results if r.both_sides is not None]

    def pct(rs, attr) -> float:
        return 100.0 * sum(bool(getattr(r, attr)) for r in rs) / len(rs) if rs else 0.0

    return {
        "n": len(results),
        "numeric_accuracy": pct(scored, "numeric_hit"),
        "numeric_mrr": 100.0 * st.mean([r.reciprocal_rank for r in scored]) if scored else 0.0,
        "recall_at_k": pct(scored, "recall_hit"),
        "year_precision": 100.0 * st.mean([r.year_precision for r in scored]) if scored else 0.0,
        "both_sides": pct(traps, "both_sides") if traps else float("nan"),
        "p50_ms": st.median(lats) if lats else 0.0,
        "p95_ms": lats[int(len(lats) * 0.95)] if lats else 0.0,
    }


def by_category(results: list[QResult]) -> dict[str, dict]:
    groups: dict[str, list[QResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    return {cat: summarise(rs) for cat, rs in groups.items()}


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--k",
        default="5,10,20",
        help="comma-separated retrieval depths. Reporting a single k hides whether "
             "a configuration failed or merely ranked the answer below the cutoff.",
    )
    ap.add_argument(
        "--runs",
        default="A,B,B2,C,D,E",
        help="comma-separated. A=fixed/dense B=SemanticChunker/dense "
             "B2=structural/dense C=fixed/hybrid D=fixed/filtered E=fixed/bm25 "
             "F=rerank over hybrid G=rerank over filtered. F and G need "
             "sentence-transformers and are not included by default.",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = spec["questions"]
    depths = [int(x) for x in str(args.k).split(",") if str(x).strip()]
    print(f"{len(questions)} questions, k={depths}\n")

    # One embeddings client shared across runs, so latency reflects retrieval
    # rather than repeated client construction.
    from langchain_openai import OpenAIEmbeddings
    from index import BASE_URL, EMBED_MODEL, api_key

    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL, base_url=BASE_URL, api_key=api_key(),
        check_embedding_ctx_length=False,
    )

    CONFIGS = {
        "A":  ("dense",    "fixed",          "baseline: fixed chunking, dense only"),
        "B":  ("dense",    "semantic_embed", "chunking effect: SemanticChunker"),
        "B2": ("dense",    "semantic",       "chunking effect: structural"),
        "C":  ("hybrid",   "fixed",          "retrieval effect: dense + BM25 fusion"),
        "D":  ("filtered", "fixed",          "metadata effect: years_present filter"),
        "E":  ("bm25",     "fixed",          "lexical only -- explains hybrid's result"),
        # Reranking is scored against C and D, not against the dense baseline.
        # Hybrid already fixed the split trap and metadata filtering already beat
        # both on numeric accuracy, so a cross-encoder earns its place only if it
        # improves on those.
        "F":  ("rerank",          "fixed",  "cross-encoder rerank over hybrid"),
        "G":  ("rerank:filtered", "fixed",  "cross-encoder rerank over filtered"),
    }

    wanted = [r.strip() for r in args.runs.split(",") if r.strip()]

    # {k: {run_id: [QResult]}}
    by_depth: dict[int, dict[str, list[QResult]]] = {}

    for k in depths:
        by_depth[k] = {}
        print(f"--- k={k} ---")
        for run_id in wanted:
            if run_id not in CONFIGS:
                continue
            kind, strategy, desc = CONFIGS[run_id]
            if not (ROOT / "data" / "chunks" / strategy).exists():
                print(f"  run {run_id}: no chunks for '{strategy}', skipping")
                continue
            try:
                results = run_config(questions, kind, strategy, k, embeddings)
            except Exception as exc:
                # Print the traceback, not just the message. A bare message made
                # runs D and G disappear from a combined scorecard with no
                # indication of why, while both worked in isolation.
                import traceback

                print(f"  run {run_id} FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            by_depth[k][run_id] = results
            s = summarise(results)
            print(
                f"  {run_id:3s} numeric={s['numeric_accuracy']:5.1f}% "
                f"mrr={s['numeric_mrr']:5.1f} recall={s['recall_at_k']:5.1f}% "
                f"yr={s['year_precision']:5.1f}% both={s['both_sides']:5.1f}% "
                f"p50={s['p50_ms']:6.1f}ms"
            )
        print()

    if not any(by_depth.values()):
        sys.exit("no runs completed")

    # The deepest k drives the detailed tables; the curve is shown separately.
    all_results = by_depth[depths[-1]]

    # ---- headline table --------------------------------------------------
    print("=" * 96)
    print("RETRIEVAL SCORECARD")
    print("=" * 96)
    hdr = (f"{'run':4s} {'chunking':15s} {'retrieval':9s} {'numeric':>8s} {'mrr':>6s} "
           f"{'recall':>7s} {'yr_prec':>8s} {'both':>6s} {'p50':>8s} {'p95':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for run_id, results in all_results.items():
        kind, strategy, _ = CONFIGS[run_id]
        s = summarise(results)
        both = "  n/a" if s["both_sides"] != s["both_sides"] else f"{s['both_sides']:5.0f}%"
        print(
            f"{run_id:4s} {strategy:15s} {kind:9s} {s['numeric_accuracy']:7.1f}% "
            f"{s['numeric_mrr']:6.1f} {s['recall_at_k']:6.1f}% {s['year_precision']:7.1f}% "
            f"{both:>6s} {s['p50_ms']:7.1f}ms {s['p95_ms']:8.1f}ms"
        )

    # ---- per category ----------------------------------------------------
    print("\n" + "=" * 96)
    print("NUMERIC ACCURACY BY CATEGORY  (aggregate scores hide where systems break)")
    print("=" * 96)
    cats = ["numeric", "temporal", "trend", "restatement", "split", "unanswerable"]
    print(f"{'run':4s} " + " ".join(f"{c[:11]:>12s}" for c in cats))
    print("-" * (5 + 13 * len(cats)))
    for run_id, results in all_results.items():
        cells = []
        per = by_category(results)
        for c in cats:
            cells.append(f"{per[c]['numeric_accuracy']:11.0f}%" if c in per else f"{'-':>12s}")
        print(f"{run_id:4s} " + " ".join(cells))

    # ---- numeric accuracy vs retrieval depth ------------------------------
    print("\n" + "=" * 96)
    print("NUMERIC ACCURACY vs RETRIEVAL DEPTH")
    print("=" * 96)
    print("A single k conflates 'the answer is not indexed' with 'it ranked below the cutoff'.")
    print("Those need opposite fixes, so the curve is reported rather than one number.\n")
    print(f"{'run':4s} {'chunking':15s} {'retrieval':9s} " + " ".join(f"{'k=' + str(k):>8s}" for k in depths))
    print("-" * (32 + 9 * len(depths)))
    for run_id in all_results:
        kind, strategy, _ = CONFIGS[run_id]
        cells = []
        for k in depths:
            rs = by_depth[k].get(run_id)
            cells.append(f"{summarise(rs)['numeric_accuracy']:7.1f}%" if rs else f"{'-':>8s}")
        print(f"{run_id:4s} {strategy:15s} {kind:9s} " + " ".join(cells))

    # ---- persist ---------------------------------------------------------
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = {
        "depths": depths,
        "k": depths[-1],
        "curve": {
            str(k): {
                run_id: summarise(rs) for run_id, rs in by_depth[k].items()
            }
            for k in depths
        },
        "runs": {
            run_id: {
                "chunking": CONFIGS[run_id][1],
                "retrieval": CONFIGS[run_id][0],
                "summary": summarise(rs),
                "by_category": by_category(rs),
                "questions": [asdict(r) for r in rs],
            }
            for run_id, rs in all_results.items()
        },
    }
    dest = REPORTS_DIR / "retrieval_scorecard.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
