"""Score the generation stage: exit correctness, numeric accuracy, faithfulness.

Retrieval scoring (src/evaluate.py) asks whether the answer was RETRIEVED. This
asks whether it was ANSWERED -- a different question, because a system can
retrieve the right passage and still state the wrong figure, or retrieve nothing
and correctly refuse.

Metrics:

    exit_correct      did it take the right one of ANSWER / CLARIFY / REFUSE?
                      Unanswerable questions must REFUSE; q_temporal_03 must
                      CLARIFY; everything else must ANSWER.
    numeric_correct   the expected figure appears in the answer text, not merely
                      in the passages.
    both_sides        restatement/split only: did the ANSWER state both
                      conflicting figures? Stating one silently is wrong even
                      when the chosen number is correct.
    faithfulness      LLM judge, 0.0-1.0, on whether claims trace to passages.
    hallucination     a figure in the answer that appears in NO passage. The
                      single worst failure: a confident number with no source.
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

from evaluate import expected_by_source, expected_figures  # noqa: E402
from generate import answer, build_llm, judge  # noqa: E402
from retrieve import build  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
REPORTS_DIR = ROOT / "reports"

# Figures stated in an answer, for hallucination checking.
#
# Comma-grouped only. A bare 4-digit run was originally included too, which
# flagged the YEAR "2027" in an answer reasoning about calendar vs fiscal years
# as an unsourced financial figure -- a false positive that reported a 5.3%
# hallucination rate for a correct, fully grounded answer.
#
# Financial figures in these filings are always comma-grouped once they exceed
# 999, and per-share amounts are matched separately below. Years, section
# numbers and percentages are not figures the system claims as answers.
ANSWER_FIGURE_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")

# Per-share amounts: "$12.05", "$1.21". Matched only with a currency marker, so
# percentages and ratios are not mistaken for claimed figures.
PER_SHARE_RE = re.compile(r"\$\s?(\d{1,3}\.\d{2})\b")


def stated_figures(text: str) -> set[str]:
    """Figures an answer claims as financial values."""
    return set(ANSWER_FIGURE_RE.findall(text)) | set(PER_SHARE_RE.findall(text))


def expected_exit(q: dict) -> str:
    if q["category"] == "unanswerable":
        return "REFUSE"
    if q["id"] == "q_temporal_03":
        return "CLARIFY"
    return "ANSWER"


@dataclass
class GResult:
    qid: str
    category: str
    question: str
    exit: str
    expected_exit: str
    exit_correct: bool
    numeric_correct: bool
    both_sides: bool | None
    faithfulness: float
    hallucinated: list[str]
    citations: list[int]
    invalid_citations: list[int]
    answer_text: str
    retrieval_ms: float
    generation_ms: float
    found_figures: list[str] = field(default_factory=list)


def score_one(q: dict, hits, ans, faith: float) -> GResult:
    text = ans.text
    figures = expected_figures(q)
    found = [f for f in figures if f in text]

    # Hallucination: a figure in the answer that no passage contains.
    passage_blob = "\n".join(h.text for h in hits)
    stated = stated_figures(text)
    hallucinated = sorted(
        s for s in stated
        if s not in passage_blob and s.replace(",", "") not in passage_blob.replace(",", "")
    )

    by_source = expected_by_source(q)
    both = all(fig in text for fig in by_source.values()) if len(by_source) >= 2 else None

    want = expected_exit(q)
    return GResult(
        qid=q["id"],
        category=q["category"],
        question=q["question"],
        exit=ans.exit,
        expected_exit=want,
        exit_correct=ans.exit == want,
        numeric_correct=bool(found) if q["category"] != "unanswerable" else ans.exit == "REFUSE",
        both_sides=both,
        faithfulness=faith,
        hallucinated=hallucinated,
        citations=ans.citations,
        invalid_citations=ans.invalid_citations,
        answer_text=text,
        retrieval_ms=0.0,
        generation_ms=ans.latency_ms,
        found_figures=found,
    )


def summarise(rs: list[GResult]) -> dict:
    scored = [r for r in rs if r.category != "unanswerable"]
    traps = [r for r in rs if r.both_sides is not None]
    lat = sorted(r.retrieval_ms + r.generation_ms for r in rs)

    def pct(items, attr) -> float:
        return 100.0 * sum(bool(getattr(i, attr)) for i in items) / len(items) if items else 0.0

    return {
        "n": len(rs),
        "exit_correct": pct(rs, "exit_correct"),
        "numeric_correct": pct(scored, "numeric_correct"),
        "both_sides": pct(traps, "both_sides") if traps else float("nan"),
        "faithfulness": st.mean([r.faithfulness for r in rs]) if rs else 0.0,
        "hallucination_rate": 100.0 * sum(1 for r in rs if r.hallucinated) / len(rs) if rs else 0.0,
        "invalid_citation_rate": 100.0 * sum(1 for r in rs if r.invalid_citations) / len(rs) if rs else 0.0,
        "p50_ms": st.median(lat) if lat else 0.0,
        "p95_ms": lat[int(len(lat) * 0.95)] if lat else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--strategy", default="fixed")
    ap.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = spec["questions"]

    print(f"{len(questions)} questions | {args.retriever}/{args.strategy} k={args.k}")
    print(f"judge: {'off' if args.no_judge else 'on'}\n")

    retriever = build(args.retriever, args.strategy)
    llm = build_llm()

    results: list[GResult] = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=args.k)
        r_ms = (time.perf_counter() - t0) * 1000

        ans = answer(q["question"], hits, llm)
        faith = 1.0
        if not args.no_judge:
            faith, _ = judge(q["question"], hits, ans, llm)

        res = score_one(q, hits, ans, faith)
        res.retrieval_ms = r_ms
        results.append(res)

        mark = "ok " if res.exit_correct and res.numeric_correct else "   "
        print(
            f"  [{mark}] {q['id']:16s} {res.exit:8s} (want {res.expected_exit:8s}) "
            f"num={'Y' if res.numeric_correct else '.'} "
            f"faith={faith:.2f} {res.generation_ms:6.0f}ms"
        )

    s = summarise(results)
    print("\n" + "=" * 84)
    print("GENERATION SCORECARD")
    print("=" * 84)
    print(f"  exit correct         {s['exit_correct']:6.1f}%")
    print(f"  numeric correct      {s['numeric_correct']:6.1f}%   (figure stated in the answer)")
    print(f"  both sides shown     {s['both_sides']:6.1f}%   (restatement + split questions)")
    print(f"  faithfulness         {s['faithfulness']:6.2f}    (0-1, LLM judge)")
    print(f"  hallucination rate   {s['hallucination_rate']:6.1f}%   (figure with no source passage)")
    print(f"  invalid citations    {s['invalid_citation_rate']:6.1f}%")
    print(f"  latency p50 / p95    {s['p50_ms']:6.0f} / {s['p95_ms']:.0f} ms")

    print("\n" + "=" * 84)
    print("BY CATEGORY")
    print("=" * 84)
    groups: dict[str, list[GResult]] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    print(f"{'category':14s} {'n':>3s} {'exit':>7s} {'numeric':>9s} {'faith':>7s}")
    print("-" * 46)
    for cat in ("numeric", "temporal", "trend", "restatement", "split", "unanswerable"):
        if cat not in groups:
            continue
        g = summarise(groups[cat])
        num = "n/a" if g["numeric_correct"] != g["numeric_correct"] else f"{g['numeric_correct']:8.0f}%"
        print(f"{cat:14s} {len(groups[cat]):3d} {g['exit_correct']:6.0f}% {num:>9s} {g['faithfulness']:7.2f}")

    bad = [r for r in results if r.hallucinated]
    if bad:
        print(f"\n{len(bad)} answer(s) contained an unsourced figure:")
        for r in bad:
            print(f"  {r.qid}: {r.hallucinated[:5]}")

    REPORTS_DIR.mkdir(exist_ok=True)
    dest = REPORTS_DIR / "generation_scorecard.json"
    dest.write_text(
        json.dumps(
            {
                "config": {
                    "retriever": args.retriever, "strategy": args.strategy, "k": args.k,
                },
                "summary": s,
                "by_category": {c: summarise(g) for c, g in groups.items()},
                "questions": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
