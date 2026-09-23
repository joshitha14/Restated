"""Does the LangGraph loop beat the linear path? Measured, not assumed.

A retry loop that costs latency and credits has to earn them. This runs the same
eval set through the graph and compares against the linear generation scorecard,
reporting what the loop bought and what it cost.

Run src/evaluate_gen.py first so there is a baseline to compare against.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate_gen import expected_exit, score_one, summarise  # noqa: E402
from graph import K_FIRST, ask, build_graph  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
REPORTS_DIR = ROOT / "reports"
BASELINE = REPORTS_DIR / "generation_scorecard.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="fixed")
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = spec["questions"]

    app = build_graph(strategy=args.strategy, kind=args.retriever)
    print(f"{len(questions)} questions through the graph "
          f"({args.retriever}/{args.strategy}, k={K_FIRST} then 20 on retry)\n")

    results = []
    rewrites_used = 0
    refused_by_verify = 0

    for q in questions:
        state = ask(app, q["question"])
        ans = state["answer"]
        hits = state.get("hits", [])

        faith = 1.0
        if not args.no_judge:
            from generate import build_llm, judge

            faith, _ = judge(q["question"], hits, ans, build_llm())

        res = score_one(q, hits, ans, faith)
        res.retrieval_ms = state.get("retrieval_ms", 0.0)
        res.generation_ms = state.get("generation_ms", 0.0)
        results.append(res)

        rw = state.get("rewrites", 0)
        rewrites_used += rw
        if any("ungrounded answer replaced" in t for t in state.get("trace", [])):
            refused_by_verify += 1

        mark = "ok " if res.exit_correct and res.numeric_correct else "   "
        print(
            f"  [{mark}] {q['id']:16s} {res.exit:8s} (want {res.expected_exit:8s}) "
            f"num={'Y' if res.numeric_correct else '.'} rewrites={rw} "
            f"{res.retrieval_ms + res.generation_ms:6.0f}ms"
        )

    s = summarise(results)

    print("\n" + "=" * 84)
    print("GRAPH SCORECARD")
    print("=" * 84)
    print(f"  exit correct         {s['exit_correct']:6.1f}%")
    print(f"  numeric correct      {s['numeric_correct']:6.1f}%")
    print(f"  both sides shown     {s['both_sides']:6.1f}%")
    print(f"  faithfulness         {s['faithfulness']:6.2f}")
    print(f"  hallucination rate   {s['hallucination_rate']:6.1f}%")
    print(f"  latency p50 / p95    {s['p50_ms']:6.0f} / {s['p95_ms']:.0f} ms")
    print(f"\n  rewrites fired       {rewrites_used} across {len(questions)} questions")
    print(f"  refused by verify    {refused_by_verify}")

    # ---- comparison against the linear path ------------------------------
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        b = base["summary"]
        print("\n" + "=" * 84)
        print("GRAPH vs LINEAR  (does the loop earn its latency?)")
        print("=" * 84)
        print(f"{'metric':22s} {'linear':>10s} {'graph':>10s} {'delta':>10s}")
        print("-" * 56)
        for key, label, unit in (
            ("exit_correct", "exit correct", "%"),
            ("numeric_correct", "numeric correct", "%"),
            ("both_sides", "both sides", "%"),
            ("faithfulness", "faithfulness", ""),
            ("hallucination_rate", "hallucination", "%"),
            ("p50_ms", "p50 latency", "ms"),
            ("p95_ms", "p95 latency", "ms"),
        ):
            bv, gv = b.get(key, 0.0), s.get(key, 0.0)
            delta = gv - bv
            sign = "+" if delta >= 0 else ""
            print(f"{label:22s} {bv:9.1f}{unit:1s} {gv:9.1f}{unit:1s} {sign}{delta:9.1f}")
    else:
        print(f"\n(no baseline at {BASELINE.relative_to(ROOT)}; run src/evaluate_gen.py first)")

    # ---- per category ----------------------------------------------------
    groups: dict[str, list] = defaultdict(list)
    for r in results:
        groups[r.category].append(r)
    print("\n" + "=" * 84)
    print("BY CATEGORY")
    print("=" * 84)
    print(f"{'category':14s} {'n':>3s} {'exit':>7s} {'numeric':>9s}")
    print("-" * 38)
    for cat in ("numeric", "temporal", "trend", "restatement", "split", "unanswerable"):
        if cat not in groups:
            continue
        g = summarise(groups[cat])
        print(f"{cat:14s} {len(groups[cat]):3d} {g['exit_correct']:6.0f}% {g['numeric_correct']:8.0f}%")

    REPORTS_DIR.mkdir(exist_ok=True)
    dest = REPORTS_DIR / "graph_scorecard.json"
    dest.write_text(
        json.dumps(
            {
                "config": {"retriever": args.retriever, "strategy": args.strategy,
                           "k_first": K_FIRST},
                "summary": s,
                "rewrites_used": rewrites_used,
                "refused_by_verify": refused_by_verify,
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
