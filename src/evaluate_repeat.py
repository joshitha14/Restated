"""Run an evaluation N times and report the spread.

`temperature=0.0` is not determinism. Two runs of the same configuration
produced numeric accuracy of 81.2% and 75.0%, and exit correctness of 84.2% and
78.9% -- a 6-point spread on n=19 questions, which is the same order as the
differences being reported BETWEEN configurations.

Quoting one run as if it were the measurement is therefore misleading. This runs
the eval several times and reports mean, min and max, so the writeup can state
the spread rather than a single sample.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

REPORTS = ROOT / "reports"

METRICS = (
    "exit_correct",
    "numeric_correct",
    "both_sides",
    "faithfulness",
    "hallucination_rate",
    "p50_ms",
)


def run_graph_once(judge_on: bool) -> dict:
    import yaml

    from evaluate_gen import score_one, summarise
    from generate import build_llm, judge
    from graph import ask, build_graph

    spec = yaml.safe_load((ROOT / "eval/questions.yaml").read_text(encoding="utf-8"))
    app = build_graph(strategy="fixed", kind="hybrid")
    llm = build_llm() if judge_on else None

    results = []
    for q in spec["questions"]:
        state = ask(app, q["question"])
        ans = state["answer"]
        hits = state.get("hits", [])
        faith = 1.0
        if judge_on:
            faith, _ = judge(q["question"], hits, ans, llm)
        res = score_one(q, hits, ans, faith)
        res.retrieval_ms = state.get("retrieval_ms", 0.0)
        res.generation_ms = state.get("generation_ms", 0.0)
        results.append(res)

    return summarise(results)


def run_linear_once(judge_on: bool) -> dict:
    import yaml

    from evaluate_gen import score_one, summarise
    from generate import answer, build_llm, judge
    from retrieve import build

    spec = yaml.safe_load((ROOT / "eval/questions.yaml").read_text(encoding="utf-8"))
    retriever = build("hybrid", "fixed")
    llm = build_llm()

    results = []
    for q in spec["questions"]:
        hits = retriever.search(q["question"], k=20)
        ans = answer(q["question"], hits, llm)
        faith = 1.0
        if judge_on:
            faith, _ = judge(q["question"], hits, ans, llm)
        res = score_one(q, hits, ans, faith)
        res.generation_ms = ans.latency_ms
        results.append(res)

    return summarise(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="repetitions")
    ap.add_argument("--mode", choices=("graph", "linear", "both"), default="both")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    modes = ["linear", "graph"] if args.mode == "both" else [args.mode]
    out: dict[str, list[dict]] = {}

    for mode in modes:
        out[mode] = []
        runner = run_graph_once if mode == "graph" else run_linear_once
        for i in range(args.n):
            s = runner(not args.no_judge)
            out[mode].append(s)
            print(
                f"  {mode:7s} run {i + 1}: exit={s['exit_correct']:5.1f}% "
                f"numeric={s['numeric_correct']:5.1f}% both={s['both_sides']:5.1f}% "
                f"halluc={s['hallucination_rate']:4.1f}% p50={s['p50_ms']:6.0f}ms"
            )
        print()

    print("=" * 78)
    print(f"SPREAD ACROSS {args.n} RUNS  (temperature=0.0 is not determinism)")
    print("=" * 78)
    print(f"{'mode':8s} {'metric':20s} {'mean':>8s} {'min':>8s} {'max':>8s} {'spread':>8s}")
    print("-" * 64)
    for mode, runs in out.items():
        for m in METRICS:
            vals = [r[m] for r in runs]
            print(
                f"{mode:8s} {m:20s} {st.mean(vals):8.1f} {min(vals):8.1f} "
                f"{max(vals):8.1f} {max(vals) - min(vals):8.1f}"
            )
        print()

    REPORTS.mkdir(exist_ok=True)
    dest = REPORTS / "variance.json"
    dest.write_text(
        json.dumps(
            {
                "n": args.n,
                "runs": out,
                "spread": {
                    mode: {
                        m: {
                            "mean": st.mean([r[m] for r in runs]),
                            "min": min(r[m] for r in runs),
                            "max": max(r[m] for r in runs),
                        }
                        for m in METRICS
                    }
                    for mode, runs in out.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
