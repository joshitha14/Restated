"""Per-question comparison across runs.

An identical aggregate score can mean the runs behave identically, or that they
succeed on different questions and the totals coincide. Only a per-question view
distinguishes them, and the difference matters: the first says the variable has
no effect, the second says it has effects that cancel.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "reports/retrieval_scorecard.json").read_text(encoding="utf-8"))
runs = data["runs"]
names = list(runs)
qids = [q["qid"] for q in runs[names[0]]["questions"]]

print("numeric_hit (Y) and rank of first chunk holding the answer\n")
print(f"{'qid':16s} {'category':13s} " + " ".join(f"{n:>7s}" for n in names))
print("-" * (30 + 8 * len(names)))

differing = []
for i, qid in enumerate(qids):
    cat = runs[names[0]]["questions"][i]["category"]
    cells, hits = [], []
    for n in names:
        q = runs[n]["questions"][i]
        hit = q["numeric_hit"]
        rank = q.get("numeric_rank")
        hits.append(hit)
        cells.append(f"{'Y' if hit else '.'}{'@' + str(rank) if rank else '':>5s}")
    if len(set(hits)) > 1:
        differing.append(qid)
    print(f"{qid:16s} {cat:13s} " + " ".join(f"{c:>7s}" for c in cells))

print(f"\nquestions where runs DIFFER on numeric_hit: {len(differing)}")
for q in differing:
    print(f"  {q}")

print("\nper-run totals")
for n in names:
    s = runs[n]["summary"]
    scored = [q for q in runs[n]["questions"] if q["category"] != "unanswerable"]
    won = sum(1 for q in scored if q["numeric_hit"])
    print(f"  {n:4s} {won}/{len(scored)} = {s['numeric_accuracy']:.1f}%  mrr={s['numeric_mrr']:.1f}")
