"""Is the answer chunk reachable at all, and at what depth?

A retrieval metric that reports 0% can mean two very different things: the
answer is not in the index, or it is there but ranked below k. They call for
opposite fixes, so this checks directly by searching deep and reporting the rank
of the first chunk containing the expected figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import expected_figures  # noqa: E402
from retrieve import build  # noqa: E402

DEEP = 50

spec = yaml.safe_load((ROOT / "eval/questions.yaml").read_text(encoding="utf-8"))
QUESTIONS = spec["questions"]

CHECK = [
    "q_numeric_01", "q_numeric_03", "q_numeric_04",
    "q_temporal_01", "q_restate_01", "q_restate_02",
    "q_split_01", "q_split_02",
]


def main() -> None:
    by_id = {q["id"]: q for q in QUESTIONS}
    for kind, strategy in (("dense", "fixed"), ("bm25", "fixed"), ("hybrid", "fixed")):
        r = build(kind, strategy)
        print(f"\n{'=' * 78}\n{kind} / {strategy}   (searching to depth {DEEP})\n{'=' * 78}")
        for qid in CHECK:
            q = by_id[qid]
            figs = expected_figures(q)
            hits = r.search(q["question"], k=DEEP)
            rank = None
            which = None
            for i, h in enumerate(hits, start=1):
                match = next((f for f in figs if f in h.text), None)
                if match:
                    rank, which = i, match
                    break
            verdict = f"rank {rank:2d}  found {which!r}" if rank else f"NOT in top {DEEP}"
            print(f"  {qid:16s} {verdict}")


if __name__ == "__main__":
    main()
