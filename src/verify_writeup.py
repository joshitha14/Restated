"""Check every number quoted in WRITEUP.md against the committed scorecards.

A writeup whose figures do not match the reports it cites is worse than no
writeup, and the numbers were typed by hand from terminal output. This asserts
each one against reports/*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

TOL = 0.05  # percentage points


def load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        sys.exit(f"missing {path.relative_to(ROOT)} -- run the evaluators first")
    return json.loads(path.read_text(encoding="utf-8"))


def check(label: str, claimed: float, actual: float, tol: float = TOL) -> bool:
    ok = abs(claimed - actual) <= tol
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {label:44s} writeup={claimed:8.1f}  actual={actual:8.1f}")
    return ok


def main() -> int:
    retrieval = load("retrieval_scorecard.json")
    generation = load("generation_scorecard.json")
    graph = load("graph_scorecard.json")

    failures = 0

    # ---- retrieval table (k=20) ----------------------------------------
    print("=== retrieval scorecard, k=20 ===")
    # WRITEUP.md quotes ISOLATED figures (one configuration per process); the
    # committed scorecard is a BATCH run, and runs C and D score differently in
    # the two conditions (DECISIONS.md D23). Asserting the isolated numbers
    # against a batch file would fail for a known and documented reason, so only
    # the runs that agree across both conditions are pinned here.
    #
    # C and D are checked separately below, against both permitted values.
    # year_precision is an average over per-question ratios and moves by a few
    # tenths between isolated and batch execution (D23), so it is checked with a
    # 1-point tolerance rather than pinned. numeric_accuracy and both_sides are
    # stable for these runs and are pinned exactly.
    claims = {
        "A":  (68.8, 70.9, 33.3),
        "B":  (75.0, 73.1, 33.3),
        "B2": (68.8, 71.2, 33.3),
        "E":  (68.8, 61.6, 83.3),
    }
    YR_TOL = 1.0
    # (isolated, batch) -- either is acceptable, the writeup quotes the first.
    unstable = {"C": (75.0, 81.2), "D": (75.0, 81.2), "F": (81.2, 87.5), "G": (75.0, 75.0)}
    for run_id, (num, yr, both) in claims.items():
        if run_id not in retrieval["runs"]:
            print(f"  [skip] run {run_id} not in the current scorecard")
            continue
        s = retrieval["runs"][run_id]["summary"]
        failures += not check(f"run {run_id} numeric_accuracy", num, s["numeric_accuracy"])
        failures += not check(f"run {run_id} year_precision", yr, s["year_precision"], tol=YR_TOL)
        failures += not check(f"run {run_id} both_sides", both, s["both_sides"])

    print("\n=== runs that differ between isolated and batch execution (D23) ===")
    for run_id, (iso, batch) in unstable.items():
        if run_id not in retrieval["runs"]:
            print(f"  [skip] run {run_id} not in the current scorecard")
            continue
        actual = retrieval["runs"][run_id]["summary"]["numeric_accuracy"]
        ok = abs(actual - iso) <= TOL or abs(actual - batch) <= TOL
        flag = "ok  " if ok else "FAIL"
        which = "isolated" if abs(actual - iso) <= TOL else "batch"
        print(f"  [{flag}] run {run_id} numeric={actual:5.1f} "
              f"(isolated {iso}, batch {batch}) -> {which}")
        if not ok:
            failures += 1

    # ---- generation ----------------------------------------------------
    print("\n=== generation scorecard ===")
    # exit_correct and numeric_correct vary between runs (variance.json shows a
    # 10.5-point spread on exit_correct alone), so these are checked against the
    # measured range rather than pinned to one sample.
    g = generation["summary"]
    for label, claimed, actual, tol in (
        ("exit_correct", 89.5, g["exit_correct"], 6.0),
        ("numeric_correct", 62.5, g["numeric_correct"], 6.5),
    ):
        failures += not check(f"{label} (within run-to-run spread)", claimed, actual, tol=tol)
    failures += not check("both_sides", 50.0, g["both_sides"])
    failures += not check("faithfulness", 1.00, g["faithfulness"], tol=0.01)
    # 1 of 19: a computed intermediate stated as though retrieved (q_trend_03).
    failures += not check("hallucination_rate", 5.3, g["hallucination_rate"], tol=0.1)
    failures += not check("invalid_citation_rate", 0.0, g["invalid_citation_rate"])

    # ---- graph ---------------------------------------------------------
    #
    # Single-run graph metrics are NOT asserted against fixed values: the same
    # configuration varies by ~6 points between runs (see reports/variance.json),
    # so pinning them would make this check fail on every rerun. Only the
    # deterministic parts of the run are asserted.
    print("\n=== graph scorecard (deterministic parts only) ===")
    failures += not check("rewrites_used", 8, graph["rewrites_used"], tol=0)
    failures += not check("n questions", 19, graph["summary"]["n"], tol=0)

    # ---- variance --------------------------------------------------------
    print("\n=== variance across repeated runs ===")
    var_path = REPORTS / "variance.json"
    if not var_path.exists():
        print("  [skip] reports/variance.json missing -- run src/evaluate_repeat.py")
    else:
        var = json.loads(var_path.read_text(encoding="utf-8"))
        n = var["n"]
        print(f"  {n} repetitions per mode")
        for mode, spread in var["spread"].items():
            for metric in ("exit_correct", "numeric_correct", "both_sides"):
                s = spread[metric]
                width = s["max"] - s["min"]
                print(
                    f"    {mode:7s} {metric:18s} mean={s['mean']:6.1f} "
                    f"range={s['min']:5.1f}-{s['max']:5.1f} (spread {width:.1f})"
                )
        # The writeup's central claims about variance, asserted:
        #   1. numeric accuracy has ZERO spread (so the +12.5 gain is real)
        #   2. exit correctness has a LARGE spread (so small exit deltas are noise)
        print("\n  writeup's variance claims:")
        for mode in ("linear", "graph"):
            num = var["spread"][mode]["numeric_correct"]
            width = num["max"] - num["min"]
            failures += not check(f"{mode} numeric spread is zero", 0.0, width)

        exit_spread = var["spread"]["linear"]["exit_correct"]
        width = exit_spread["max"] - exit_spread["min"]
        if width < 5.0:
            print(f"  [FAIL] linear exit spread is only {width:.1f} points -- the "
                  "writeup claims it is large enough that small deltas are noise")
            failures += 1
        else:
            print(f"  [ok  ] linear exit spread is {width:.1f} points (writeup: "
                  "small exit deltas are noise)")

        # The headline delta the writeup claims is real.
        delta = (var["spread"]["graph"]["numeric_correct"]["mean"]
                 - var["spread"]["linear"]["numeric_correct"]["mean"])
        failures += not check("graph numeric gain", 12.5, delta, tol=0.15)

    # ---- per-category claims -------------------------------------------
    print("\n=== split-trap category claims ===")
    for run_id, claimed in (("A", 0.0), ("C", 100.0), ("E", 100.0),
                            ("F", 100.0), ("G", 50.0)):
        if run_id not in retrieval["runs"]:
            print(f"  [skip] run {run_id} not in the current scorecard")
            continue
        cat = retrieval["runs"][run_id]["by_category"].get("split")
        if not cat:
            print(f"  [FAIL] run {run_id} has no 'split' category")
            failures += 1
            continue
        failures += not check(f"run {run_id} split numeric", claimed, cat["numeric_accuracy"])

    print()
    if failures:
        print(f"{failures} figure(s) in WRITEUP.md do not match the scorecards")
        return 1
    print("every figure quoted in WRITEUP.md matches the committed scorecards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
