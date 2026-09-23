"""Check that cleaned blocks preserve the figures eval/questions.yaml depends on.

The cleaner is the biggest time risk in the build (handoff §5), and a cleaner
that silently drops or mis-aligns a figure invalidates every downstream
measurement. This asserts each critical figure survives into a block, and that
table blocks kept their column alignment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"

# (fiscal year of filing, figure as printed, expected block_type, label it sits with)
CHECKS = [
    (2026, "193,737", "table", "Data Center"),
    (2026, "215,938", "table", "Total revenue"),
    (2026, "31,376", "table", "Networking"),
    (2026, "115,186", "table", "Data Center"),
    (2026, "47,525", "table", "Data Center"),
    (2026, "1,481", "table", "Taiwan"),
    (2025, "20,573", "table", "Taiwan"),
    (2026, "77,482", "table", "United States"),
    (2025, "61,257", "table", "United States"),
    (2026, "87,960", "table", "Total"),
    (2025, "81,453", "table", "Total"),
    (2025, "130,497", "table", "Total revenue"),
    (2024, "60,922", "table", "Total revenue"),
    (2025, "1.21", "table", "Basic"),
    (2024, "12.05", "table", "Basic"),
    (2025, "24,940", "table", "Diluted weighted average shares"),
    (2024, "2,494", "table", "Diluted weighted average shares"),
]


def load(fy: int) -> list[dict]:
    path = CLEAN_DIR / f"FY{fy}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/clean.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    blocks = {fy: load(fy) for fy in (2024, 2025, 2026)}
    failures: list[str] = []

    print("=== critical figures survive cleaning ===")
    for fy, figure, want_type, label in CHECKS:
        hits = [
            b for b in blocks[fy]
            if figure in b["text"] and label.lower() in b["text"].lower()
        ]
        typed = [b for b in hits if b["block_type"] == want_type]
        ok = bool(typed)
        if not ok:
            failures.append(f"FY{fy} {figure} ({label}) not in a {want_type} block")
        cap = (typed[0]["table_caption"][:44] if typed else "")
        yrs = (typed[0]["years_present"] if typed else [])
        print(
            f"  [{'ok ' if ok else 'MISS'}] FY{fy} {figure:>9s} with {label:32s}"
            f" blocks={len(hits)} years={yrs}  {cap}"
        )

    # Column alignment: the revenue table must still have Data Center's three
    # figures in the right order on one row.
    print("\n=== column alignment (problem 2: $ in its own cell) ===")
    for fy, expect in ((2026, ["193,737", "115,186", "47,525"]),
                       (2025, ["115,186", "47,525", "15,005"])):
        found = False
        for b in blocks[fy]:
            if b["block_type"] != "table" or "Data Center" not in b["text"]:
                continue
            for line in b["text"].splitlines():
                if not line.startswith("| Data Center"):
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                nums = [c for c in cells if c and c[0].isdigit()]
                ok = nums[: len(expect)] == expect
                print(f"  [{'ok ' if ok else 'MISS'}] FY{fy} Data Center row -> {nums[:4]}")
                if not ok:
                    failures.append(f"FY{fy} Data Center columns mis-aligned: {nums[:4]}")
                found = True
                break
            if found:
                break
        if not found:
            failures.append(f"FY{fy} no Data Center table row found")
            print(f"  [MISS] FY{fy} no '| Data Center' row in any table block")

    # No $-only or empty columns should remain.
    print("\n=== residual symbol-only cells ===")
    for fy in (2024, 2025, 2026):
        bad = 0
        for b in blocks[fy]:
            if b["block_type"] != "table":
                continue
            for line in b["text"].splitlines():
                if line.startswith("|") and " $ |" in line:
                    bad += 1
        print(f"  FY{fy}: {bad} rows still containing a lone '$' cell")
        if bad:
            failures.append(f"FY{fy} has {bad} rows with un-folded $ cells")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("cleaner preserves all critical figures with correct column alignment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
