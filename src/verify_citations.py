"""Assert no block has a blank citation, and show citations for key figures.

A blank citation cannot be scored for groundedness, and the restatement and
split questions depend entirely on knowing WHICH filing a figure came from --
the same number is correct or incorrect according to its source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"

SPOTCHECKS = [
    (2026, "1,481", "Taiwan"),
    (2025, "20,573", "Taiwan"),
    (2026, "77,482", "United States"),
    (2026, "87,960", "Total"),
    (2025, "81,453", "Total"),
    (2026, "193,737", "Data Center"),
    (2026, "230", "Patents and licensed technology"),
    (2024, "12.05", "Basic"),
    (2025, "1.21", "Basic"),
]


def blocks_for(fy: int) -> list[dict]:
    path = CLEAN_DIR / f"FY{fy}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/clean.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    data = {fy: blocks_for(fy) for fy in (2024, 2025, 2026)}
    failures: list[str] = []

    print("=== citation coverage ===")
    for fy, blocks in data.items():
        tables = [b for b in blocks if b["block_type"] == "table"]
        prose = [b for b in blocks if b["block_type"] == "prose"]
        no_cap = sum(1 for b in tables if not b["table_caption"])
        no_cit = sum(1 for b in blocks if not b["citation"])
        print(
            f"  FY{fy}: {len(tables)} tables ({no_cap} without caption), "
            f"{len(prose)} prose, blank citations: {no_cit}"
        )
        if no_cit:
            failures.append(f"FY{fy} has {no_cit} blocks with a blank citation")

    print("\n=== citations for figures the eval set depends on ===")
    for fy, figure, label in SPOTCHECKS:
        hit = next(
            (
                b
                for b in data[fy]
                if b["block_type"] == "table"
                and figure in b["text"]
                and label.lower() in b["text"].lower()
            ),
            None,
        )
        if hit is None:
            failures.append(f"FY{fy} {figure} ({label}) not found in a table block")
            print(f"  [MISS] FY{fy} {figure:>9s} {label}")
            continue
        print(f"  FY{fy} {figure:>9s} {label:32s} -> {hit['citation']}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("every block carries a non-empty citation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
