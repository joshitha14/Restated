"""Assert the chunking experiment is validly controlled.

The comparison between the two strategies is only meaningful if they differ in
ONE respect: where prose boundaries fall. This checks that.

If any assertion here fails, run A vs run B measures something other than
chunking strategy and the headline result is void.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "chunks"
FYS = (2024, 2025, 2026)

# Figures the eval set depends on; each must survive intact in BOTH strategies.
CRITICAL = [
    (2026, "193,737", "Data Center"),
    (2026, "215,938", "Total revenue"),
    (2026, "31,376", "Networking"),
    (2026, "115,186", "Data Center"),
    (2026, "47,525", "Data Center"),
    (2026, "77,482", "United States"),
    (2025, "61,257", "United States"),
    (2026, "25,048", "China"),
    (2025, "17,108", "China"),
    (2026, "87,960", "Total"),
    (2025, "81,453", "Total"),
    (2026, "230", "Patents and licensed technology"),
    (2025, "449", "Patents and licensed technology"),
    (2025, "130,497", "Total revenue"),
    (2024, "60,922", "Total revenue"),
    (2025, "1.21", "Basic"),
    (2024, "12.05", "Basic"),
    (2025, "24,940", "Diluted weighted average shares"),
    (2024, "2,494", "Diluted weighted average shares"),
]


def load(strategy: str, fy: int) -> list[dict]:
    path = CHUNK_DIR / strategy / f"FY{fy}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/chunk.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    data = {s: {fy: load(s, fy) for fy in FYS} for s in ("fixed", "semantic")}
    failures: list[str] = []

    # --- CONTROL 1: tables held whole and identical in both strategies -------
    print("=== control 1: tables identical across strategies ===")
    for fy in FYS:
        ftab = [c for c in data["fixed"][fy] if c["block_type"] == "table"]
        stab = [c for c in data["semantic"][fy] if c["block_type"] == "table"]
        same_count = len(ftab) == len(stab)
        ftexts = sorted(c["text"] for c in ftab)
        stexts = sorted(c["text"] for c in stab)
        identical = ftexts == stexts
        print(
            f"  FY{fy}: fixed={len(ftab)} semantic={len(stab)} "
            f"count_match={same_count} text_identical={identical}"
        )
        if not same_count:
            failures.append(f"FY{fy}: table count differs ({len(ftab)} vs {len(stab)})")
        if not identical:
            failures.append(f"FY{fy}: table TEXT differs between strategies")

    # --- CONTROL 2: no table was split --------------------------------------
    print("\n=== control 2: no table split (header row present in every table chunk) ===")
    for strategy in ("fixed", "semantic"):
        headerless = 0
        for fy in FYS:
            for c in data[strategy][fy]:
                if c["block_type"] != "table":
                    continue
                # A markdown table keeps its separator row; losing it means a cut.
                if "---" not in c["text"]:
                    headerless += 1
        print(f"  {strategy:9s}: {headerless} table chunks missing a header separator")
        if headerless:
            failures.append(f"{strategy}: {headerless} table chunks lost their header")

    # --- CONTROL 3: comparable prose size (not a size experiment) -----------
    print("\n=== control 3: prose chunk sizes comparable ===")
    stats = {}
    for strategy in ("fixed", "semantic"):
        sizes = sorted(
            c["n_chars"]
            for fy in FYS
            for c in data[strategy][fy]
            if c["block_type"] == "prose"
        )
        med = sizes[len(sizes) // 2]
        stats[strategy] = (len(sizes), med, sizes[-1])
        print(f"  {strategy:9s}: n={len(sizes)}, median={med}, max={sizes[-1]}")

    fmed, smed = stats["fixed"][1], stats["semantic"][1]
    ratio = max(fmed, smed) / min(fmed, smed)
    print(f"  median ratio: {ratio:.2f}x  (want < 1.5x)")
    if ratio >= 1.5:
        failures.append(f"prose median sizes differ by {ratio:.2f}x -- size confound")

    for strategy, (_, _, mx) in stats.items():
        if mx > 2000:
            failures.append(f"{strategy}: oversized prose chunk ({mx} chars)")

    # --- CONTROL 4: critical figures survive in both ------------------------
    print("\n=== control 4: eval figures present in both strategies ===")
    for fy, figure, label in CRITICAL:
        row = []
        for strategy in ("fixed", "semantic"):
            hits = [
                c for c in data[strategy][fy]
                if figure in c["text"] and label.lower() in c["text"].lower()
            ]
            row.append(len(hits))
            if not hits:
                failures.append(f"{strategy} FY{fy}: {figure} ({label}) missing")
        flag = "ok " if all(r > 0 for r in row) else "MISS"
        print(f"  [{flag}] FY{fy} {figure:>9s} {label:32s} fixed={row[0]} semantic={row[1]}")

    # --- CONTROL 5: metadata intact ----------------------------------------
    print("\n=== control 5: metadata completeness ===")
    for strategy in ("fixed", "semantic"):
        allc = [c for fy in FYS for c in data[strategy][fy]]
        no_cit = sum(1 for c in allc if not c["citation"])
        no_sec = sum(1 for c in allc if not c["item_section"])
        no_fy = sum(1 for c in allc if not c["fiscal_year"])
        dupe = len(allc) - len({c["chunk_id"] for c in allc})
        print(
            f"  {strategy:9s}: n={len(allc)}, blank citation={no_cit}, "
            f"blank section={no_sec}, blank fy={no_fy}, duplicate ids={dupe}"
        )
        for name, count in (
            ("citation", no_cit), ("item_section", no_sec),
            ("fiscal_year", no_fy), ("duplicate chunk_id", dupe),
        ):
            if count:
                failures.append(f"{strategy}: {count} chunks with bad {name}")

    print()
    if failures:
        print(f"{len(failures)} CONTROL FAILURE(S) -- the A/B comparison is not valid:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("experiment is validly controlled: strategies differ only in prose boundaries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
