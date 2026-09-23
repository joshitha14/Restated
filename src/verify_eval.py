"""Verify every hardcoded figure in eval/questions.yaml against the filings.

An eval set with a wrong expected value silently scores a correct system as
wrong, which is worse than having no eval set. This checks each asserted figure
appears in the filing it is attributed to.

Run after any edit to questions.yaml.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Period end per fiscal year; the extension varies (.htm from EDGAR, .html from
# a browser-saved copy), so it is resolved at runtime rather than hardcoded.
PERIOD_BY_FY = {2026: "20260125", 2025: "20250126", 2024: "20240128"}


def filing_path(fy: int) -> Path:
    for ext in (".htm", ".html"):
        path = RAW_DIR / f"nvda-{PERIOD_BY_FY[fy]}{ext}"
        if path.exists():
            return path
    raise SystemExit(
        f"no filing for FY{fy} in {RAW_DIR}\n"
        "run:  SEC_USER_AGENT='Your Name you@example.com' python src/fetch.py"
    )

# (description, fiscal-year-of-filing, figure-as-printed)
CHECKS = [
    ("q_numeric_01  Data Center FY2026", 2026, "193,737"),
    ("q_numeric_02  total revenue FY2026", 2026, "215,938"),
    ("q_numeric_03  Networking FY2026", 2026, "31,376"),
    ("q_numeric_04  Gaming FY2025 (in FY2026 filing)", 2026, "11,350"),
    ("q_numeric_04  Gaming FY2025 (in FY2025 filing)", 2025, "11,350"),
    ("q_temporal_01 Data Center FY2025 (in FY2026 filing)", 2026, "115,186"),
    ("q_temporal_02 total revenue FY2024 (in FY2026 filing)", 2026, "60,922"),
    ("q_temporal_02 total revenue FY2024 (in FY2024 filing)", 2024, "60,922"),
    ("q_temporal_03 total revenue FY2025 (in FY2025 filing)", 2025, "130,497"),
    ("q_trend_01    Data Center FY2024 (in FY2026 filing)", 2026, "47,525"),
    ("q_restate_01  US FY2025 per FY2026 filing", 2026, "77,482"),
    ("q_restate_01  US FY2025 per FY2025 filing", 2025, "61,257"),
    ("q_restate_02  China FY2025 per FY2026 filing", 2026, "25,048"),
    ("q_restate_02  China FY2025 per FY2025 filing", 2025, "17,108"),
    ("q_restate_03  segment op income FY2025 per FY2026", 2026, "87,960"),
    ("q_restate_03  segment op income FY2025 per FY2025", 2025, "81,453"),
    ("q_restate_04  patents FY2025 per FY2026 filing", 2026, "230"),
    ("q_restate_04  patents FY2025 per FY2025 filing", 2025, "449"),
    ("q_split_01    basic EPS FY2024 post-split", 2025, "1.21"),
    ("q_split_01    basic EPS FY2024 pre-split", 2024, "12.05"),
    ("q_split_02    diluted shares FY2024 post-split", 2025, "24,940"),
    ("q_split_02    diluted shares FY2024 pre-split", 2024, "2,494"),
]


def load_text(fy: int) -> str:
    html = filing_path(fy).read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"<[^>]+>", " ", html)
    txt = txt.replace("&#160;", " ").replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", txt)


def main() -> int:
    texts = {fy: load_text(fy) for fy in PERIOD_BY_FY}
    failures = []

    for desc, fy, figure in CHECKS:
        count = texts[fy].count(figure)
        ok = count > 0
        flag = "ok " if ok else "MISS"
        print(f"  [{flag}] {desc:52s} {figure:>9s}  (x{count} in FY{fy} filing)")
        if not ok:
            failures.append((desc, fy, figure))

    print()
    if failures:
        print(f"{len(failures)} FAILED — eval/questions.yaml has wrong expected values:")
        for desc, fy, figure in failures:
            print(f"  {figure} not found in FY{fy} filing  ({desc})")
        return 1

    print(f"all {len(CHECKS)} asserted figures verified present in their source filing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
