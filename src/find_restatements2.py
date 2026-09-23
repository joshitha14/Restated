"""Restatement scan over CLEANED blocks, keyed on the caption.

Supersedes the raw-HTML scan in find_restatements.py, which keyed on
(caption, label) using best-effort captions and therefore still collided: it
reported Taiwan FY2025 as a 13.9x "restatement" by comparing a LONG-LIVED
ASSETS table in the FY2026 filing against a REVENUE table in the FY2025 filing.
Both rows are labelled 'Taiwan'.

That was the D3 failure mode reproducing itself inside the diagnostic. This
version reads data/clean/*.jsonl, where captions are reliable, and requires the
captions to describe the same KIND of table before comparing figures.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"

NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?%?$")


def to_float(s: str) -> float | None:
    s = s.strip().replace("$", "").replace(",", "").replace("%", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def topic(caption: str) -> str:
    """Coarse table subject, so unlike tables are never compared.

    NVIDIA reworded several captions between filings (notably the geographic
    revenue basis change), so the raw caption cannot be the join key either.
    """
    c = caption.lower()
    if "long-lived" in c or "long lived" in c:
        return "long_lived_assets"
    if "revenue by geographic" in c or "geographic area" in c:
        return "revenue_by_geography"
    if "specialized market" in c or "end market" in c:
        return "revenue_by_market"
    if "operating income by reportable" in c or "reportable segment" in c:
        return "segment_operating_income"
    if "amortizable intangible" in c or "intangible asset" in c:
        return "intangible_assets"
    if "statements of income" in c:
        return "income_statement"
    if "balance sheet" in c or "par value" in c:
        return "balance_sheet"
    if "reconciliation of the denominator" in c or "per share" in c:
        return "eps_reconciliation"
    if "cash flow" in c:
        return "cash_flows"
    if "deferred tax" in c or "temporary differences" in c:
        return "deferred_taxes"
    if "equity award" in c or "incentive plan" in c:
        return "equity_awards"
    return "other:" + re.sub(r"[^a-z ]", "", c)[:40].strip()


def parse_rows(block: dict) -> list[tuple[str, list[str]]]:
    out = []
    for line in block["text"].splitlines():
        if not line.startswith("|") or set(line) <= set("| -"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        label, rest = cells[0], cells[1:]
        if not label or NUM_RE.match(label):
            continue
        nums = [c for c in rest if NUM_RE.match(c)]
        if nums:
            out.append((label.rstrip(" .:"), nums))
    return out


def facts_for(fy: int) -> dict:
    path = CLEAN_DIR / f"FY{fy}.jsonl"
    out: dict[tuple[str, str, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        b = json.loads(line)
        if b["block_type"] != "table":
            continue
        years = b["years_present"]
        if len(years) < 2:
            continue
        t = topic(b["table_caption"])
        for label, nums in parse_rows(b):
            for year, val in zip(years, nums[: len(years)]):
                v = to_float(val)
                if v is not None:
                    out.setdefault((t, label, year), f"{v:g}")
    return out


def classify(values: dict[int, str]) -> str:
    nums = [float(v) for v in values.values()]
    if any(n == 0 for n in nums):
        return "RECLASSIFIED"
    hi, lo = max(map(abs, nums)), min(map(abs, nums))
    for p in (10.0, 100.0, 1000.0):
        if abs(hi / lo - p) / p < 0.02:
            return "SPLIT"
    if [n for n in nums if abs(n) <= 100] and [n for n in nums if abs(n) > 1000]:
        return "UNIT_MISMATCH"
    return "RECLASSIFIED"


def main() -> None:
    per_fy = {}
    for fy in (2026, 2025, 2024):
        per_fy[fy] = facts_for(fy)
        print(f"FY{fy}: {len(per_fy[fy]):,} (topic, label, year) figures")

    seen: dict[tuple[str, str, int], dict[int, str]] = defaultdict(dict)
    for fy, facts in per_fy.items():
        for k, v in facts.items():
            seen[k][fy] = v

    buckets: dict[str, list] = defaultdict(list)
    for key, by_src in seen.items():
        if len(by_src) < 2 or len(set(by_src.values())) == 1:
            continue
        buckets[classify(by_src)].append((key, by_src))

    for kind in ("RECLASSIFIED", "SPLIT", "UNIT_MISMATCH"):
        rows = sorted(buckets[kind], key=lambda r: (r[0][0], r[0][1]))
        print(f"\n=== {kind} ({len(rows)}) ===")
        for (t, label, year), by_src in rows:
            detail = "  ".join(
                f"FY{s}: {v}" for s, v in sorted(by_src.items(), reverse=True)
            )
            print(f"  [{t}] {label!r} FY{year}")
            print(f"      {detail}")

    print("\nsummary: " + ", ".join(
        f"{k}={len(buckets[k])}" for k in ("RECLASSIFIED", "SPLIT", "UNIT_MISMATCH")
    ))


if __name__ == "__main__":
    main()
