"""Fetch NVDA 10-K filings from SEC EDGAR.

Resolves ticker -> CIK at runtime rather than hardcoding. A wrong CIK fails in
the worst way: it silently returns a different company's filings rather than
erroring.

SEC requires a real User-Agent (name + email) or requests are blocked. This is
the most common reason a fetch script silently fails, so it is asserted here
rather than left to fail at request time.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_DIR = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# SEC fair-access guidance is 10 req/s; we stay well under.
REQUEST_DELAY = 0.5


def user_agent() -> str:
    """Return the SEC-required User-Agent, or exit with an actionable message."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        sys.exit(
            "SEC_USER_AGENT must be set to a real name and email, e.g.\n"
            '  setx SEC_USER_AGENT "Jane Doe jane@example.com"\n'
            "SEC blocks requests without one."
        )
    return ua


def _get(url: str, ua: str) -> requests.Response:
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers={"User-Agent": ua}, timeout=30)
    resp.raise_for_status()
    return resp


def resolve_cik(ticker: str, ua: str) -> int:
    """Map a ticker symbol to its CIK via SEC's official mapping file."""
    data = _get(TICKER_URL, ua).json()
    ticker = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return int(row["cik_str"])
    raise LookupError(f"ticker {ticker!r} not found in SEC mapping")


@dataclass(frozen=True)
class Filing:
    """One filing's identifying metadata, as reported by EDGAR."""

    accession: str  # with dashes, e.g. 0001045810-26-000021
    form_type: str
    filing_date: str  # ISO, when it was filed
    report_date: str  # ISO, period of report -- the fiscal-year anchor
    primary_doc: str

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    def url(self, cik: int) -> str:
        return (
            ARCHIVE_DIR.format(cik=cik, accession=self.accession_nodash)
            + self.primary_doc
        )

    @property
    def local_name(self) -> str:
        return self.primary_doc


def list_filings(cik: int, ua: str, form: str = "10-K", limit: int = 3) -> list[Filing]:
    """Return the most recent `limit` filings of `form` type, newest first."""
    recent = _get(SUBMISSIONS_URL.format(cik=cik), ua).json()["filings"]["recent"]
    out: list[Filing] = []
    for i, ft in enumerate(recent["form"]):
        # Exclude amendments (10-K/A) -- they restate and would double-count a year.
        if ft != form:
            continue
        out.append(
            Filing(
                accession=recent["accessionNumber"][i],
                form_type=ft,
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_doc=recent["primaryDocument"][i],
            )
        )
        if len(out) == limit:
            break
    return out


def download(ticker: str = "NVDA", limit: int = 3) -> list[Filing]:
    """Download the N most recent 10-Ks into data/raw/, skipping existing files."""
    ua = user_agent()
    cik = resolve_cik(ticker, ua)
    print(f"{ticker} -> CIK {cik:010d}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    filings = list_filings(cik, ua, limit=limit)

    for f in filings:
        dest = RAW_DIR / f.local_name
        print(f"  {f.report_date}  filed {f.filing_date}  {f.accession}  {f.primary_doc}")
        if dest.exists():
            print(f"    exists, skipping ({dest.stat().st_size:,} bytes)")
            continue
        body = _get(f.url(cik), ua).content
        dest.write_bytes(body)
        print(f"    downloaded {len(body):,} bytes")

    return filings


if __name__ == "__main__":
    download()
