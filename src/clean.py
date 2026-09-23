"""EDGAR HTML -> section-tagged, table-aware blocks.

The only genuinely custom code in the pipeline (handoff §5). Everything
downstream is stock LangChain.

Three problems in EDGAR markup, all verified against the real filings rather
than assumed:

1. `<table>` is used for both financial data and page layout. A generic loader
   flattens both, so a data row arrives with no indication which figure belongs
   to which year. Layout tables are detected and dropped.

2. Currency symbols occupy their own cells. A row arrives as
   ['Data Center', '$', '193,737', '$', '115,186', '$', '47,525'] against a
   3-column header. Symbol-only cells are folded away before alignment.

3. Item section headings need tagging so answers can cite a section.

Also handled, found during inspection:

4. Empty <tr> rows (table 55 of the FY2026 filing has 13 rows, of which several
   have zero cells). Dropped.

5. Header and data rows have different cell counts (2 then 4). Alignment is on
   the widest row, not the first.

FALLBACK: if table extraction yields implausible results, `--dumb` emits tables
as raw text with structure preserved as best-effort. A degraded cleaner with a
completed comparison beats a perfect cleaner and no results.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "clean"

DATE_RE = re.compile(r"(?:Jan|January|Feb|February|Mar|March|Dec|December)\w*\s+(\d{1,2}),?\s+(\d{4})", re.I)
NUM_RE = re.compile(r"^\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?%?$")
SYMBOL_ONLY = {"", "$", "%", "(", ")", "—", "-", "–", "*"}

# "Item 7." / "Item 1A." at the start of a heading.
ITEM_RE = re.compile(r"^\s*Item\s+(\d{1,2}[A-C]?)\s*[.:—-]?\s*(.{0,120})", re.I)


# --------------------------------------------------------------------------
# metadata from inline XBRL (DECISIONS.md D2 -- never from prose or filing date)
# --------------------------------------------------------------------------

def dei_fact(html: str, name: str) -> str | None:
    m = re.search(
        rf'(?is)<ix:nonNumeric[^>]*name="{re.escape(name)}"[^>]*>(.*?)</ix:nonNumeric>', html
    )
    if not m:
        m = re.search(rf'(?is)name="{re.escape(name)}"[^>]*>(.*?)<', html)
    if not m:
        return None
    return clean_text(m.group(1))


@dataclass
class FilingMeta:
    company: str
    cik: str
    ticker: str
    form_type: str
    fiscal_year: int
    period_end: str
    source_file: str
    source_url: str

    @property
    def accession_hint(self) -> str:
        return self.source_file


def read_meta(path: Path) -> FilingMeta:
    html = path.read_text(encoding="utf-8", errors="ignore")
    fy = dei_fact(html, "dei:DocumentFiscalYearFocus")
    cik = dei_fact(html, "dei:EntityCentralIndexKey") or ""
    if not fy:
        raise ValueError(f"{path.name}: no dei:DocumentFiscalYearFocus -- cannot tag fiscal_year")

    # Period end from the filename stem (nvda-20260125) -- the DEI fact is split
    # across nodes for the day/year and is unreliable to reassemble.
    stem = re.search(r"(\d{8})", path.stem)
    period_end = (
        f"{stem.group(1)[:4]}-{stem.group(1)[4:6]}-{stem.group(1)[6:]}" if stem else ""
    )

    return FilingMeta(
        company=dei_fact(html, "dei:EntityRegistrantName") or "NVIDIA CORP",
        cik=cik,
        ticker=dei_fact(html, "dei:TradingSymbol") or "NVDA",
        form_type=dei_fact(html, "dei:DocumentType") or "10-K",
        fiscal_year=int(fy),
        period_end=period_end,
        source_file=path.name,
        source_url=(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type=10-K"
        ),
    )


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------

def clean_text(html: str) -> str:
    txt = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html)
    txt = re.sub(r"<[^>]+>", " ", txt)
    for ent, rep in (
        ("&#160;", " "), ("&nbsp;", " "), ("&amp;", "&"), ("&#8217;", "'"),
        ("&#8220;", '"'), ("&#8221;", '"'), ("&#8211;", "-"), ("&#8212;", "--"),
        ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'"), ("&quot;", '"'),
    ):
        txt = txt.replace(ent, rep)
    txt = re.sub(r"&#\d+;", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def is_number(s: str) -> bool:
    return bool(s) and bool(NUM_RE.match(s.strip()))


def strip_xbrl_header(html: str) -> str:
    """Remove the hidden inline-XBRL header block.

    Every inline-XBRL filing carries an <ix:header> holding the machine-readable
    fact context: namespace URIs, ISO duration codes, unit identifiers. It is
    invisible to a human reader but is ~19,000 characters of text, and a naive
    extractor emits it as one enormous "paragraph":

        0001045810 2026 FY false 362 460 P1Y P2Y P3Y
        http://fasb.org/us-gaap/2025#AccruedLiabilitiesCurrent iso4217:USD

    Left in, it becomes the largest chunk in the corpus and can be retrieved for
    any query. The DEI facts are read from this region deliberately by
    `dei_fact()` BEFORE this strip is applied, so removing it loses nothing.
    """
    return re.sub(r"(?is)<ix:header\b.*?</ix:header>", " ", html)


# Markers of machine metadata rather than narrative text.
NOISE_RE = re.compile(
    r"(?:iso4217:|xbrli:|utr:|http://fasb\.org|http://www\.nvidia\.com/\d{8}|"
    r"http://xbrl\.sec\.gov|us-gaap:|srt:|\bP\d+[YMD]\b)"
)


def is_machine_noise(text: str) -> bool:
    """True if the text is XBRL plumbing rather than prose a human would read."""
    if NOISE_RE.search(text):
        return True
    tokens = text.split()
    if len(tokens) < 8:
        return False
    # Narrative prose is mostly words; fact contexts are mostly numbers/codes.
    wordish = sum(1 for t in tokens if t[:1].isalpha() and len(t) > 2)
    return wordish / len(tokens) < 0.45


# --------------------------------------------------------------------------
# table extraction
# --------------------------------------------------------------------------

@dataclass
class Table:
    caption: str
    header_years: list[int]
    rows: list[list[str]]
    markdown: str
    n_numeric_cells: int


def row_cells(row_html: str) -> list[str]:
    return [clean_text(c) for c in re.findall(r"(?is)<t[dh]\b.*?</t[dh]>", row_html)]


def fold_symbols(cells: list[str]) -> list[str]:
    """Drop symbol-only cells ($ in its own cell) so columns align (problem 2)."""
    return [c for c in cells if c.strip() not in SYMBOL_ONLY]


def is_layout_table(rows: list[list[str]]) -> bool:
    """A layout table has no numeric content worth keeping (problem 1)."""
    if not rows:
        return True
    numeric = sum(1 for r in rows for c in r if is_number(c))
    cells = sum(len(r) for r in rows) or 1
    # Page furniture: a couple of cells, or almost no numbers.
    if cells <= 3:
        return True
    return numeric / cells < 0.10


def header_years(rows: list[list[str]]) -> list[int]:
    for r in rows[:4]:
        found = DATE_RE.findall(" ".join(r))
        if len(found) >= 2:
            return [int(y) for _, y in found]
    for r in rows[:4]:
        found = DATE_RE.findall(" ".join(r))
        if found:
            return [int(y) for _, y in found]
    return []


def to_markdown(rows: list[list[str]]) -> str:
    """Render as a pipe table, padded to the widest row (problem 5)."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    head, *body = padded
    out = ["| " + " | ".join(head) + " |", "|" + "|".join([" --- "] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def caption_before(html: str, pos: int) -> str:
    """Nearest descriptive text before a table, used to disambiguate line labels.

    Labels alone are not unique -- 'Other', 'Total' and 'Balance at end of
    period' appear in many unrelated tables (DECISIONS.md D3).

    Preference order matters. A lead-in sentence ("The following table
    summarizes...") is more informative than a bare heading, but some tables
    have only a heading ("Operating Income by Reportable Segments") and others
    have neither. Sentences are therefore preferred, with headings as fallback.
    """
    window = html[max(0, pos - 3000) : pos]
    cands = [
        t
        for t in (
            clean_text(m.group(1))
            for m in re.finditer(r"(?is)<div\b[^>]*>((?:(?!<div\b).)*?)</div>", window)
        )
        if 8 <= len(t) <= 240 and not t.startswith("|") and not is_number(t)
    ]
    if not cands:
        return ""

    # A lead-in sentence, nearest-last.
    for t in reversed(cands):
        if t.rstrip().endswith(":") or re.search(r"(?i)following table|are as follows|consist", t):
            return t
    # Otherwise a title-ish heading: mostly capitalised words, no terminal period.
    for t in reversed(cands):
        words = [w for w in t.split() if w[:1].isalpha()]
        if words and not t.endswith(".") and sum(w[:1].isupper() for w in words) >= max(2, len(words) // 2):
            return t
    return cands[-1]


def extract_tables(html: str, dumb: bool = False) -> list[tuple[int, Table]]:
    out: list[tuple[int, Table]] = []
    for m in re.finditer(r"(?is)<table\b.*?</table>", html):
        raw_rows = re.findall(r"(?is)<tr\b.*?</tr>", m.group(0))
        rows: list[list[str]] = []
        for r in raw_rows:
            cs = fold_symbols(row_cells(r))
            if cs:  # drop empty <tr> (problem 4)
                rows.append(cs)

        if not dumb and is_layout_table(rows):
            continue
        if not rows:
            continue

        out.append(
            (
                m.start(),
                Table(
                    caption=caption_before(html, m.start()),
                    header_years=header_years(rows),
                    rows=rows,
                    markdown=to_markdown(rows),
                    n_numeric_cells=sum(1 for r in rows for c in r if is_number(c)),
                ),
            )
        )
    return out


# --------------------------------------------------------------------------
# item sections
# --------------------------------------------------------------------------

def find_item_sections(html: str) -> list[tuple[int, str]]:
    """Character offset -> 'Item 7' style label, from bold/heading-ish nodes."""
    hits: list[tuple[int, str]] = []
    # Leaf <div>s only -- these filings have no <p>/<b>/<h*> (see build_blocks).
    for m in re.finditer(r"(?is)<div\b[^>]*>((?:(?!<div\b).){0,400}?)</div>", html):
        txt = clean_text(m.group(1))
        im = ITEM_RE.match(txt)
        if not im:
            continue
        # Require a title after the number, to skip table-of-contents page refs.
        title = im.group(2).strip(" .:-")
        if len(title) < 4:
            continue
        hits.append((m.start(), f"Item {im.group(1).upper()}"))

    # Keep the LAST occurrence of each item -- the first is the TOC entry.
    seen: dict[str, int] = {}
    for pos, label in hits:
        seen[label] = max(seen.get(label, 0), pos)
    return sorted((pos, label) for label, pos in seen.items())


def item_at(sections: list[tuple[int, str]], pos: int) -> str:
    label = "front matter"
    for spos, slabel in sections:
        if spos <= pos:
            label = slabel
        else:
            break
    return label


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

@dataclass
class Block:
    text: str
    block_type: str  # prose | table
    company: str
    ticker: str
    cik: str
    form_type: str
    fiscal_year: int
    period_end: str
    item_section: str
    table_caption: str = ""
    years_present: list[int] = field(default_factory=list)
    citation: str = ""
    source_file: str = ""
    source_url: str = ""


def make_citation(
    fiscal_year: int,
    item_section: str,
    caption: str = "",
    years: list[int] | None = None,
) -> str:
    """Human-readable citation, degrading gracefully when the caption is missing.

    Caption extraction is not reliable for every table -- the FY2026 geographic
    revenue table has no preceding descriptive text at all. A citation must
    never come out blank, because a blank citation cannot be scored for
    groundedness and the restatement questions depend entirely on knowing WHICH
    filing a figure came from.

    Falls back: caption -> years covered -> section alone.

        FY2026, Item 7, "Revenue by geographic area..."
        FY2026, Item 8 (covering FY2026-FY2024)
        FY2026, Item 7
    """
    head = f"FY{fiscal_year}, {item_section}"
    if caption:
        short = caption if len(caption) <= 80 else caption[:77].rstrip() + "..."
        return f'{head}, "{short}"'
    if years and len(years) > 1:
        return f"{head} (covering FY{max(years)}-FY{min(years)})"
    if years:
        return f"{head} (FY{years[0]})"
    return head


def build_blocks(path: Path, dumb: bool = False) -> list[Block]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    # read_meta() parses the DEI facts out of <ix:header>, so it must run BEFORE
    # the header is stripped.
    meta = read_meta(path)
    html = strip_xbrl_header(html)
    sections = find_item_sections(html)
    tables = extract_tables(html, dumb=dumb)

    blocks: list[Block] = []

    def base(pos: int) -> dict:
        return dict(
            company=meta.company,
            ticker=meta.ticker,
            cik=meta.cik,
            form_type=meta.form_type,
            fiscal_year=meta.fiscal_year,
            period_end=meta.period_end,
            item_section=item_at(sections, pos),
            source_file=meta.source_file,
            source_url=meta.source_url,
        )

    # Table blocks. years_present, not a scalar fiscal_year, per D7.
    for pos, t in tables:
        caption = t.caption
        years = t.header_years or [meta.fiscal_year]
        common = base(pos)
        header = f"{caption}\n" if caption else ""
        blocks.append(
            Block(
                text=f"{header}{t.markdown}",
                block_type="table",
                table_caption=caption,
                years_present=years,
                citation=make_citation(
                    meta.fiscal_year, common["item_section"], caption, years
                ),
                **common,
            )
        )

    # Prose blocks: everything outside tables.
    #
    # NVIDIA's filings contain ZERO <p> tags (verified: 0 <p>, ~1,900 <div>,
    # ~4,200 <span>). Body text is <div> wrapping <span>, with headings marked
    # only by font-weight:700 in inline CSS. So paragraphs are leaf <div>s --
    # ones containing no nested <div>.
    spans = [(m.start(), m.end()) for m in re.finditer(r"(?is)<table\b.*?</table>", html)]
    cursor = 0
    for start, end in spans + [(len(html), len(html))]:
        segment = html[cursor:start]
        offset = cursor
        cursor = end
        for pm in re.finditer(r"(?is)<div\b[^>]*>((?:(?!<div\b).)*?)</div>", segment):
            txt = clean_text(pm.group(1))
            if len(txt) < 80 or is_machine_noise(txt):
                continue
            common = base(offset + pm.start())
            blocks.append(
                Block(
                    text=txt,
                    block_type="prose",
                    citation=make_citation(meta.fiscal_year, common["item_section"]),
                    **common,
                )
            )

    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumb", action="store_true", help="fallback: keep all tables as raw text")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # EDGAR serves .htm; a browser-saved copy is usually .html. Accept both, and
    # de-duplicate by period-end date so a directory holding both copies of the
    # same filing is not cleaned twice.
    by_period: dict[str, Path] = {}
    for path in sorted(RAW_DIR.glob("nvda-*.htm")) + sorted(RAW_DIR.glob("nvda-*.html")):
        stem = re.search(r"(\d{8})", path.stem)
        key = stem.group(1) if stem else path.stem
        by_period.setdefault(key, path)
    files = [by_period[k] for k in sorted(by_period)]
    if not files:
        raise SystemExit(
            f"no filings in {RAW_DIR}\n"
            "run:  SEC_USER_AGENT='Your Name you@example.com' python src/fetch.py"
        )

    for path in files:
        blocks = build_blocks(path, dumb=args.dumb)
        meta = read_meta(path)
        dest = OUT_DIR / f"FY{meta.fiscal_year}.jsonl"
        with dest.open("w", encoding="utf-8") as fh:
            for b in blocks:
                fh.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")

        n_tab = sum(1 for b in blocks if b.block_type == "table")
        n_pro = len(blocks) - n_tab
        secs = sorted({b.item_section for b in blocks})
        print(
            f"FY{meta.fiscal_year}  {path.name}\n"
            f"    {len(blocks):5,} blocks  ({n_tab} table, {n_pro:,} prose)\n"
            f"    sections: {len(secs)}  -> {', '.join(secs[:12])}"
            + ("..." if len(secs) > 12 else "")
            + f"\n    -> {dest.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
