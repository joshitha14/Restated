"""Two chunking strategies over the cleaned blocks.

THE EXPERIMENTAL CONTROL (handoff §4): tables are held whole by BOTH strategies.
If only one protects tables, the comparison measures table handling rather than
boundary placement, and the result means nothing. Both strategies therefore
route table blocks through the same code path (`emit_table`) and differ ONLY in
how they group prose.

    fixed    RecursiveCharacterTextSplitter over concatenated prose, 1000 chars
             with 150 overlap. Boundaries fall wherever the character count
             lands, so a sentence or a figure can be split mid-thought.

    semantic Prose is grouped by structure already recovered in clean.py --
             Item section, then paragraph adjacency -- and packed up to a size
             budget without crossing a section boundary. Boundaries fall where
             the document says they do.

A table that exceeds the size budget is still NOT split. That is deliberate: a
half table is worse than a long chunk, because the header row carries the fiscal
years and without it every figure is unattributable.

Output: data/chunks/{strategy}/FY{year}.jsonl, one chunk per line, carrying the
metadata schema from DECISIONS.md D3/D7 plus a stable chunk_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
CHUNK_DIR = ROOT / "data" / "chunks"

FISCAL_YEARS = (2024, 2025, 2026)

# Prose target. 1000/150 is the common LangChain default pairing and is used for
# the fixed arm so the baseline is recognisable rather than tuned.
FIXED_SIZE = 1000
FIXED_OVERLAP = 150

# The semantic arm packs prose up to the same budget, so average chunk size is
# comparable and the comparison is about WHERE boundaries fall, not how much
# text each chunk holds.
SEMANTIC_MAX = 1000
SEMANTIC_MIN = 200  # below this, merge forward rather than emit a stub


@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: str
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
    n_chars: int = 0
    n_source_blocks: int = 1


def load_blocks(fy: int) -> list[dict]:
    path = CLEAN_DIR / f"FY{fy}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run src/clean.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_id(strategy: str, fy: int, seq: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{strategy}-FY{fy}-{seq:04d}-{digest}"


# "fiscal year 2026" / "fiscal 2026" / "fiscal years 2026 and 2025"
FY_MENTION_RE = re.compile(r"fiscal(?:\s+year)?s?\s+((?:\d{4}(?:\s*(?:,|and)\s*)?)+)", re.I)
YEAR_RE = re.compile(r"\b(20[12]\d)\b")


def prose_years(text: str, filing_fy: int) -> list[int]:
    """Fiscal years a prose chunk is about.

    Table chunks get `years_present` from their column headers (D7), but prose
    initially carried an empty list -- so a prose chunk had NO year metadata at
    all, and dense similarity alone cannot tell FY2026 language from FY2024
    language. That is the period-confusion problem the corpus was chosen to
    expose, and it left the metadata-filtered arm with nothing to filter on.

    Two sources, in order:
      1. Years named in the text ("fiscal years 2026 and 2025")
      2. The filing's own fiscal year, always included -- a chunk from the
         FY2026 filing is FY2026 context even when it names no year

    Applied identically to both strategies, so the A/B control is unaffected.
    """
    years = {filing_fy}
    for match in FY_MENTION_RE.finditer(text):
        for y in YEAR_RE.findall(match.group(1)):
            year = int(y)
            # Guard against stray 4-digit numbers; the corpus spans FY2022-FY2026.
            if filing_fy - 4 <= year <= filing_fy:
                years.add(year)
    return sorted(years, reverse=True)


def carry(block: dict) -> dict:
    """Metadata fields that pass through from block to chunk unchanged."""
    return {
        k: block[k]
        for k in (
            "company", "ticker", "cik", "form_type", "fiscal_year", "period_end",
            "item_section", "table_caption", "years_present", "citation",
            "source_file", "source_url",
        )
    }


# --------------------------------------------------------------------------
# shared table path -- identical for both strategies (the control)
# --------------------------------------------------------------------------

def emit_table(block: dict, strategy: str, fy: int, seq: int) -> Chunk:
    """Emit a table block as exactly one chunk, never split, for any strategy."""
    text = block["text"]
    return Chunk(
        chunk_id=make_id(strategy, fy, seq, text),
        text=text,
        strategy=strategy,
        block_type="table",
        n_chars=len(text),
        n_source_blocks=1,
        **carry(block),
    )


# --------------------------------------------------------------------------
# strategy 1: fixed-size
# --------------------------------------------------------------------------

def split_fixed(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """RecursiveCharacterTextSplitter if available, else an equivalent fallback.

    The fallback exists so ingest is runnable before dependencies are installed;
    it uses the same separator ladder and the same size/overlap, so results are
    equivalent for prose without tables.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _split_fixed_fallback(text, size, overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def _split_fixed_fallback(text: str, size: int, overlap: int) -> list[str]:
    seps = ["\n\n", "\n", ". ", " "]

    def rec(s: str) -> list[str]:
        if len(s) <= size:
            return [s] if s.strip() else []
        for sep in seps:
            if sep not in s:
                continue
            parts, buf = [], ""
            for piece in s.split(sep):
                cand = piece if not buf else buf + sep + piece
                if len(cand) <= size:
                    buf = cand
                else:
                    if buf:
                        parts.append(buf)
                    buf = piece
            if buf:
                parts.append(buf)
            if len(parts) > 1:
                return [p for part in parts for p in rec(part)]
        return [s[i : i + size] for i in range(0, len(s), size) if s[i : i + size].strip()]

    chunks = rec(text)
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        out.append((prev[-overlap:] + " " + cur).strip())
    return out


def chunk_fixed(blocks: list[dict], fy: int) -> list[Chunk]:
    """Tables whole; prose concatenated per section and split on character count.

    Prose is concatenated WITHIN an Item section rather than across the whole
    filing, so a fixed-size boundary cannot merge Item 1A risk language with
    Item 7 MD&A. Without this the fixed arm would lose its section metadata
    entirely and the comparison would confound chunking with metadata quality.
    """
    out: list[Chunk] = []
    seq = 0

    for block in blocks:
        if block["block_type"] == "table":
            out.append(emit_table(block, "fixed", fy, seq))
            seq += 1

    by_section: dict[str, list[dict]] = {}
    for block in blocks:
        if block["block_type"] == "prose":
            by_section.setdefault(block["item_section"], []).append(block)

    for section, group in by_section.items():
        joined = "\n\n".join(b["text"] for b in group)
        for piece in split_fixed(joined):
            if not piece.strip():
                continue
            meta = carry(group[0])
            meta["table_caption"] = ""
            meta["years_present"] = prose_years(piece, fy)
            out.append(
                Chunk(
                    chunk_id=make_id("fixed", fy, seq, piece),
                    text=piece,
                    strategy="fixed",
                    block_type="prose",
                    n_chars=len(piece),
                    n_source_blocks=len(group),
                    **meta,
                )
            )
            seq += 1

    return out


# --------------------------------------------------------------------------
# strategy 2: semantic
# --------------------------------------------------------------------------

HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,&'()/-]{3,80}$")

# Sentence end: terminal punctuation, then whitespace, then a capital or digit.
# Avoids splitting on "U.S.", "Inc.", "$1.5 billion" and similar.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_long_paragraph(text: str, budget: int = SEMANTIC_MAX) -> list[str]:
    """Split an oversized paragraph on sentence boundaries only.

    A few risk-factor paragraphs in Item 1A run past 3,000 characters. Emitting
    them whole would make some semantic chunks 3x the fixed arm's, which would
    confound the comparison with chunk size. Splitting them mid-word would
    abandon this arm's whole premise.

    Sentence boundaries are the compromise: the unit stays readable and a figure
    keeps the sentence that qualifies it.
    """
    if len(text) <= budget:
        return [text]

    out, buf = [], ""
    for sentence in SENTENCE_RE.split(text):
        cand = sentence if not buf else f"{buf} {sentence}"
        if len(cand) <= budget or not buf:
            buf = cand
        else:
            out.append(buf)
            buf = sentence
    if buf:
        out.append(buf)
    return out


def looks_like_heading(text: str) -> bool:
    """A short, title-cased line with no terminal period starts a new topic."""
    t = text.strip()
    if len(t) > 90 or t.endswith((".", ";", ":")):
        return False
    words = [w for w in t.split() if w[:1].isalpha()]
    if len(words) < 2:
        return False
    return bool(HEADING_RE.match(t)) and sum(w[:1].isupper() for w in words) >= len(words) // 2


def chunk_semantic(blocks: list[dict], fy: int) -> list[Chunk]:
    """Tables whole; prose packed along structural boundaries.

    Boundaries respected, in order of precedence:
      1. Item section     -- never pack across it
      2. Heading-like block -- starts a new chunk
      3. Size budget      -- pack paragraphs until SEMANTIC_MAX

    Paragraph integrity is preserved: a paragraph is never cut mid-sentence, so
    a figure and the sentence qualifying it stay together. That is the mechanism
    this arm is meant to test.
    """
    out: list[Chunk] = []
    seq = 0

    for block in blocks:
        if block["block_type"] == "table":
            out.append(emit_table(block, "semantic", fy, seq))
            seq += 1

    prose = [b for b in blocks if b["block_type"] == "prose"]

    buf: list[dict] = []

    def flush() -> None:
        nonlocal seq, buf
        if not buf:
            return
        text = "\n\n".join(b["text"] for b in buf)
        meta = carry(buf[0])
        meta["table_caption"] = ""
        meta.pop("years_present", None)  # set per piece below
        # A single paragraph can exceed the budget on its own; split it on
        # sentence boundaries rather than emitting a 3x-oversized chunk.
        for piece in split_long_paragraph(text):
            out.append(
                Chunk(
                    chunk_id=make_id("semantic", fy, seq, piece),
                    text=piece,
                    strategy="semantic",
                    block_type="prose",
                    n_chars=len(piece),
                    n_source_blocks=len(buf),
                    # Per piece, not per buffer: a split paragraph may name
                    # different years in each half.
                    years_present=prose_years(piece, fy),
                    **meta,
                )
            )
            seq += 1
        buf = []

    for block in prose:
        crosses_section = bool(buf) and block["item_section"] != buf[-1]["item_section"]
        starts_topic = bool(buf) and looks_like_heading(block["text"])
        would_exceed = (
            sum(len(b["text"]) + 2 for b in buf) + len(block["text"]) > SEMANTIC_MAX
        )

        if crosses_section or starts_topic:
            flush()
        elif would_exceed and sum(len(b["text"]) for b in buf) >= SEMANTIC_MIN:
            flush()

        buf.append(block)

    flush()
    return out


# --------------------------------------------------------------------------
# strategy 3: SemanticChunker (embedding-based topic detection)
# --------------------------------------------------------------------------

def section_documents(blocks: list[dict]) -> list:
    """Join prose blocks back into one document per Item section.

    The cleaner deliberately emits small blocks so tables can be isolated and
    every block tagged with its section. But an embedding-based chunker needs
    CONTINUOUS text: it looks for the point where a topic shifts, and a lone
    paragraph has no such point. Handing it ~600 paragraphs per filing would
    mean ~600 round trips to the embedding model, each asking a question with
    no meaningful answer.

    Merging also keeps the comparison honest -- every strategy receives the
    same input, so the only thing that differs is where boundaries are placed.
    """
    from langchain_core.documents import Document

    by_section: dict[str, list[dict]] = {}
    for b in blocks:
        if b["block_type"] == "prose":
            by_section.setdefault(b["item_section"], []).append(b)

    return [
        Document(
            page_content="\n\n".join(b["text"] for b in group),
            metadata=dict(group[0]),
        )
        for group in by_section.values()
    ]


def chunk_semantic_embed(blocks: list[dict], fy: int, embeddings=None) -> list[Chunk]:
    """Tables whole; prose split where consecutive-sentence meaning shifts.

    Uses LangChain's SemanticChunker at the 90th percentile, which is the
    standard reading of "semantic chunking" and what the course material
    recommends. Contrast with chunk_semantic(), which splits on document
    STRUCTURE (section -> heading -> budget) rather than embedding distance.

    Requires an embeddings object; this arm costs API calls, unlike the others.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    if embeddings is None:
        raise ValueError("chunk_semantic_embed requires an embeddings object")

    out: list[Chunk] = []
    seq = 0

    for block in blocks:
        if block["block_type"] == "table":
            out.append(emit_table(block, "semantic_embed", fy, seq))
            seq += 1

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    for doc in splitter.split_documents(section_documents(blocks)):
        text = doc.page_content
        if not text.strip():
            continue
        meta = carry(doc.metadata)
        meta["table_caption"] = ""
        meta["years_present"] = prose_years(text, fy)
        out.append(
            Chunk(
                chunk_id=make_id("semantic_embed", fy, seq, text),
                text=text,
                strategy="semantic_embed",
                block_type="prose",
                n_chars=len(text),
                n_source_blocks=1,
                **meta,
            )
        )
        seq += 1

    return out


# --------------------------------------------------------------------------

STRATEGIES = {
    "fixed": chunk_fixed,
    "semantic": chunk_semantic,              # structural
    "semantic_embed": chunk_semantic_embed,  # SemanticChunker (needs embeddings)
}


def write_chunks(strategy: str, fy: int, chunks: list[Chunk]) -> Path:
    dest = CHUNK_DIR / strategy / f"FY{fy}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return dest


def summarise(strategy: str, fy: int, chunks: list[Chunk]) -> None:
    tab = [c for c in chunks if c.block_type == "table"]
    pro = [c for c in chunks if c.block_type == "prose"]
    sizes = sorted(c.n_chars for c in pro)
    med = sizes[len(sizes) // 2] if sizes else 0
    print(
        f"  {strategy:9s} FY{fy}  {len(chunks):4d} chunks"
        f"  ({len(tab)} table, {len(pro)} prose)"
        f"  prose chars: median {med}, max {max(sizes) if sizes else 0}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strategy",
        choices=[*STRATEGIES, "both", "all"],
        default="both",
        help="'both' = fixed + structural (free); 'all' adds semantic_embed (costs API calls)",
    )
    args = ap.parse_args()

    if args.strategy == "both":
        names = ["fixed", "semantic"]
    elif args.strategy == "all":
        names = list(STRATEGIES)
    else:
        names = [args.strategy]

    # Only the SemanticChunker arm needs an embeddings client.
    embeddings = None
    if "semantic_embed" in names:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from index import BASE_URL, EMBED_MODEL, api_key
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=BASE_URL,
            api_key=api_key(),
            check_embedding_ctx_length=False,
        )

    for strategy in names:
        print(f"\n=== {strategy} ===")
        for fy in FISCAL_YEARS:
            blocks = load_blocks(fy)
            if strategy == "semantic_embed":
                chunks = chunk_semantic_embed(blocks, fy, embeddings)
            else:
                chunks = STRATEGIES[strategy](blocks, fy)
            write_chunks(strategy, fy, chunks)
            summarise(strategy, fy, chunks)

    print(f"\n-> {CHUNK_DIR.relative_to(ROOT)}/{{{','.join(names)}}}/FY*.jsonl")


if __name__ == "__main__":
    main()
