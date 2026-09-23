"""Is a wrong answer retrieval's fault or generation's?

Runs the same question twice: once on what the retriever actually returned, and
once on passages guaranteed to contain the answer. If the first refuses and the
second answers, the generator is fine and retrieval is the problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import answer, build_llm  # noqa: E402
from retrieve import Hit, build  # noqa: E402

QUESTION = "What was NVIDIA's Data Center revenue in fiscal year 2026?"
FIGURE = "193,737"


def oracle_hits() -> list[Hit]:
    """The chunks that actually contain the answer, whatever their rank."""
    out = []
    for fy in (2026, 2025, 2024):
        path = ROOT / "data" / "chunks" / "fixed" / f"FY{fy}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            if FIGURE in c["text"]:
                out.append(
                    Hit(
                        chunk_id=c["chunk_id"], text=c["text"], score=1.0,
                        rank=len(out), citation=c["citation"],
                        fiscal_year=c["fiscal_year"], item_section=c["item_section"],
                        block_type=c["block_type"], years_present=c["years_present"],
                        table_caption=c.get("table_caption", ""),
                    )
                )
    return out


def main() -> None:
    llm = build_llm()

    for k in (10, 20):
        hits = build("hybrid", "fixed").search(QUESTION, k=k)
        has = any(FIGURE in h.text for h in hits)
        a = answer(QUESTION, hits, llm)
        print(f"\n--- hybrid k={k}: figure in passages = {has} ---")
        print(f"exit={a.exit}  citations={a.citations}")
        print(a.text[:260])

    oracle = oracle_hits()
    print(f"\n--- oracle passages ({len(oracle)} chunks containing {FIGURE}) ---")
    for h in oracle:
        print(f"    {h.citation[:70]}")
    a = answer(QUESTION, oracle, llm)
    print(f"exit={a.exit}  citations={a.citations}")
    print(a.text[:400])


if __name__ == "__main__":
    main()
