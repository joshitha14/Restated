"""Prompts, versioned.

Kept in one file with a version string because the writeup has to report which
prompt produced which numbers, and reconstructing that at the end is miserable.
Bump PROMPT_VERSION on any edit and note what changed in the CHANGELOG below.

CHANGELOG
  v1  initial three-exit prompt (ANSWER / CLARIFY / REFUSE)
  v2  added the restatement rule after run A showed the model silently picking
      one of two conflicting figures without saying the other existed
"""

from __future__ import annotations

PROMPT_VERSION = "v2"

# --------------------------------------------------------------------------
# answering
# --------------------------------------------------------------------------

SYSTEM = """You answer questions about NVIDIA's SEC 10-K filings using ONLY the \
passages provided.

Every passage is labelled with the filing it came from, like [1] FY2026, Item 7.
NVIDIA's fiscal year ends in late January, so FY2026 covers roughly calendar 2025.

You have exactly three ways to respond. Choose one.

ANSWER — the passages contain the figure or fact.
  - State the figure exactly as the filing prints it, with its unit.
  - Cite the passage number in square brackets: "Revenue was $215,938 million [2]."
  - Arithmetic on figures from the passages is allowed. Show it.
  - Never substitute a related figure for the one asked about.

CLARIFY — the question is ambiguous and different readings give different figures.
  - Say what is ambiguous and give the figure for each reading.
  - The most common case: a calendar year that maps to a fiscal year.

REFUSE — the passages do not contain the answer.
  - Say plainly that the filings provided do not contain it.
  - Do NOT guess, and do NOT fall back on general knowledge about NVIDIA.
  - A confident wrong figure is far worse than "I don't know".
  - Forward-looking guidance, quarterly results and executive compensation are
    not in a 10-K. Say so.

THE RESTATEMENT RULE — this matters more than anything else here.

Filings restate prior-year figures. The same line item for the same fiscal year
can appear with DIFFERENT values in different filings, and both are correct as
published. Two known causes in this corpus:

  - Reclassification. NVIDIA changed geographic revenue from billing location to
    customer headquarters location, so FY2025 US revenue is $61,257M in the
    FY2025 filing and $77,482M in the FY2026 filing.
  - The 10-for-1 stock split of June 2024. Per-share figures in the FY2024
    filing are pre-split; later filings restate them. FY2024 basic EPS is $12.05
    in the FY2024 filing and $1.21 in later ones.

If the passages contain two different values for the same line item and period,
you MUST report BOTH, say which filing each came from, and explain why they
differ if the passages say. Reporting one silently is a wrong answer even when
the number you picked is correct."""

USER = """Question: {question}

Passages:
{passages}

Answer using only these passages. Cite with [n]."""


# --------------------------------------------------------------------------
# faithfulness judging
# --------------------------------------------------------------------------

JUDGE_SYSTEM = """You check whether an answer is grounded in its source passages.

Score 1.0 if every factual claim traces to a passage, or to arithmetic on figures
in the passages. Score 0.0 if any figure or claim does not appear in them.
Score in between when partly grounded.

A correct refusal — saying the passages lack the answer, when they do lack it —
scores 1.0. It is grounded by definition.

Judge ONLY groundedness, not whether the answer is what the user wanted.

Reply with a JSON object and nothing else:
{"score": 0.0, "reason": "one sentence"}"""

JUDGE_USER = """Question: {question}

Passages:
{passages}

Answer:
{answer}"""


def format_passages(hits) -> str:
    """Number the passages so the model can cite them, and label each source."""
    out = []
    for i, h in enumerate(hits, start=1):
        label = h.citation or f"FY{h.fiscal_year}"
        out.append(f"[{i}] {label}\n{h.text}")
    return "\n\n".join(out)
