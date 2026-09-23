"""Check classify_exit against the real answers it previously got wrong.

Cases are taken verbatim from reports/graph_scorecard.json rather than invented,
so this tests the actual failure rather than a guess at it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import classify_exit  # noqa: E402

# (name, opening of the real answer, expected exit)
CASES = [
    (
        "clarify-with-refusal-aside",
        "CLARIFY — the question is ambiguous because \"2025\" could refer to calendar "
        "year 2025 or NVIDIA's fiscal year 2025 (which ended January 26, 2025).\n\n"
        "- If the question refers to **NVIDIA's fiscal year 2025**, the total revenue "
        "was **$130,497 million** [1].\n\n- If the question refers to **calendar year "
        "2025**, the filings provided do not contain a breakdown by calendar year, so "
        "the answer cannot be determined from these passages.",
        "CLARIFY",
    ),
    (
        # Undeclared, states both restated figures with citations. ANSWER is
        # right: eval/questions.yaml expects ANSWER for q_restate_03, and the
        # restatement rule asks for both figures WITHIN an answer rather than a
        # separate CLARIFY exit. Reporting both is the requirement; CLARIFY is
        # reserved for a question that is genuinely ambiguous as asked.
        "answer-with-however",
        "The total operating income by reportable segment for fiscal 2025 is reported "
        "differently in two filings due to restatements.\n\nIn the FY2026 filing, the "
        "total is $87,960 million [1]. However, in the FY2025 filing, the same period "
        "is reported as $81,453 million [2].",
        "ANSWER",
    ),
    (
        "mislabelled-refuse-that-answers",
        "REFUSE — the passages do not contain the answer.\n\nExplanation: Two different "
        "values appear in the provided passages:\n\n- In the FY2024 filing ([6]), the "
        "diluted weighted average shares are reported as **2,494 million**.\n- In the "
        "FY2025 and FY2026 filings ([4] and [2]), the same period shows **24,940 "
        "million**.",
        "ANSWER",
    ),
    (
        "genuine-refusal",
        "REFUSE — the passages provided do not contain NVIDIA's projected Data Center "
        "revenue for fiscal year 2027.\n\nThe passages include historical data and "
        "commentary, but no forward-looking projection is provided.",
        "REFUSE",
    ),
    (
        "plain-answer",
        "Data Center revenue for fiscal year 2026 was $193,737 million [3].",
        "ANSWER",
    ),
    (
        "declared-answer",
        "ANSWER — Total revenue for fiscal 2026 was $215,938 million [2].",
        "ANSWER",
    ),
]


def main() -> int:
    failures = 0
    for name, text, want in CASES:
        got = classify_exit(text)
        ok = got == want
        if not ok:
            failures += 1
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name:34s} got={got:8s} want={want}")

    print()
    if failures:
        print(f"{failures} case(s) failed")
        return 1
    print(f"all {len(CASES)} cases classified correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
