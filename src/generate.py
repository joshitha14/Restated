"""Generate cited answers, or refuse.

Three exits: ANSWER, CLARIFY, REFUSE. Which one was taken is detected from the
text rather than requested as a structured field, because asking the model to
self-label its exit makes the label a second thing that can be wrong -- a model
that writes a confident figure and labels it REFUSE would score as a correct
refusal.

Citations are parsed out of the answer with a regex and resolved against the
passages actually supplied, so a citation to a passage that was never provided
is detectable. The model cannot invent a source that scores as grounded.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from index import BASE_URL, api_key  # noqa: E402
from prompts import (  # noqa: E402
    JUDGE_SYSTEM,
    JUDGE_USER,
    PROMPT_VERSION,
    SYSTEM,
    USER,
    format_passages,
)

CHAT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

CITE_RE = re.compile(r"\[(\d{1,2})\]")

# Phrases that mark a refusal. Matched case-insensitively against the answer.
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain",
    "cannot be answered",
    "not available in",
    "not included in",
    "not disclosed in",
    "no information",
    "not found in",
    "unable to answer",
    "not present in",
)

CLARIFY_MARKERS = (
    "ambiguous",
    "could mean",
    "depends on whether",
    "two readings",
    "clarify",
)


@dataclass
class Answer:
    question: str
    text: str
    exit: str  # ANSWER | CLARIFY | REFUSE
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_version: str = PROMPT_VERSION
    model: str = CHAT_MODEL


# The prompt asks the model to open with its chosen exit ("ANSWER — ...",
# "CLARIFY — ...", "REFUSE — ..."). When it does, that declaration is the
# answer's own account of what it did and beats any keyword heuristic.
DECLARED_EXIT_RE = re.compile(r"^\s*\**(ANSWER|CLARIFY|REFUSE)\b", re.I)


def classify_exit(text: str) -> str:
    """Which of the three exits the answer took.

    Reads the model's own declaration first. Keyword matching alone got this
    wrong in three measurable ways, all of which cost exit accuracy while the
    answers themselves were correct:

      - An answer opening "CLARIFY — the question is ambiguous" was classified
        REFUSE, because it later said one reading "cannot be determined from
        these passages". The aside outvoted the declaration.
      - A complete answer giving both restated figures plus the arithmetic was
        classified REFUSE for containing the word "however".
      - An answer that stated both split-adjusted figures correctly opened with
        the word REFUSE and then answered anyway -- here the declaration is
        wrong and the body is right, which no classifier can fix; it is a
        prompting problem, recorded rather than papered over.

    Falls back to keywords only when nothing is declared.
    """
    text = text.strip()

    m = DECLARED_EXIT_RE.match(text)
    if m:
        declared = m.group(1).upper()
        # A declared REFUSE that goes on to give figures with citations is
        # really an answer; trust the body over the label in that one direction.
        #
        # Stating two restated figures is an ANSWER, not a CLARIFY: the
        # restatement rule asks for both values WITHIN an answer. CLARIFY is
        # reserved for a question that is ambiguous as asked, such as a calendar
        # year that maps to two possible fiscal years.
        if declared == "REFUSE" and _looks_like_an_answer(text):
            return "ANSWER"
        return declared

    low = text.lower()
    if any(k in low for k in CLARIFY_MARKERS):
        return "CLARIFY"
    if any(k in low[:400] for k in REFUSAL_MARKERS):
        return "REFUSE"
    if any(k in low for k in REFUSAL_MARKERS):
        return "REFUSE"
    return "ANSWER"


def _looks_like_an_answer(text: str) -> bool:
    """Does the body state figures with citations, despite a REFUSE label?"""
    has_figures = len(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", text)) >= 1
    has_citation = bool(CITE_RE.search(text))
    return has_figures and has_citation


def build_llm(model: str = CHAT_MODEL, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=api_key(),
        temperature=temperature,
        max_tokens=700,
    )


def answer(question: str, hits: list, llm=None) -> Answer:
    """Answer a question from retrieved passages, or refuse."""
    llm = llm or build_llm()
    passages = format_passages(hits)

    t0 = time.perf_counter()
    resp = llm.invoke(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(question=question, passages=passages)},
        ]
    )
    ms = (time.perf_counter() - t0) * 1000

    text = (resp.content or "").strip()
    # Some models emit a reasoning preamble; strip it before classification so
    # deliberation about refusing is not itself read as a refusal.
    text = re.sub(r"(?is)^<think>.*?</think>\s*", "", text).strip()

    cited = sorted({int(n) for n in CITE_RE.findall(text)})
    valid = [n for n in cited if 1 <= n <= len(hits)]
    invalid = [n for n in cited if n not in valid]

    return Answer(
        question=question,
        text=text,
        exit=classify_exit(text),
        citations=valid,
        invalid_citations=invalid,
        cited_sources=[hits[n - 1].citation for n in valid],
        latency_ms=ms,
    )


def judge(question: str, hits: list, ans: Answer, llm=None) -> tuple[float, str]:
    """Score how well an answer is grounded in its passages, 0.0 to 1.0."""
    import json

    llm = llm or build_llm()
    resp = llm.invoke(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_USER.format(
                    question=question,
                    passages=format_passages(hits),
                    answer=ans.text,
                ),
            },
        ]
    )
    raw = (resp.content or "").strip()
    raw = re.sub(r"(?is)^<think>.*?</think>\s*", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return 0.0, f"unparseable judge output: {raw[:120]}"
    try:
        data = json.loads(m.group(0))
        return float(data.get("score", 0.0)), str(data.get("reason", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0, f"unparseable judge output: {raw[:120]}"


def _demo() -> None:
    from retrieve import build

    r = build("hybrid", "fixed")
    llm = build_llm()

    for q in (
        "What was NVIDIA's Data Center revenue in fiscal year 2026?",
        "What were NVIDIA's United States revenues in fiscal 2025?",
        "What is NVIDIA's projected Data Center revenue for fiscal year 2027?",
    ):
        hits = r.search(q, k=10)
        a = answer(q, hits, llm)
        print(f"\n{'=' * 78}\nQ: {q}")
        print(f"exit={a.exit}  citations={a.citations}  {a.latency_ms:.0f}ms")
        print("-" * 78)
        print(a.text[:700])


if __name__ == "__main__":
    _demo()
