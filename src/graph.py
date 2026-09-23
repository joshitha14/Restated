"""LangGraph agent: retrieve -> grade -> rewrite once -> generate -> verify -> refuse.

Built LAST, deliberately. If every measurement had run through a loop with a
retry node, a score improving would be ambiguous: better chunking, or the retry
happening to fire? The linear path in evaluate.py produced the numbers first;
this graph is measured against them.

    retrieve ──> grade ──(sufficient)──> generate ──> verify ──> END
                   │                                     │
                   │(insufficient, once)                 │(ungrounded)
                   ↓                                     ↓
                rewrite ──────────────> retrieve       refuse ──> END

Why each node earns its place:

  grade     The measured failure mode was retrieval depth, not hallucination:
            at k=10 the answer was absent and the system refused; at k=20 it
            answered correctly. A grader that detects "the figure the question
            asks for is not in these passages" can trigger one deeper retry
            instead of refusing a question the corpus can answer.

  rewrite   Financial queries fail lexically in a specific way -- a user writes
            "earnings per share", the filing says "net income per share". One
            rewrite with the filing's vocabulary is cheap and targeted.

  verify    Citations are checked against the passages actually supplied, so a
            citation to a passage that was never provided is caught. The model
            cannot invent a source that survives verification.

Retry is capped at ONE. An uncapped loop can spend unbounded credits on a
question the corpus genuinely cannot answer, and "unanswerable" is a category
this eval set deliberately contains.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import answer as generate_answer  # noqa: E402
from generate import build_llm  # noqa: E402
from prompts import PROMPT_VERSION  # noqa: E402
from retrieve import build as build_retriever  # noqa: E402

# Retrieval depths. The grader escalates from the first to the second exactly
# once; both come from the measured curve in reports/retrieval_scorecard.json,
# where numeric accuracy rose sharply between k=10 and k=20.
K_FIRST = 10
K_RETRY = 20

MAX_REWRITES = 1

# A question asking for a figure. Used by the grader: if the question wants a
# number and no passage contains one in a table, retrieval probably came up short.
NUMERIC_QUESTION_RE = re.compile(
    r"\b(how much|what was|what were|revenue|income|earnings|per share|margin|"
    r"expense|assets|shares|total)\b",
    re.I,
)


class State(TypedDict, total=False):
    """Everything the graph carries between nodes."""

    question: str
    query: str            # possibly rewritten
    hits: list
    k: int
    rewrites: int
    answer: object        # generate.Answer
    grounded: bool
    trace: Annotated[list[str], lambda a, b: a + b]
    retrieval_ms: float
    generation_ms: float


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def make_retrieve(retriever):
    def retrieve(state: State) -> dict:
        k = state.get("k", K_FIRST)
        query = state.get("query") or state["question"]
        t0 = time.perf_counter()
        hits = retriever.search(query, k=k)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "hits": hits,
            "retrieval_ms": state.get("retrieval_ms", 0.0) + ms,
            "trace": [f"retrieve(k={k}, query={query[:48]!r}) -> {len(hits)} hits"],
        }

    return retrieve


def grade(state: State) -> dict:
    """Do these passages plausibly contain the answer?

    Deliberately a cheap heuristic, not an LLM call. An LLM grader would add a
    round trip to every question to answer something the text mostly settles:
    a question asking for a figure needs a passage containing figures.
    """
    hits = state.get("hits", [])
    question = state["question"]

    if not hits:
        return {"trace": ["grade: no hits -> insufficient"]}

    wants_number = bool(NUMERIC_QUESTION_RE.search(question))
    has_table = any(h.block_type == "table" for h in hits)
    has_digits = any(re.search(r"\d{1,3}(?:,\d{3})+", h.text) for h in hits)

    sufficient = (not wants_number) or has_table or has_digits
    return {
        "trace": [
            f"grade: wants_number={wants_number} has_table={has_table} "
            f"has_digits={has_digits} -> {'sufficient' if sufficient else 'insufficient'}"
        ]
    }


def should_retry(state: State) -> Literal["rewrite", "generate"]:
    hits = state.get("hits", [])
    question = state["question"]

    if state.get("rewrites", 0) >= MAX_REWRITES:
        return "generate"
    if not hits:
        return "rewrite"

    wants_number = bool(NUMERIC_QUESTION_RE.search(question))
    has_table = any(h.block_type == "table" for h in hits)
    has_digits = any(re.search(r"\d{1,3}(?:,\d{3})+", h.text) for h in hits)

    if wants_number and not (has_table or has_digits):
        return "rewrite"
    return "generate"


# Query terms -> the vocabulary the filings actually use. Financial questions
# fail lexically in a predictable way, and BM25 is exact-match, so aligning the
# words is worth more than re-embedding the same question.
REWRITE_TERMS = {
    r"\bearnings per share\b": "net income per share basic diluted",
    r"\bEPS\b": "net income per share",
    r"\brevenue\b": "revenue total revenue by end market",
    r"\bprofit\b": "gross profit operating income net income",
    r"\bsales\b": "revenue",
    r"\bgrowth\b": "increase compared to fiscal year",
    r"\bgeographic\b": "revenue by geographic area",
    r"\bshares outstanding\b": "weighted average shares",
}


def rewrite(state: State) -> dict:
    """Rewrite once, toward the filings' vocabulary, and search deeper."""
    query = state.get("query") or state["question"]
    new = query
    applied = []
    for pattern, replacement in REWRITE_TERMS.items():
        if re.search(pattern, new, re.I):
            new = re.sub(pattern, replacement, new, flags=re.I)
            applied.append(pattern.strip(r"\b"))

    return {
        "query": new,
        "k": K_RETRY,
        "rewrites": state.get("rewrites", 0) + 1,
        "trace": [f"rewrite: k->{K_RETRY}, terms={applied or 'none (depth only)'}"],
    }


def make_generate(llm):
    def generate(state: State) -> dict:
        ans = generate_answer(state["question"], state.get("hits", []), llm)
        return {
            "answer": ans,
            "generation_ms": state.get("generation_ms", 0.0) + ans.latency_ms,
            "trace": [f"generate: exit={ans.exit} citations={ans.citations}"],
        }

    return generate


def verify(state: State) -> dict:
    """Check the answer's citations resolve, and its figures have a source.

    A refusal is grounded by definition -- it asserts nothing about the filings.
    """
    ans = state["answer"]
    hits = state.get("hits", [])

    if ans.exit == "REFUSE":
        return {"grounded": True, "trace": ["verify: refusal, grounded by definition"]}

    if ans.invalid_citations:
        return {
            "grounded": False,
            "trace": [f"verify: citations {ans.invalid_citations} do not exist -> ungrounded"],
        }

    blob = "\n".join(h.text for h in hits)
    stated = set(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", ans.text))
    unsourced = [s for s in stated if s not in blob and s.replace(",", "") not in blob]

    if unsourced:
        return {
            "grounded": False,
            "trace": [f"verify: unsourced figures {unsourced[:4]} -> ungrounded"],
        }

    return {"grounded": True, "trace": ["verify: all figures and citations sourced"]}


def route_after_verify(state: State) -> Literal["refuse", "__end__"]:
    return "__end__" if state.get("grounded") else "refuse"


def refuse(state: State) -> dict:
    """Replace an ungrounded answer with an explicit refusal.

    A confident wrong figure is far worse than "I don't know", so an answer that
    fails verification is not returned even though the model produced it.
    """
    ans = state["answer"]
    ans.text = (
        "The filings provided do not support a reliable answer to this question. "
        "The retrieved passages did not contain the figure requested, and the "
        "drafted answer could not be verified against them."
    )
    ans.exit = "REFUSE"
    return {"answer": ans, "trace": ["refuse: ungrounded answer replaced"]}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------

def build_graph(retriever=None, llm=None, strategy: str = "fixed", kind: str = "hybrid"):
    from langgraph.graph import END, START, StateGraph

    retriever = retriever or build_retriever(kind, strategy)
    llm = llm or build_llm()

    g = StateGraph(State)
    g.add_node("retrieve", make_retrieve(retriever))
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", make_generate(llm))
    g.add_node("verify", verify)
    g.add_node("refuse", refuse)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", should_retry, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"refuse": "refuse", "__end__": END})
    g.add_edge("refuse", END)

    return g.compile()


def ask(app, question: str, k: int = K_FIRST) -> State:
    return app.invoke(
        {"question": question, "query": question, "k": k, "rewrites": 0, "trace": []}
    )


def _demo() -> None:
    app = build_graph()
    print(f"prompt version: {PROMPT_VERSION}\n")

    for q in (
        "What was NVIDIA's Data Center revenue in fiscal year 2026?",
        "What was NVIDIA's basic earnings per share in fiscal 2024?",
        "What is NVIDIA's projected Data Center revenue for fiscal year 2027?",
    ):
        state = ask(app, q)
        ans = state["answer"]
        print(f"{'=' * 78}\nQ: {q}")
        for step in state["trace"]:
            print(f"   . {step}")
        print(f"   exit={ans.exit}  rewrites={state.get('rewrites', 0)}  "
              f"grounded={state.get('grounded')}")
        print("-" * 78)
        print(ans.text[:420])
        print()


if __name__ == "__main__":
    _demo()
