# restated

A measurement harness for retrieval over NVIDIA's 10-K filings.

**Research question:** does chunking strategy affect whether the system
retrieves the *original* or the *restated* figure — and does reranking help or
hurt?

The pipeline is the apparatus. The report is the deliverable.

---

## Why this corpus

Three consecutive NVIDIA 10-Ks (FY2024, FY2025, FY2026) from SEC EDGAR — one
company across periods, rather than the more obvious many-companies-one-year
design.

| Fiscal year | Period end | Filed | Accession |
|---|---|---|---|
| FY2026 | 2026-01-25 | 2026-02-25 | `0001045810-26-000021` |
| FY2025 | 2025-01-26 | 2025-02-26 | `0001045810-25-000023` |
| FY2024 | 2024-01-28 | 2024-02-21 | `0001045810-24-000029` |

CIK `0001045810`, resolved from the ticker at runtime. `data/raw/` is gitignored;
`src/fetch.py` reproduces the corpus byte-for-byte.

Note each filing date is ~1 month after its period end. Tagging `fiscal_year`
from the filing date would shift every chunk into the wrong year — which is why
it comes from inline XBRL instead (DECISIONS.md D2).

A cross-company corpus tests **attribution** (don't mix up AMD's number with
Intel's). A single-company multi-period corpus tests **period confusion**, which
is harder for dense retrieval: every 10-K says "revenue increased" in
near-identical language year over year, so the embeddings cannot separate the
years and metadata has to do the work.

Three years rather than two gives two YoY comparisons plus trend questions
requiring all three filings at once.

HTML rather than PDF, because EDGAR HTML preserves `<table>` structure. For a
project where table handling is the central problem, that is decisive.

## What the corpus actually contains

The interesting part was not assumed — it was found by scanning for line items
whose prior-year figure changed between filings (`src/find_restatements2.py`).
Two independent traps fell out:

### 1. Reclassification

NVIDIA changed the basis of geographic revenue disaggregation between filings,
and says so in the table caption:

- FY2025 filing: "based upon the **billing location of the customer**"
- FY2026 filing: "based upon the **location of the customers' headquarters**"

So FY2025 geographic revenue has two correct values:

| Line item (FY2025) | Per FY2026 filing (HQ) | Per FY2025 filing (billing) | Ratio |
|---|---|---|---|
| United States | 77,482 | 61,257 | 1.26x |
| China (incl. HK) | 25,048 | 17,108 | 1.46x |
| Other | 4,367 | 7,875 | 0.55x |

US and China rise while "Other" falls — what a billing→HQ reattribution looks
like. Each figure appears in only one filing, so which number the system returns
is decided entirely by which chunk it retrieves.

Notably, revenue *by end market* is **stable** across filings (Data Center
FY2025 is `115,186` in both). Had the eval set been written against revenue —
the obvious choice — the restatement category would have measured nothing.

### 2. The 10-for-1 stock split

NVIDIA split 10-for-1 in June 2024. The FY2024 filing states per-share figures
pre-split; later filings restate them:

| Line item (FY2024) | Per FY2025/26 filings | Per FY2024 filing |
|---|---|---|
| Basic EPS | 1.21 | 12.05 |
| Diluted weighted average shares | 24,940 | 2,494 |

Exactly 10x, unambiguous, both correct. A system answering "$12.05" is right
about the number and wrong about the basis — only a citation reveals which.

Counts across the corpus: **33 reclassifications, 19 split-adjusted line items**
(`src/find_restatements2.py`, keyed on table topic over cleaned blocks).

> An earlier scan reported Taiwan FY2025 as a 13.9x restatement (`1,481` vs
> `20,573`). That was wrong — a label collision between a *long-lived assets*
> table and a *revenue* table, both with a row labelled `Taiwan`. The diagnostic
> had reproduced the exact failure mode the pipeline is designed to avoid. See
> DECISIONS.md D4 for the correction; it is kept in the record because it is the
> clearest evidence for why `table_caption` is load-bearing.

## Method

```
EDGAR HTML
  → clean   (drop layout tables, fold $-cells, tag Item sections, XBRL metadata)
  → chunk   THREE ways: fixed / SemanticChunker / structural
            (tables held whole in ALL THREE — the experimental control)
  → embed   → three Chroma collections, one per strategy
  → retrieve dense | BM25 | hybrid (reciprocal rank fusion) | metadata-filtered
  → generate with citations, or refuse
  → graph   retrieve → grade → rewrite once → generate → verify → refuse
```

A **cross-encoder reranker** (`BAAI/bge-reranker-base`, local via
sentence-transformers) is built and measured as runs F and G. It works — and it
is **dominated by metadata filtering**, which reaches the same accuracy for 3% of
the latency. See Results.

### The experimental control that matters

**Tables are kept whole by both chunking strategies.** If only one protects
tables, the comparison measures table handling rather than boundary placement,
and the result means nothing.

Both strategies route table blocks through the same code path and differ *only*
in how they group prose:

| | fixed | semantic |
|---|---|---|
| chunks | 1,435 | 1,421 |
| prose median chars | 918 | 777 |
| prose max chars | 1,151 | 1,000 |
| table chunks | 173 | 173 (byte-identical) |

`src/verify_chunks.py` asserts this rather than assuming it — five controls,
including that table text is **byte-identical** across strategies, and that all
19 eval-critical figures survive in both. It runs before any indexing: if a
control fails, the A/B result is void, and that is cheaper to learn before
spending embedding credits.

### Runs

| Run | Chunking | Retrieval | Isolates |
|---|---|---|---|
| A | Fixed-size | Dense only | Baseline |
| B | Semantic | Dense only | Chunking effect |
| C | Better of A/B | Hybrid + rerank | Reranking effect |
| D | Better of A/B | Metadata-filtered | `years_present` as a hard filter |

### Embeddings

`Qwen/Qwen3-Embedding-8B` via Nebius Token Factory (OpenAI-compatible endpoint,
`base_url=https://api.tokenfactory.nebius.com/v1/`) — the only embedding model
the account exposes. Two Chroma collections, 2,871 vectors, ~184s to build.

`OpenAIEmbeddings` needs `check_embedding_ctx_length=False` against Nebius,
which otherwise rejects LangChain's pre-tokenized input with
`400 {'detail': 'Tokenized input is not supported'}`.

### Results

**Retrieval** (k=20, 16 scoreable questions):

Isolated runs — one configuration per process. See *batch instability* below.

| Run | Chunking | Retrieval | numeric | yr_prec | both sides | p50 |
|---|---|---|---|---|---|---|
| A | fixed | dense | 68.8% | 70.9% | 33% | 182ms |
| B | SemanticChunker | dense | 75.0% | 73.1% | 33% | 180ms |
| B2 | structural | dense | 68.8% | 71.2% | 33% | 201ms |
| C | fixed | hybrid | 75.0% | 67.8% | **67%** | 194ms |
| D | fixed | filtered | 75.0% | **90.9%** | 33% | 200ms |
| E | fixed | BM25 only | 68.8% | 61.6% | **83%** | **2.9ms** |
| F | fixed | **+ rerank** | **81.2%** | 74.7% | **67%** | 9,482ms |
| G | fixed | rerank + filter | 75.0% | **92.2%** | 50% | 12,553ms |

**Chunking strategy matters less than retrieval method.** The three chunking arms
span 68.8–75.0%. Swapping dense for metadata-filtered on the *same* chunks
reaches 81.2%.

**Dense retrieval cannot find exact figures.** EPS `1.21` is not in dense's top
50 results; BM25 ranks it **first**. Per-category numeric accuracy on `split`:
dense 0%, BM25 and hybrid 100%. Embeddings encode meaning, and `1.21` is a token,
not a meaning.

**BM25 surfaces both sides of a restatement 83% of the time** versus 33% for
dense — in 2.3ms against dense's 200ms, because it makes no API call.

**Reranking** (`BAAI/bge-reranker-base`, local) is the only configuration that
beats 75% numeric accuracy — run F at **81.2%**, a 6.2-point gain over the hybrid
retrieval it re-scores. It costs **49x the latency** to get there: 9.5s against
194ms.

**Stacking it with metadata filtering is worse than either alone.** Run G returns
to 75.0% and halves both-sides coverage. Filtering narrows the pool to one fiscal
year, so the reranker can no longer surface the *other* filing's version of a
restated figure — the two work against each other on exactly the questions this
corpus was built to test.

**Batch instability.** Running several configurations in one process yields
different scores than running each alone: C and D both score 81.2% in a full A–G
batch but 75.0% in isolation, reproducibly. The cause is not identified; a
related Chroma defect in the same area is documented in DECISIONS.md D23. Every
figure here is the **isolated** one — the conservative choice, independent of
execution order, and no conclusion changes under the batch figures.

**Generation** (hybrid retrieval, k=20, 19 questions):

| Metric | Score |
|---|---|
| exit correct (ANSWER/CLARIFY/REFUSE) | 84.2% |
| numeric correct | 62.5% |
| both sides shown | 50.0% |
| faithfulness (LLM judge) | 1.00 |
| hallucination rate | **5.3%** (1 of 19) |
| invalid citations | 0.0% |

**One hallucination in 19 questions.** Refusal was 100% correct on unanswerable
questions, and the dominant failure mode is declining to answer rather than
inventing. The exception: on a trend question requiring arithmetic across three
years, the model stated `123,411` — an intermediate value it computed, presented
as though retrieved. The LLM faithfulness judge scored that answer 1.00; the
mechanical figure-level check caught it. Across 3 repeated runs the linear path
hallucinated in 1 of 3 and the LangGraph path in 0 of 3.

**LangGraph loop vs the linear path** (same eval set, same retriever):

| Metric | Linear | Graph | Delta |
|---|---|---|---|
| exit correct | 89.5% | 84.2% | −5.3 |
| numeric correct | 62.5% | **81.2%** | **+18.8** |
| both sides shown | 50.0% | **66.7%** | **+16.7** |
| faithfulness | 1.00 | 1.00 | 0.0 |
| hallucination rate | 0.0% | 0.0% | 0.0 |
| p50 latency | 3,301ms | 6,884ms | **+3,583** |

**The loop is a real quality win at roughly 2x latency.** The gain comes almost
entirely from grade-and-retry: 8 of 19 questions triggered one rewrite, and the
k=10 → k=20 escalation closes exactly the gap the depth curve identified.

Whether that trade is worth taking depends on the **latency ceiling** — a
first-class constraint in the brief that this project never pinned down. At a
5-second budget the graph does not qualify; at 20 seconds it clearly does.

The **verify node never fired**: across 19 questions the generator never produced
an unsourced figure or an invalid citation. Good for the system, and an honest
finding about the node — on this corpus it is insurance, not a contributor.

*Caveat:* n=16 scoreable questions, so one question is 6.25 points. These are
directional findings. The per-category table matters more than the aggregate.

### Metrics

recall@k, faithfulness, **numeric accuracy** (exact match on figures), refusal
accuracy, p95 latency. Reranking is reported **with timings** — a quality win
that costs two seconds is a tradeoff, not a free win.

Scored per query category, never only in aggregate: aggregate scores hide where
systems break.

## Eval set

19 questions in `eval/questions.yaml`, across six categories: numeric (4),
temporal (3), trend (3), restatement (4), split (2), unanswerable (3). The three
unanswerable questions are not scoreable for numeric accuracy, so retrieval
metrics are over 16.

Written **before** retrieval was built — this is the one step that cannot be
recovered later, and it is what stops the eval being tuned toward whatever the
pipeline already does well.

Every expected figure is verified against the filings by `src/verify_eval.py`
(22/22 passing). An eval set with a wrong expected value scores a correct system
as wrong, which is worse than no eval set.

## Layout

```
restated/
├── data/
│   ├── raw/                   # three .htm filings (gitignored)
│   ├── clean/                 # FY{year}.jsonl blocks (gitignored)
│   ├── chunks/                # {fixed,semantic,semantic_embed}/ (gitignored)
│   └── chroma/                # vector store (gitignored)
├── notebooks/main.ipynb       # the narrative: runs top to bottom
├── src/
│   │  pipeline
│   ├── fetch.py               # ticker → CIK → filings
│   ├── clean.py               # HTML → section-tagged, table-aware blocks
│   ├── chunk.py               # the three splitters
│   ├── index.py               # embed → Chroma, one collection per strategy
│   ├── retrieve.py            # dense / BM25 / hybrid (RRF) / metadata-filtered
│   ├── prompts.py             # versioned; v2 adds the restatement rule
│   ├── generate.py            # cited answers, or refusal
│   ├── graph.py               # LangGraph: grade → rewrite once → verify
│   │
│   │  measurement
│   ├── evaluate.py            # retrieval scorecard, curve over k
│   ├── evaluate_gen.py        # generation scorecard + faithfulness judge
│   ├── evaluate_graph.py      # graph vs linear — does the loop earn its cost?
│   ├── compare_runs.py        # per-question view across runs
│   ├── test_classify.py       # exit classifier, cases from real answers
│   │
│   │  diagnostics
│   ├── find_restatements2.py  # topic-keyed scan over cleaned blocks
│   ├── debug_reach.py         # is the answer indexed, and at what rank?
│   ├── debug_gen.py           # retrieval failure or generation failure?
│   ├── debug_eval.py          # why did a question score that way?
│   ├── debug_graph.py         # right figure, wrong exit — why?
│   ├── inspect_chunks.py      # chunk size distribution / outliers
│   │
│   │  verification — each asserts something the report depends on
│   ├── check_env.py           # packages, corpus, chunks, API key
│   ├── verify_eval.py         # eval figures vs. filings (22/22)
│   ├── verify_clean.py        # cleaned blocks preserve those figures
│   ├── verify_citations.py    # no blank citations
│   ├── verify_chunks.py       # the A/B experiment is validly controlled
├── eval/questions.yaml
├── reports/                   # scorecards, committed
├── DECISIONS.md               # what was decided and why (D1–D17)
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in both keys
```

Two credentials:

- `NEBIUS_API_KEY` — from [tokenfactory.nebius.com](https://tokenfactory.nebius.com/project/api-keys)
  (**Token Factory**, not AI Cloud)
- `SEC_USER_AGENT` — a real name and email. SEC blocks requests without one;
  this is the most common reason a fetch script silently fails.

**Easiest path: open `notebooks/main.ipynb` and run it top to bottom.** It is the
narrative version of everything below.

Or from the command line:

```bash
python src/check_env.py         # packages, corpus, chunks, API key
python src/fetch.py             # download three 10-Ks
python src/clean.py             # → data/clean/FY{2024,2025,2026}.jsonl
python src/verify_clean.py      # assert critical figures survived
python src/verify_citations.py  # assert no blank citations
python src/verify_eval.py       # assert eval figures match the filings
python src/chunk.py --strategy all   # → data/chunks/{fixed,semantic,semantic_embed}/
python src/verify_chunks.py     # assert the A/B experiment is controlled
python src/index.py --strategy all   # embed → data/chroma/  (spends credits)

python src/evaluate.py --runs A,B,B2,C,D,E --k 5,10,20   # retrieval scorecard
python src/evaluate_gen.py --k 20 --retriever hybrid      # generation scorecard
python src/graph.py                                       # the agent, with traces
python src/evaluate_graph.py                              # graph vs linear
```

`src/index.py --limit 20` is a smoke test: 20 chunks per strategy, enough to
confirm credentials and batching before committing the full corpus to embedding
credits. `--list-models` shows what the account exposes.

`src/clean.py --dumb` is the fallback: keeps all tables as raw text without
layout-table filtering. A degraded cleaner with a completed comparison beats a
perfect cleaner and no results.

## Build order

Chunking and embedding are pure LangChain. LangGraph enters only at query time.

1. Ingest ✅
2. Chunk — two strategies ✅
3. Embed + store — three collections ✅
4. Retrieve — dense, BM25, hybrid, metadata-filtered ✅
5. **Measure** — retrieval + generation scorecards ✅
6. **Only then LangGraph** — retrieve → grade → rewrite once → generate →
   verify → refuse ✅
4. Retrieve — dense, then hybrid + rerank
5. **Measure**
6. **Only then LangGraph** — retrieve → grade → rewrite-and-retry once →
   generate → verify citations → refuse if ungrounded

**Do not build the graph first.** If every measurement runs through a loop with
a retry node, you cannot tell whether a score improved because of the chunking
change or because the retry happened to fire. The measurement path stays linear
until the numbers are locked.
