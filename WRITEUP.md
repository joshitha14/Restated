# restated — Financial Document Intelligence Pipeline RAG

**The Gen Academy — Mastering Agentic AI Bootcamp, Week 2, Project 2**
Track 2: LangChain + LangGraph

**Research question:** does chunking strategy affect whether the system retrieves
the *original* or the *restated* figure — and does retrieval method matter more?

---

## 1. Project overview

The graded deliverable for Project 2 is a **measurement**, not an assistant: a
chunking-strategy comparison and a retrieval-impact analysis. So the pipeline was
built as an apparatus for producing numbers, and the numbers are the result.

Three things follow from that framing, and they shaped every decision below:

1. **The eval question set was written before retrieval existed.** This is the
   one step that cannot be recovered later — writing questions after seeing what
   the pipeline retrieves well is how an eval set gets quietly tuned toward the
   system's existing strengths.
2. **The experimental control is asserted in code, not assumed.** Tables are held
   whole by every chunking strategy; `src/verify_chunks.py` checks that their
   text is *byte-identical* across arms before any indexing runs.
3. **The LangGraph loop was built last.** If every measurement ran through a
   retry node, a score improving would be ambiguous — better chunking, or the
   retry happening to fire?

### Headline results

**Chunking strategy matters less than retrieval method.** The three chunking arms
span 68.8–75.0% numeric accuracy. Swapping dense retrieval for metadata-filtered
retrieval on the *same* chunks reaches 81.2%.

**Dense retrieval cannot find exact figures.** EPS `1.21` does not appear in
dense retrieval's top 50 results; BM25 ranks it **first**. Embeddings encode
meaning, and `1.21` is a token, not a meaning.

**One hallucination in 19 questions, and it was caught mechanically.** The
system's dominant failure mode is declining to answer rather than inventing —
but the single exception is instructive: a computed intermediate value presented
as though retrieved, on a question requiring arithmetic across three years. The
LLM faithfulness judge scored that answer 1.00; a mechanical figure-level check
caught it. See §5.

---

## 2. Datasets used

**NVIDIA (NVDA), three consecutive 10-K filings, HTML from SEC EDGAR.**

| Fiscal year | Period end | Filed | Accession | Size |
|---|---|---|---|---|
| FY2026 | 2026-01-25 | 2026-02-25 | `0001045810-26-000021` | 1,967,816 B |
| FY2025 | 2025-01-26 | 2025-02-26 | `0001045810-25-000023` | 2,067,520 B |
| FY2024 | 2024-01-28 | 2024-02-21 | `0001045810-24-000029` | 2,085,566 B |

CIK `0001045810`, resolved from the ticker at runtime against SEC's official
mapping rather than hardcoded — a wrong CIK fails in the worst way, silently
returning a *different company's* filings instead of erroring.

### Why one company across periods, not many companies in one year

A cross-company corpus (the obvious design) tests **attribution**: don't report
AMD's number when asked about Intel. A single-company multi-period corpus tests
**period confusion**, which is harder for dense retrieval, because every 10-K
says "revenue increased" in near-identical language year over year. The
embeddings cannot separate the years; metadata has to do the work.

Three years rather than two gives two year-over-year comparisons plus trend
questions requiring all three filings simultaneously.

### Why HTML, not PDF

EDGAR HTML preserves `<table>` structure. A PDF has only a text layer, so column
alignment must be inferred. For a project where table handling is the central
problem, this is decisive.

### What the corpus actually contains

The interesting property of this corpus was **found, not assumed**. A diagnostic
(`src/find_restatements2.py`) scanned for line items whose prior-year figure
changed between filings. Two independent traps fell out.

**Trap 1 — reclassification (33 line items).** NVIDIA changed the basis of
geographic revenue disaggregation between filings, and states the change in the
table caption itself:

- FY2025 filing: "based upon the **billing location of the customer**"
- FY2026 filing: "based upon the **location of the customers' headquarters**"

| Line item (FY2025) | Per FY2026 filing (HQ) | Per FY2025 filing (billing) |
|---|---|---|
| United States | 77,482 | 61,257 |
| China (incl. HK) | 25,048 | 17,108 |
| Other | 4,367 | 7,875 |

US and China rise while "Other" falls — what a billing→HQ reattribution looks
like. Each figure appears in **only one filing**, so which number the system
returns is decided entirely by which chunk it retrieves.

**Trap 2 — the 10-for-1 stock split (19 line items).** NVIDIA split 10-for-1 in
June 2024, so the FY2024 filing states per-share figures pre-split while later
filings restate them:

| Line item (FY2024) | Per FY2025/26 | Per FY2024 |
|---|---|---|
| Basic EPS | 1.21 | 12.05 |
| Diluted weighted average shares | 24,940 | 2,494 |

Exactly 10x, unambiguous, both correct as published. A system answering "$12.05"
is right about the number and **wrong about the basis** — only a citation reveals
which.

**What is *not* restated matters too.** Revenue by end market is stable across
filings (Data Center FY2025 is `115,186` in both). Had the eval set been written
against revenue — the obvious choice — the restatement category would have
measured nothing.

---

## 3. Prompts and agent instructions

Prompts live in `src/prompts.py` with a version string, because the writeup has
to report which prompt produced which numbers and reconstructing that at the end
is miserable. Currently **v2**.

### The three-exit contract

The system must choose exactly one of:

- **ANSWER** — state the figure exactly as the filing prints it, cite the passage
  as `[n]`, show any arithmetic
- **CLARIFY** — the question is ambiguous as asked; give the figure for each
  reading
- **REFUSE** — the passages do not contain it. *A confident wrong figure is far
  worse than "I don't know."*

### The restatement rule (the v1 → v2 change)

Added after observing v1 silently pick one of two conflicting figures without
indicating the other existed. The operative instruction:

> If the passages contain two different values for the same line item and
> period, you MUST report BOTH, say which filing each came from, and explain why
> they differ if the passages say. **Reporting one silently is a wrong answer
> even when the number you picked is correct.**

The prompt names both known mechanisms (the geographic reclassification and the
stock split) with concrete figures, so the model has a worked example of the
pattern rather than an abstract rule.

### Faithfulness judge

A separate prompt scores groundedness 0.0–1.0: every claim must trace to a
passage or to arithmetic on figures in the passages. A *correct refusal* scores
1.0 — it asserts nothing about the filings, so it is grounded by definition.

### Citation handling

Citations are **parsed out of the answer with a regex and resolved against the
passages actually supplied**, rather than trusted. A citation to a passage that
was never provided is detectable, so the model cannot invent a source that scores
as grounded.

---

## 4. Method

```
EDGAR HTML
  → clean    (drop layout tables, fold $-cells, tag Item sections, XBRL metadata)
  → chunk    THREE ways (tables held whole in ALL THREE — the control)
  → embed    → three Chroma collections, one per strategy
  → retrieve dense | BM25 | hybrid (RRF) | metadata-filtered
  → generate cited answer, or refusal
  → graph    retrieve → grade → rewrite once → generate → verify → refuse
```

**Models:** `Qwen/Qwen3-Embedding-8B` for embeddings and
`Qwen/Qwen3-235B-A22B-Instruct-2507` for generation, both via Nebius Token
Factory's OpenAI-compatible endpoint. The embedding model was not a preference —
it is the only one the account exposes.

### The experimental control

**Tables are held whole by all three chunking strategies.** If only one protected
tables, the comparison would measure table handling rather than boundary
placement, and the result would mean nothing.

A table exceeding the size budget is *still* not split: a half table is worse
than a long chunk, because the header row carries the fiscal years and without it
every figure is unattributable.

`src/verify_chunks.py` asserts five controls before any indexing:

1. Table chunks are **byte-identical** across strategies (not merely equal in count)
2. No table lost its header separator row
3. Prose median sizes within 1.5x (measured: 1.00x)
4. All 19 eval-critical figures present in every strategy
5. No blank citations, sections, fiscal years, or duplicate chunk IDs

| | fixed | SemanticChunker | structural |
|---|---|---|---|
| chunks | 1,450 | 723 | 1,421 |
| prose median chars | 808 | 924 | 777 |
| prose min / max | 92 / 999 | **3 / 19,154** | 77 / 1,000 |

`SemanticChunker` produced both a **3-character chunk** and a **19,154-character
chunk**. It has no size ceiling, so one chunk is 19x the fixed arm's maximum —
long enough that its embedding averages away whatever it is about.

### Eval set

19 questions in `eval/questions.yaml`: numeric (4), temporal (3), trend (3),
restatement (4), split (2), unanswerable (3). The three unanswerable questions
are not scoreable for numeric accuracy, so retrieval metrics are over 16.

Every expected figure is verified against the filings by `src/verify_eval.py`
(22/22 passing). An eval set with a wrong expected value scores a correct system
as wrong — worse than having no eval set.

---

## 5. Results

### Retrieval (k=20, 16 scoreable questions)

All figures below are from **isolated runs** — one configuration per process.
See the note on batch instability at the end of this section.

| Run | Chunking | Retrieval | numeric | yr_prec | both sides | p50 |
|---|---|---|---|---|---|---|
| A | fixed | dense | 68.8% | 70.9% | 33% | 182ms |
| B | SemanticChunker | dense | 75.0% | 73.1% | 33% | 180ms |
| B2 | structural | dense | 68.8% | 71.2% | 33% | 201ms |
| C | fixed | hybrid | 75.0% | 67.8% | **67%** | 194ms |
| D | fixed | filtered | 75.0% | **90.9%** | 33% | 200ms |
| E | fixed | BM25 only | 68.8% | 61.6% | **83%** | **2.9ms** |

### Numeric accuracy by category

| Run | numeric | temporal | trend | restatement | **split** |
|---|---|---|---|---|---|
| A (dense) | 100% | 100% | 67% | 50% | **0%** |
| B (dense) | 100% | 100% | 67% | 75% | **0%** |
| C (hybrid) | 100% | 67% | 67% | 50% | **100%** |
| E (BM25) | 75% | 67% | 0% | **100%** | **100%** |

**Dense retrieval scores 0% on the split trap. BM25 scores 100%.** Searching to
depth 50 shows why:

| Question | dense | BM25 | hybrid |
|---|---|---|---|
| q_numeric_01 (`193,737`) | 12 | 20 | 12 |
| q_split_01 (EPS `1.21`) | **not in top 50** | **1** | 5 |
| q_split_02 (shares `24,940`) | 35 | **1** | 2 |
| q_restate_01 (US revenue) | **not in top 50** | **17** | 32 |

**BM25 surfaces both sides of a restatement 83% of the time versus dense's 33%**
— in 2.3ms against dense's 200ms, because it makes no API call. For questions
whose entire point is that two filings disagree, lexical matching on the figures
themselves beats semantic similarity outright.

**Metadata filtering is the cheapest win available on period confusion:** year
precision 70.9% → 90.9% for ~20ms. This is the problem the corpus was chosen to
expose being solved by metadata rather than by a better embedding. It does not
help numeric accuracy on its own (75.0%, same as hybrid).

### Reranking (runs F and G)

| Run | Retrieval | numeric | yr_prec | both sides | p50 |
|---|---|---|---|---|---|
| C | hybrid | 75.0% | 67.8% | 67% | **194ms** |
| D | filtered | 75.0% | **90.9%** | 33% | **200ms** |
| F | **+ rerank over hybrid** | **81.2%** | 74.7% | 67% | 9,482ms |
| G | + rerank over filtered | 75.0% | **92.2%** | 50% | 12,553ms |

`BAAI/bge-reranker-base`, run locally so the added latency is CPU time rather
than network time — which is what makes it stable enough to quote.

**Reranking is the only configuration that beats 75% numeric accuracy** — run F
at 81.2%, a 6.2-point gain over the hybrid retrieval it re-scores. It costs
**49x the latency** (9.5s against 194ms) to get there.

**Stacking it with filtering is worse than either alone.** Run G returns to
75.0% and halves both-sides coverage (67% → 50%). Filtering narrows the
candidate pool to one fiscal year, so the reranker can no longer surface the
*other* filing's version of a restated figure — the two techniques work against
each other on exactly the questions this corpus exists to test.

### A note on batch instability

Running several configurations **in one process** produces different numbers
than running each alone. Runs C and D both score 81.2% inside a full A–G batch
but 75.0% in isolation, reproducibly in both cases.

The cause is not identified. It is bounded, though: each configuration is
perfectly deterministic *in isolation* (three consecutive runs of C gave 75.0%
every time), and the corpus, index and code are unchanged between the two
conditions. A related and separately reproducible defect in the same area —
Chroma raising `InternalError: Error executing plan` on metadata-filtered
queries after several collections have been opened in one process — is
documented in DECISIONS.md D23.

**Every figure in this document is the isolated one**, which is the conservative
choice: it is lower for C and D, and it does not depend on execution order.
No conclusion in this writeup changes under the batch figures.

### Generation (hybrid, k=20, 19 questions)

| Metric | Score |
|---|---|
| exit correct | 89.5% |
| numeric correct | 62.5% |
| both sides shown | 50.0% |
| faithfulness (LLM judge) | 1.00 |
| hallucination rate | **5.3%** (1 of 19) |
| invalid citations | 0.0% |

**The single hallucination is worth naming.** On `q_trend_03` — *"which end
market has grown fastest?"* — the model stated `123,411`, a figure that appears
in no filing. It is an intermediate value the model computed while comparing
growth across three years, then presented as though it came from the source.

This is the failure mode arithmetic invites: the prompt permits arithmetic on
retrieved figures and asks for the working to be shown, and a computed
intermediate is indistinguishable in form from a retrieved one. The faithfulness
judge scored the answer 1.00, because the arithmetic *was* grounded in the
passages — which is exactly why a mechanical figure-level check earns its place
alongside an LLM judge. The two disagree, and the mechanical one was right.

Across 3 repeated runs the linear path hallucinated in 1 of 3 runs and the
LangGraph path in 0 of 3 (`reports/variance.json`), so this is an occasional
failure rather than a systematic one.

### LangGraph loop vs the linear path

**A single run is not a measurement for every metric here.** `temperature=0.0`
is not determinism. Each configuration was therefore run 3x
(`src/evaluate_repeat.py`, results in `reports/variance.json`):

| Metric | Linear (mean, range) | Graph (mean, range) | Delta |
|---|---|---|---|
| exit correct | 78.9% (73.7–84.2) | 82.5% (78.9–84.2) | +3.6 *(within noise)* |
| **numeric correct** | **62.5%** (no spread) | **75.0%** (no spread) | **+12.5** |
| **both sides shown** | **50.0%** (no spread) | **66.7%** (no spread) | **+16.7** |
| faithfulness | 0.98 (0.9–1.0) | 0.9 (no spread) | — |
| hallucination rate | 1.8% (0.0–5.3) | **0.0%** (no spread) | −1.8 |
| p50 latency | 2,401ms (±22) | 3,298ms (±450) | **+897** |

**Which metrics are stable, and which are not, is itself the finding:**

- **Numeric accuracy and both-sides shown have *zero* spread** across all six
  runs. Whether the right figure is retrieved and stated is fully determined by
  the pipeline, not by sampling. The +12.5 and +16.7 point gains from the graph
  are therefore real, not noise.
- **Exit correctness swings 10.5 points on the linear path** and 5.3 on the
  graph. The three-way ANSWER/CLARIFY/REFUSE choice is genuinely unstable — the
  model sometimes answers a borderline question and sometimes clarifies it. The
  graph's apparent +3.6 exit advantage is *within* that spread and should not be
  claimed.
- **One linear run hallucinated (5.3%); the graph never did** across three runs.
  A single linear run would have reported either 0% or 5.3% with equal
  confidence.

**The loop is a real quality win at ~1.4x latency.** The gain comes almost
entirely from grade-and-retry: 8 of 19 questions triggered one rewrite, and the
k=10 → k=20 escalation closes exactly the gap identified by the depth curve.

Whether that trade is worth taking depends on the **latency ceiling** — a
first-class constraint in the brief that this project never pinned down. At a
2.5-second budget the graph does not qualify; at 5 seconds it clearly does.

**Caveat:** n=16 scoreable questions, so a one-question swing is 6.25 percentage
points. Differences in *exit correctness* below ~10 points are noise; differences
in *numeric accuracy* are not, because that metric does not vary between runs.
The per-category tables matter more than the aggregates.

---

## 6. Iterations tried

Full decision log in `DECISIONS.md` (D1–D19). The ones that changed the result:

### The headline finding was wrong, and the correction is the better finding

An early scan reported **Taiwan FY2025 as `1,481` vs `20,573` — a 13.9x
restatement**, and it was written up as the strongest case in the corpus.

It was wrong. The `1,481` comes from a **long-lived assets** table; the `20,573`
from a **revenue** table. Both rows are labelled `Taiwan`.

The diagnostic had committed the exact error the pipeline is designed to prevent:
**treating a line-item label as a unique key**. It is kept in the record because
it is the clearest possible evidence for why `table_caption` is load-bearing
rather than nice-to-have — without it, a system asked "revenue attributed to
Taiwan" would happily return a long-lived-assets figure.

Once captions were reliable, the *real* restatement surfaced with its cause
stated in the filings themselves (the billing→HQ basis change), which is a
stronger finding than an unexplained 14x jump.

### Five runs scored identically — that was a bug, not a result

The first retrieval scorecard reported **37.5% numeric accuracy for all five
runs**, to one decimal place. Five genuinely different configurations cannot
coincide that exactly. Two bugs:

1. **Three questions were unscoreable.** The figure extractor handled ints and
   dicts but not prose, so questions whose expected value reads
   `"193737 - 115186 = 78551 (68.2% growth)"` yielded *no* figures and could
   never register a hit. A metric that cannot be satisfied depresses every run by
   the same amount, which looks like a valid comparison.
2. **k=5 was too shallow to measure anything.** The answers were indexed all
   along, just ranked below the cutoff.

Fixed by reporting numeric accuracy as a **curve over k** and adding MRR. A
single k conflates "not indexed" with "ranked too low" — opposite fixes.

A per-question view (`src/compare_runs.py`) then showed the identical aggregates
were a *coincidence*: run C won both split questions and lost two others. Only
the per-question view distinguishes "no effect" from "effects that cancel".

### A verifier written for one purpose caught a different problem

`src/verify_writeup.py` asserts every figure quoted in this document against
`reports/*.json`, so the writeup cannot drift from its evidence. It was written
to catch careless transcription. It failed on **10 figures** the first time.

The cause was not a typo. Re-executing the notebook re-ran every evaluation and
overwrote the scorecards — and the new run genuinely differed on several metrics
at `temperature=0.0`.

Running each configuration 3x showed the picture is more specific than "the
results are noisy", and the specificity matters:

| Metric | Spread (linear) | Spread (graph) |
|---|---|---|
| numeric correct | **0.0** | **0.0** |
| both sides shown | **0.0** | **0.0** |
| exit correct | **10.5** | **5.3** |
| hallucination rate | **5.3** | 0.0 |

**Retrieval-driven metrics are fully deterministic; generation-driven ones are
not.** Whether the right figure is found and stated does not vary. Whether the
model classifies a borderline question as ANSWER or CLARIFY does, by up to 10
points.

My first instinct was to caveat *every* comparison as noisy. That would have been
wrong in the other direction — it would have discarded the +12.5 point numeric
gain, which is real and perfectly stable. The fix was to measure which metrics
vary rather than to hedge uniformly.

A verifier that only confirms what you already believe is not worth writing.

### The hallucination detector had a false positive

The same re-run reported a 5.3% hallucination rate, which would have been the
worst result in the project. It was the detector: the flagged "figure" was
**`2027`** — a *year*, in an answer reasoning about calendar versus fiscal years.
The regex matched any run of four or more digits.

Now matches comma-grouped figures and currency-marked per-share amounts only.
The corrected result is still 0%, but it is measured correctly rather than
correct by luck — which matters, because that number is the strongest single
claim here.

### The exit classifier read keywords instead of the model's declaration

The graph's first scorecard showed numeric accuracy **up** 18.8 and exit
correctness **down** 21.1 — answers finding the right figure and then being
labelled wrong.

The classifier was keyword-matching. An answer opening `"CLARIFY — the question
is ambiguous"` was classified REFUSE because it later noted one reading "cannot
be determined from these passages". An aside outvoted the declaration.

Fixed by reading the model's own declared exit first, with one asymmetric
override: a declared REFUSE whose body states figures *with citations* is treated
as an ANSWER. `src/test_classify.py` pins six cases taken verbatim from real
answers — and one of them initially "failed" because the *test* was wrong, which
is the argument for testing against recorded output rather than guesses.

### Cleaner bugs found only by inspecting output

- **Zero `<p>` tags.** NVIDIA's filings wrap body text in `<div>`/`<span>`, with
  headings marked only by `font-weight:700` in inline CSS. The first cleaner
  produced 600 tables and **0 prose blocks**.
- **The hidden `<ix:header>`** holds ~19,000 characters of XBRL machine metadata
  (`iso4217:USD`, `P1Y`, fasb.org URIs). Invisible to a reader, fully
  extractable, and it became the single largest chunk in the corpus — retrievable
  for any query.
- **`fiscal_year` from prose is actively wrong.** The phrase `for the fiscal year
  ended` matches the *comparative prior year* before the cover page in all three
  filings. It comes from inline XBRL (`dei:DocumentFiscalYearFocus`) instead.

### A silent fallback that produced plausible-looking output

`split_fixed()` falls back to a hand-written splitter when
`langchain_text_splitters` is unavailable. Every chunking run before indexing
used that fallback, because nothing was installed yet. Installing the real
dependency changed the fixed arm's prose median from 918 to 763 chars.

A silent fallback that produces plausible output is a good way to measure the
wrong thing for a week.

### The reranker: built last, and the result argued against it

This was nearly skipped. By the time it came up, hybrid retrieval had already
fixed the split trap dense failed, and metadata filtering had already beaten both
for ~13ms. Adding a large dependency to re-rank candidates BM25 already ranks
first looked hard to justify.

Building it anyway produced a better finding than skipping it would have.

**It works.** Run F improves on hybrid by 6.2 points of numeric accuracy, and
fixes temporal (67% → 100%) and restatement (50% → 75%). A prediction I made
before running it — that a cross-encoder would demote numeric tables in favour of
prose, because a markdown table does not *read* like an answer — was **wrong**.
Numeric accuracy stayed at 100%.

**And it is dominated.** Run F lands on *exactly* the same 81.2% as run D,
metadata filtering, which gets there in 245ms against F's 7,257ms. Same accuracy,
30x the latency.

**Combining them is worse than either alone.** Run G scores 75.0% — below both
components — and halves split-trap accuracy. The mechanism is specific:
filtering narrows the candidate pool to one fiscal year, so the reranker can no
longer surface the *other* filing's version of a restated figure. The two
techniques work against each other on precisely the questions this corpus exists
to test.

The conclusion is that reranking is not the right answer for this corpus — not
because it fails, but because a metadata filter extracted at clean time matches
it for 3% of the cost. **Skipping it would have reached the right conclusion for
the wrong reason, and been unreportable either way.**

---

## 7. Learnings

**Measure before you optimise, and make the measurement fail loudly.** Three of
the most important findings here came from a metric behaving impossibly — five
identical scores, accuracy rising while exit correctness fell, and a writeup
verifier failing on its own committed evidence. All three were bugs in the
measurement, not the system. A metric that can only look plausible is worse than
no metric.

**Check your numbers against their source, mechanically.** The verifier that
compares this document to `reports/*.json` took twenty minutes to write and
caught two things no amount of re-reading would have: that the results were
non-deterministic, and — on its last run — that the "zero hallucinations" claim
I had repeated in three documents was false.

**An LLM judge and a mechanical check disagree, and the mechanical one wins.**
The faithfulness judge scored the one hallucinated answer **1.00**: the
arithmetic genuinely followed from the passages, so by its own criterion the
answer was grounded. A regex asking "does this figure appear in any retrieved
passage?" caught what the judge could not. Neither check subsumes the other,
and a system reporting only the judge's score would have reported perfect
faithfulness alongside a fabricated number.

**`temperature=0.0` does not mean reproducible — but measure *which* metrics
vary before you caveat them.** Exit classification swung 10.5 points between
identical runs, while numeric accuracy had *zero* spread across six runs. The
useful move was not "treat everything as noisy", which would have thrown away a
real 12.5-point result; it was running the eval repeatedly and reporting the
spread per metric. Retrieval-driven metrics here are deterministic;
generation-driven ones are not.

**Dense retrieval is the wrong tool for exact figures.** This was the most
actionable finding. `1.21` is a token, not a meaning, and embeddings encode
meaning. On the split-trap category dense scored 0% and BM25 scored 100%, at 1/80
of the latency. Any financial RAG system that uses embeddings alone will fail
this class of question silently.

**Metadata beats model quality for period disambiguation.** Year precision rose
70.6% → 90.9% by filtering on `years_present` — a field extracted from table
column headers at clean time. No better embedding would have achieved that,
because both years are described in near-identical language.

**A label is not a key.** `Other`, `Total` and `Taiwan` each appear in many
unrelated tables. This is obvious in retrospect and cost a false headline finding
to learn. `table_caption` earns its place in the metadata schema.

**Refusal is a feature, and it is measurable.** The system refused rather than
inventing in every case where retrieval came up short — 0% hallucination across
19 questions. Isolating that (`src/debug_gen.py` runs the same question on
retrieved passages and on oracle passages) proved every failure was retrieval
depth, not fabrication. That distinction determines what you fix next.

**Build the loop last.** Every number in section 5 was produced by a linear path.
Had the graph existed first, the chunking comparison would have been confounded
by retries firing unpredictably, and none of the retrieval findings would be
defensible.

**Build the thing you have already argued against.** The reranker was nearly
skipped on the reasoning that the measurements had made it redundant. That
reasoning turned out to be *correct* — and worthless, because "we assumed it
would not help" is not a finding. Building it produced a measured result (same
accuracy as metadata filtering, 30x the latency) and an unexpected one (stacking
it with filtering is worse than either alone, because filtering starves the
reranker of the second filing it needs to surface a restated figure). A cheap
experiment that confirms your prior is still worth running when the alternative
is an unsupported assertion in the writeup.

### What I would do differently

- **Pin the latency ceiling first.** The brief calls it a first-class constraint.
  Without it, the central graph-vs-linear tradeoff cannot be resolved — only
  reported.
- **More questions per category.** At n=16, one question is 6.25 points. Three to
  four per category is the minimum for a per-category score to mean anything;
  eight would make the differences between arms statistically meaningful rather
  than directional.
- **Test the scorer against recorded output from the start.** Both scorer bugs
  would have been caught on day one by a handful of fixture cases.

---

## 8. Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env      # NEBIUS_API_KEY + SEC_USER_AGENT
```

Open `notebooks/main.ipynb` and run top to bottom — 63 cells, 29 of them code,
all executing cleanly against live APIs. Or from the command line, see the
`Setup` section of `README.md`.

`data/raw/` is gitignored; `src/fetch.py` reproduces the corpus byte-for-byte
(verified by SHA-256 against manually downloaded copies).

**Repository layout, decision log (D1–D19), and per-run scorecards:**
`README.md`, `DECISIONS.md`, `reports/*.json`.
