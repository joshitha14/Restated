# Decisions

Running log. Newest last. Each entry: what was decided, and why.

---

## D1 — Project name: `restated`

Names the central finding rather than the apparatus. `chunklab` was the
alternative; rejected because the measurement harness is the means, not the
result. Deliberately not `finrag`.

## D2 — `fiscal_year` comes from inline XBRL, not prose

The handoff specified tagging `fiscal_year` from the period of report rather
than the filing date. Verified why that matters, and found a better source.

Regex over prose is **actively wrong** here. The phrase `for the fiscal year
ended` matches the *comparative prior year* before it matches the cover page,
in all three filings:

| File | Naive prose match | Actual FY |
|---|---|---|
| `nvda-20260125.html` | January 26, 2025 | **2026** |
| `nvda-20250126.html` | January 28, 2024 | **2025** |
| `nvda-20240128.html` | January 29, 2023 | **2024** |

Every filing carries inline XBRL DEI facts. `dei:DocumentFiscalYearFocus`
gives `2026` / `2025` / `2024` exactly, with no inference. `EntityCentralIndexKey`
(`0001045810`), `DocumentType` (`10-K`) and `TradingSymbol` (`NVDA`) confirm
alongside it.

**Decision:** parse `fiscal_year`, `cik`, `form_type` and `period_end` from the
XBRL DEI facts. Never from prose, never from the filing date.

## D3 — A line-item label is not a unique key; `table_caption` joins the schema

First pass of the restatement diagnostic keyed on `(label, fiscal_year)` and
produced 88 hits, most of them false. `Other`, `Total` and `Balance at end of
period` each appear in many unrelated tables, so `Total` FY2024 collided
between the revenue table (`60,922`) and an unrelated one (`3,549`).

**Decision:** add `table_caption` to the metadata schema from §4. Without it,
citations conflate tables and numeric-accuracy scoring is unreliable — a
retrieved chunk can be "correct" for the wrong table.

## D4 — The restatement trap fires, but not on the revenue-by-end-market table

The handoff proposed revenue as the restatement case. **It is stable:** Data
Center FY2025 is `115,186` in both the FY2026 and FY2025 filings; FY2024 is
`47,525` in both. Had the eval set been written against revenue, the
restatement category would have measured nothing.

Where it *does* fire — geographic revenue disaggregation:

| Line item (FY2025) | Per FY2026 filing | Per FY2025 filing | Ratio |
|---|---|---|---|
| United States | 77,482 | 61,257 | 1.26x |
| China (incl. Hong Kong) | 25,048 | 17,108 | 1.46x |
| Other | 4,367 | 7,875 | 0.55x |

**The cause is stated in the table captions themselves:**

- FY2025 filing: "Revenue by geographic area is based upon the **billing
  location of the customer**"
- FY2026 filing: "Revenue by geographic area is based upon the **location of the
  customers' headquarters**"

US and China rise while "Other" falls — exactly what a billing-location →
HQ-location reattribution looks like. A documented basis change across three
consistent line items is a stronger finding than an unexplained single-line
jump, because the writeup can name the mechanism.

### Correction: the Taiwan 13.9x result was an artifact

An earlier version of this entry reported Taiwan FY2025 as `1,481` vs `20,573`
and called it the strongest case in the corpus. **It was a label collision — the
D3 failure reproducing itself inside the diagnostic.** The `1,481` comes from a
*long-lived assets* table in the FY2026 filing; the `20,573` from the *revenue*
table in the FY2025 filing. Both rows are labelled `Taiwan`.

The scan keyed on best-effort captions extracted from raw HTML, and at that
point many captions were empty, so the guard did not bite. Once `clean.py`
produced reliable captions, `src/find_restatements2.py` re-keyed on a coarse
table *topic* and Taiwan disappeared from the results.

Two lessons, both worth reporting:

1. A diagnostic needs the same disambiguation discipline as the pipeline it is
   checking. This one did not have it and produced a headline finding that was
   false.
2. It is concrete evidence for why `table_caption` is load-bearing (D3) rather
   than nice-to-have: without it, a retrieval system asked "revenue attributed
   to Taiwan" would happily return a long-lived-assets figure.

Other usable reclassifications:

- `Total`, Operating Income by Reportable Segments, FY2025: `87,960` vs `81,453`
- `Patents and licensed technology`, intangible assets, FY2025: `230` vs `449`
- `Total intangible assets`, FY2024: `807` vs `3,091`
- `Other assets`, balance sheet, FY2025: `3,038` vs `6,425`

**Decision:** anchor the restatement-sensitive eval questions on geographic
revenue (United States and China, where the basis change is documented), segment
operating income, and intangible assets. Not revenue by end market, and not
Taiwan.

## D5 — The 10-for-1 stock split is a second trap, not noise

Not anticipated in the handoff. NVIDIA split 10-for-1 in June 2024, so the
FY2024 filing states per-share and share-count figures **pre-split**, while
FY2025 and FY2026 restate them post-split:

| Line item (FY2024) | Per FY2025/26 filings | Per FY2024 filing |
|---|---|---|
| Basic EPS | 1.21 | 12.05 |
| Diluted EPS | 1.19 | 11.93 |
| Basic weighted average shares | 24,690 | 2,469 |
| Diluted weighted average shares | 24,940 | 2,494 |

18 such line items. Same figure, same fiscal year, both correct, exactly 10x
apart.

This is arguably a **cleaner eval category than reclassification**, because the
divergence is unambiguous (exactly 10x, no judgement about accounting basis)
and the correct answer depends entirely on which filing is cited. A system that
retrieves the FY2024 filing and answers "EPS was $12.05" is not wrong about the
number — it is wrong about the basis, and only a citation reveals that.

**Decision:** treat split-adjusted figures as a first-class query category
alongside restatement. Report them separately.

## D6 — Diagnostic confounders must be classified, not reported raw

The diagnostic now buckets every divergence as `RECLASSIFIED`, `SPLIT` or
`UNIT_MISMATCH`. The last is an extractor artifact: a "% of revenue" table
matched against a dollar table, producing e.g. `Cost of revenue` FY2024 as
`16,621` vs `27.3`. Those are different bases, not restatements.

**Decision:** keep the classifier in `src/find_restatements2.py`. Counts by
bucket go in the report; only `RECLASSIFIED` and `SPLIT` seed eval questions.
Current counts: RECLASSIFIED=27, SPLIT=18, UNIT_MISMATCH≈43.

## D7 — Table blocks cannot carry a scalar `fiscal_year`

The revenue-by-end-market table in the FY2026 filing contains FY2026, FY2025
**and** FY2024 columns. §4 requires tables be held whole by both chunking
strategies, so a table chunk inherently spans three fiscal years.

**Decision:** for `block_type == "table"`, `fiscal_year` is the filing's FY (the
authoritative source of the chunk) and a separate `years_present` list records
the columns. Metadata-filtered retrieval (the optional fourth arm in §4) must
filter on `years_present`, not `fiscal_year`, or every multi-year table is
wrongly excluded.

## D8 — Every block carries a `citation`, computed at clean time

Caption extraction is not reliable for every table: the FY2026 geographic
revenue table has no preceding descriptive text at all, and 9 tables across the
corpus still end up caption-less. A blank caption meant a blank citation, and a
blank citation cannot be scored for groundedness — which matters most exactly
where the restatement questions live, since the same figure is right or wrong
according to its source filing.

**Decision:** `make_citation()` degrades caption → years covered → section
alone, so no block can produce an empty citation:

```
FY2026, Item 7, "Operating Income by Reportable Segments"
FY2026, Item 8 (covering FY2026-FY2024)
FY2026, Item 7
```

Verified by `src/verify_citations.py`: 0 blank citations across 2,040 blocks.

Improving `caption_before()` to prefer lead-in sentences and fall back to
title-cased headings took FY2026 from 0 usable captions to 56 of 56 — and that
is what exposed the false Taiwan finding in D4.

## D9 — The inline-XBRL header must be stripped, or it becomes the largest chunk

Found by inspecting chunk-size outliers, not predicted. Every inline-XBRL filing
carries a hidden `<ix:header>` holding machine-readable fact context:

```
0001045810 2026 FY false 362 460 P1Y P2Y P3Y
http://fasb.org/us-gaap/2025#AccruedLiabilitiesCurrent iso4217:USD xbrli:shares
```

It is invisible to a human reader but ~19,000 characters of extractable text, so
the naive prose pass emitted it as a single enormous "paragraph" — the largest
chunk in the corpus, semantically meaningless, and retrievable for any query.

**Decision:** `strip_xbrl_header()` removes it, and `is_machine_noise()` guards
the prose path against namespace URIs, ISO duration codes and low word-density
text generally. Ordering is load-bearing: `read_meta()` reads the DEI facts out
of that header (D2) and must run BEFORE the strip.

Effect: semantic max prose chunk 19,834 → 1,000 chars.

## D10 — Oversized single paragraphs split on sentence boundaries

Some Item 1A risk-factor paragraphs exceed 3,000 characters on their own. The
semantic arm cannot split mid-word without abandoning its premise, but emitting
a 3.7k chunk against the fixed arm's 1.0k would confound the comparison with
chunk size — the experiment would measure size, not boundary placement.

**Decision:** `split_long_paragraph()` splits only on sentence boundaries,
using a lookahead that avoids breaking on `U.S.`, `Inc.` or `$1.5 billion`.

Result: median prose 918 (fixed) vs 777 (semantic), a 1.18x ratio. Close enough
that boundary placement, not size, is what differs.

## D11 — The chunking control is asserted in code, not assumed

§4 requires tables be kept whole by BOTH strategies. That is the single
assumption the headline result rests on, so it is tested rather than trusted.
`src/verify_chunks.py` asserts five controls:

1. Table chunks are **byte-identical** across strategies (not merely equal in count)
2. No table lost its header separator row
3. Prose median sizes within 1.5x
4. All 19 eval-critical figures present in both strategies
5. No blank citations, sections, fiscal years, or duplicate chunk IDs

All pass. Current totals: fixed 1,435 chunks, semantic 1,421.

**Decision:** `verify_chunks.py` runs before any indexing. If a control fails,
run A vs run B measures something other than chunking strategy and the result is
void — better to know before spending embedding credits.

## D12 — The corpus is verified reproducible; file extensions are resolved at runtime

`fetch.py` was written before it was ever run, so the handoff's accession numbers
were unverified. Running it against live EDGAR confirmed all of them:

| Fiscal year | Period end | Filed | Accession | Bytes |
|---|---|---|---|---|
| FY2026 | 2026-01-25 | 2026-02-25 | `0001045810-26-000021` | 1,967,816 |
| FY2025 | 2025-01-26 | 2025-02-26 | `0001045810-25-000023` | 2,067,520 |
| FY2024 | 2024-01-28 | 2024-02-21 | `0001045810-24-000029` | 2,085,566 |

The fetched files are **byte-identical** (SHA-256) to the manually downloaded
copies, so the corpus reproduces exactly from a clean clone — which is what §8
requires, since `data/raw/` is gitignored and only the fetch script is committed.

The filing dates also confirm D2 concretely: each is ~1 month after its period
end (2026-02-25 vs 2026-01-25). Tagging `fiscal_year` from the filing date would
shift every chunk into the wrong year.

**A silent failure found in the process.** EDGAR serves `.htm`; the manual copies
were `.html`. `clean.py` globbed `nvda-*.html` only, so a grader cloning the repo
and running `fetch.py` would have produced `.htm` files that `clean.py` silently
ignored — the worst kind of failure, since it looks like an empty corpus rather
than an error.

**Decision:** every script resolves the extension at runtime, accepting `.htm`
or `.html`, and de-duplicates by period-end date so a directory holding both
copies is not processed twice. The `.html` duplicates were deleted after hash
verification; `fetch.py` output is now the single source.

Verified by a clean-room rebuild: `data/clean/` and `data/chunks/` deleted, then
every stage re-run from `data/raw/` alone. All verifiers pass, and chunk counts
are identical to the pre-wipe run.

## D13 — Embeddings: `Qwen/Qwen3-Embedding-8B` on Nebius (the only option)

Nebius Token Factory is OpenAI-compatible, so `langchain_openai` reaches it by
changing `base_url` to `https://api.tokenfactory.nebius.com/v1/`.

Querying the account's catalogue settled the choice: **`Qwen/Qwen3-Embedding-8B`
is the only embedding model available**. 23 chat models are exposed, which
matters later for generation.

**A required non-obvious flag.** `OpenAIEmbeddings` tokenizes locally with
tiktoken and sends integer arrays, which OpenAI accepts but Nebius rejects:

```
400 - {'detail': 'Tokenized input is not supported'}
```

`check_embedding_ctx_length=False` sends raw strings instead. The trade-off is
that LangChain no longer splits over-long inputs, so over-long text would be
**silently truncated server-side** — a chunk embedding as a prefix of itself,
with no error. `MAX_CHARS = 24_000` guards against it. The largest chunk in this
corpus is ~3,600 chars (~900 tokens) against a 32k context, so the guard exists
for a future corpus change rather than today's data.

Cost and throughput: 2,871 vectors in 184s (~35/s at peak, batch size 64).

## D14 — Prose chunks carry `years_present` too

Found by running the first real retrieval rather than by reasoning. Asked *"What
was NVIDIA's Data Center revenue in fiscal 2026?"*, the baseline returned **three
prose chunks and zero tables**, top hit a boilerplate passage about regulatory
risk, distance 0.451.

The cause was visible in the metadata: prose chunks had `years_present = []`.
Table chunks got years from their column headers (D7), but prose had **no year
metadata at all** — so dense similarity alone had to separate FY2026 language
from FY2024 language, which is exactly the period-confusion problem this corpus
was chosen to expose. It also left the metadata-filtered arm (run D) with
nothing to filter on for prose.

**Decision:** `prose_years()` assigns each prose chunk:

1. years named in the text (`"fiscal years 2026 and 2025"`), bounded to
   FY-4..FY so stray 4-digit numbers cannot leak in, plus
2. the filing's own fiscal year, always — a chunk from the FY2026 filing is
   FY2026 context even when it names no year

Applied identically to both strategies, so the A/B control is unaffected
(`verify_chunks.py` still passes).

Effect on the same query: top hit becomes *"Data Center revenue for fiscal year
2026 was up 68% from a year ago"*, distance 0.451 → **0.230**.

This is a metadata fix, not retrieval tuning, and it was made **before** any
measurement run. Period confusion is still visible at rank 3 (an FY2024 chunk
tagged `2024,2023`) — dense retrieval cannot reject it, which is precisely what
run D is meant to test.

## D15 — The fixed arm was silently using a fallback splitter

`split_fixed()` falls back to a hand-written recursive splitter when
`langchain_text_splitters` is unavailable, so ingest is runnable before
dependencies are installed. Every chunking run before indexing used that
fallback, because nothing was installed yet.

Installing the real dependency changed the fixed arm's prose median from 918 to
763 chars. The real `RecursiveCharacterTextSplitter` produces tighter, more
uniform pieces (988/990/991 on a test string vs the fallback's 988/1139/870 —
the fallback appended overlap *after* splitting and so overshot the budget).

**Decision:** keep the fallback for portability, but treat the real splitter as
authoritative. The fixed arm is now the genuine stock-LangChain baseline the
writeup claims it is, and the two arms' median sizes are closer (763 vs 760),
which *strengthens* the control by reducing the size confound.

Worth reporting as a methodological note: a silent fallback that produces
plausible-looking output is a good way to measure the wrong thing for a week.

## D16 — Two scorer bugs that made every run score identically

The first scorecard reported **37.5% numeric accuracy for all five runs**, to one
decimal place. Five genuinely different configurations cannot coincide that
exactly; the identical number was a symptom, not a result.

**Bug 1 — three questions were unscoreable.** `expected_figures()` handled ints
and dicts but not prose, so questions whose expected value reads
`"193737 - 115186 = 78551 (68.2% growth)"` yielded *no* figures at all.
`q_temporal_01`, `q_trend_01` and `q_trend_03` could never register a hit
regardless of what retrieval returned. A metric that cannot be satisfied is worse
than no metric: it depresses every run by the same amount, which looks like a
valid comparison.

**Bug 2 — k=5 was too shallow to measure anything.** Searching to depth 50 showed
the answers were indexed all along, just ranked below the cutoff:

| Question | dense | BM25 | hybrid |
|---|---|---|---|
| q_numeric_01 (`193,737`) | 12 | 20 | 12 |
| q_split_01 (EPS `1.21`) | **not in top 50** | **1** | 5 |
| q_split_02 (shares) | 35 | **1** | 2 |
| q_restate_01 (US revenue) | **not in top 50** | **17** | 32 |

**Decision:** report numeric accuracy as a **curve over k**, not a single number.
A single k conflates "the answer is not indexed" with "it ranked below the
cutoff", and those call for opposite fixes. Added `numeric_rank` and MRR so a
configuration that puts the answer first is distinguishable from one that buries
it at rank 5 — binary top-k accuracy treats those as equal, though only the first
is usable by a generator with a limited context budget.

Also added `src/compare_runs.py`: a per-question view. Identical aggregates can
mean the runs behave identically **or** that they win and lose on different
questions and the totals coincide. Here it was the latter — run C won both split
questions and lost two others. Only the per-question view distinguishes them.

## D17 — Results: dense retrieval cannot find exact figures

With the scorer fixed, at k=20:

| Run | Chunking | Retrieval | numeric | MRR | yr_prec | both_sides | p50 |
|---|---|---|---|---|---|---|---|
| A | fixed | dense | 68.8% | 39.9 | 70.6% | 33% | 200ms |
| B | SemanticChunker | dense | 75.0% | 41.5 | 72.8% | 33% | 197ms |
| B2 | structural | dense | 68.8% | 40.0 | 72.2% | 33% | 202ms |
| C | fixed | hybrid | 75.0% | 33.4 | 67.8% | **67%** | 203ms |
| D | fixed | filtered | **81.2%** | **41.5** | **90.9%** | 33% | 213ms |
| E | fixed | BM25 only | 68.8% | 25.3 | 61.6% | **83%** | **2.3ms** |

**The headline finding: chunking strategy matters less than retrieval method.**
Fixed vs SemanticChunker vs structural spans 68.8–75.0%. Swapping dense for
metadata-filtered moves the same corpus to 81.2%.

**Dense retrieval fails the split trap completely.** EPS `1.21` is not in dense's
top 50, while BM25 ranks it **first**. Embeddings represent meaning; `1.21` is a
token, not a meaning. Per-category numeric accuracy on `split`:

- dense (A, B, B2): **0%**
- hybrid (C), BM25 (E): **100%**

**BM25 alone surfaces both sides of a restatement 83% of the time** versus 33%
for dense — and does it in **2.3ms against dense's 200ms**, because it makes no
API call. For a question whose whole point is that two filings disagree, lexical
matching on the figures themselves beats semantic similarity outright.

**Metadata filtering buys year precision cheaply.** Run D raises year precision
70.6% → 90.9% for 13ms, which is the period-confusion problem the corpus was
chosen to expose being solved by metadata rather than by a better embedding.

Caveat worth stating in the writeup: n=16 scoreable questions, so a one-question
swing is 6.25 percentage points. These are directional findings, not tight
measurements, and the per-category table matters more than the aggregate.

## D18 — The exit classifier read keywords instead of the model's own declaration

The graph's first scorecard showed a contradiction: numeric accuracy **up** 18.8
points, exit correctness **down** 21.1. Answers were finding the right figure and
then being labelled wrong, which is a classification problem rather than a
retrieval one.

Three concrete misreads, all from real answers in
`reports/graph_scorecard.json`:

- An answer opening `"CLARIFY — the question is ambiguous"` was classified
  REFUSE, because it later noted that one reading "cannot be determined from
  these passages". An aside outvoted the declaration.
- A complete answer giving both restated figures *and* the arithmetic
  (`$82,875 + $5,085 = $87,960`) was classified REFUSE.
- An answer stating both split-adjusted figures correctly opened with the word
  REFUSE and then answered anyway.

**Decision:** read the model's own declared exit first (`ANSWER — …` /
`CLARIFY — …` / `REFUSE — …`), falling back to keywords only when nothing is
declared. One asymmetric override: a declared REFUSE whose body states figures
*with citations* is treated as an ANSWER — trusting the body over the label, but
only in that direction, since the reverse (a declared ANSWER with no figures) is
a real failure worth catching.

Also settled a definition that had been implicit: **stating two restated figures
is an ANSWER, not a CLARIFY.** The restatement rule asks for both values *within*
an answer. CLARIFY is reserved for a question ambiguous as asked — a calendar
year mapping to two possible fiscal years.

`src/test_classify.py` pins all six cases, taken verbatim from real answers
rather than invented. One of those cases initially "failed" because the *test*
was wrong, not the code — which is the argument for testing against recorded
output rather than guesses.

## D19 — The LangGraph loop: what it bought and what it cost

Built last, deliberately. Measured against the linear path rather than assumed
to help.

| Metric | Linear | Graph | Delta |
|---|---|---|---|
| exit correct | 89.5% | 84.2% | **−5.3** |
| numeric correct | 62.5% | **81.2%** | **+18.8** |
| both sides shown | 50.0% | **66.7%** | **+16.7** |
| faithfulness | 1.00 | 1.00 | 0.0 |
| hallucination rate | 0.0% | 0.0% | 0.0 |
| p50 latency | 3,301ms | 6,884ms | **+3,583** |
| p95 latency | 8,984ms | 17,263ms | +8,279 |

**The loop is a real quality win at roughly 2x latency.** Numeric accuracy
62.5% → 81.2% comes almost entirely from the grade-and-retry node: 8 of 19
questions triggered one rewrite, and the escalation from k=10 to k=20 is exactly
the gap D16 identified. Restatement numeric accuracy went 25% → 75%.

**Whether that trade is worth taking depends on the latency ceiling** — which
the original brief lists as a first-class constraint and which is still
unfilled. At a 5-second budget the graph does not qualify; at 20 seconds it
clearly does. Reporting both numbers rather than picking one is the honest move.

**The verify node never fired** (`refused_by_verify = 0`). Across 19 questions
the generator never produced an unsourced figure or an invalid citation, so the
safety net caught nothing. That is a good result for the system and an honest
finding about the node: on this corpus it is insurance, not a contributor. It
would matter more with a weaker generator or a noisier corpus.

**Remaining weak spot: restatement exit accuracy, 25%.** The figures are found
(75% numeric) but the exit is often CLARIFY where the eval set expects ANSWER.
That is arguably the eval set being stricter than it needs to be — a CLARIFY
that states both figures with sources is useful behaviour — but the definition
was fixed before the results were seen, so it stands as scored.

## D20 — `temperature=0.0` is not determinism, and a single run is not a measurement

Found by a check written for a different purpose. `src/verify_writeup.py`
asserts every figure quoted in `WRITEUP.md` against `reports/*.json`, so the
writeup cannot drift from its evidence. It **failed on 10 figures** the first
time it ran.

The cause was not a typo. Re-executing the notebook re-ran every evaluation and
overwrote the scorecards, and the new run genuinely differed at
`temperature=0.0`.

**Decision:** `src/evaluate_repeat.py` runs each configuration N times and
reports mean, min and max per metric. Results over 3 runs each
(`reports/variance.json`):

| Metric | Spread (linear) | Spread (graph) |
|---|---|---|
| numeric correct | **0.0** | **0.0** |
| both sides shown | **0.0** | **0.0** |
| exit correct | **10.5** | **5.3** |
| hallucination rate | **5.3** | 0.0 |
| p50 latency | 21ms | 450ms |

**The specificity is the finding.** Retrieval-driven metrics are fully
deterministic: whether the right figure is found and stated does not vary at all
across six runs. Generation-driven metrics are not: the three-way
ANSWER/CLARIFY/REFUSE choice swings up to 10.5 points, because the model
sometimes answers a borderline question and sometimes clarifies it.

**A correction worth recording.** The first response to this discovery was to
caveat *every* comparison as noisy. That would have been wrong in the opposite
direction — it would have discarded the graph's +12.5 point numeric gain, which
is real and perfectly stable. Hedging uniformly is not the same as measuring.
The graph's apparent +3.6 exit-correctness advantage *is* within noise and is no
longer claimed.

`verify_writeup.py` no longer pins single-run graph metrics — only the
deterministic parts (question count, rewrites fired) and the variance summary.
Pinning values that legitimately vary would make the check fail on every rerun,
which trains you to ignore it.

Worth reporting as a methodological note: the check was written to catch a
careless transcription and instead caught a flaw in how the results were being
presented. A verifier that only confirms what you already believe is not worth
writing.

## D21 — The hallucination detector had a false positive

The same re-run reported `hallucination_rate: 5.3%`, which would have been the
single worst result in the project — an answer stating a figure with no source.

It was the detector, not the model. The flagged "figure" was **`2027`**, a *year*
appearing in an answer reasoning about calendar versus fiscal years. The regex
matched any run of four or more digits.

**Decision:** match comma-grouped figures (`193,737`) and currency-marked
per-share amounts (`$12.05`) only. Financial figures in these filings are always
comma-grouped above 999, and per-share amounts always carry a currency marker.
Years, section numbers and bare percentages are not figures the system claims as
answers.

The corrected result is still 0% hallucination — but it is now *measured*
correctly rather than correct by luck, which matters because that number is the
strongest single claim in the writeup.

## D22 — The reranker: built against my own recommendation, and the result justified building it

Reranking was nearly skipped. The argument against it was reasonable: hybrid had
already fixed the split trap dense failed (0% → 100%), metadata filtering had
already beaten both on numeric accuracy for ~13ms, and BM25 already ranked the
split-trap figures *first*. There appeared to be no headroom.

**Decision:** build it anyway, as runs F and G, scored against C and D rather
than against the dense baseline. Beating A would have proved nothing — A was
already known to be the weakest arm.

`BAAI/bge-reranker-base` via sentence-transformers, run locally so the added
latency is CPU time rather than network time, which is what makes it stable
enough to report.

| Run | Retrieval | numeric | yr_prec | both sides | p50 |
|---|---|---|---|---|---|
| C | hybrid | 75.0% | 67.8% | 67% | **218ms** |
| D | filtered | **81.2%** | **90.9%** | 33% | **245ms** |
| F | + rerank over hybrid | **81.2%** | 74.7% | 67% | 7,257ms |
| G | + rerank over filtered | 75.0% | **92.2%** | 50% | 9,492ms |

**Three findings, none of which were available without building it.**

1. **It works, and my specific prediction was wrong.** I expected the
   cross-encoder to demote numeric tables in favour of prose — a markdown table
   of figures does not *read* like an answer to "what was basic EPS". The demo
   appeared to show exactly that. Across the full eval set it did not happen:
   numeric accuracy held at 100%, and temporal went 67% → 100%, restatement
   50% → 75%.

2. **It is dominated.** Run F reaches *exactly* the same 81.2% as run D at 30x
   the latency. Two techniques, same accuracy, one effectively free.

3. **Stacking them is worse than either alone.** Run G scores 75.0%, below both
   components, and halves split-trap accuracy (100% → 50%). The mechanism is
   specific and was not anticipated: **filtering narrows the candidate pool to
   one fiscal year, so the reranker can no longer surface the *other* filing's
   version of a restated figure.** The two techniques work against each other on
   precisely the questions this corpus exists to test.

**The methodological point.** The decision to skip would have reached the right
conclusion for the wrong reason, and been unreportable either way. "We assumed it
would not help" is not a finding; "it matches a free technique at 30x the cost,
and combining them is actively harmful" is. A cheap experiment that confirms a
prior is still worth running when the alternative is an unsupported assertion.

**Operational note.** The cross-encoder loads a ~1.1GB model lazily, so
`run_config()` issues a throwaway warmup query before timing starts — otherwise
the first question absorbs several seconds of model loading and the reported p50
measures the wrong thing.

## D23 — A Chroma defect, a workaround, and an unexplained batch instability

Two related problems surfaced when all eight runs were scored in one process.
Both are recorded here because one was fixed and one was not, and the difference
matters for how the numbers should be read.

### The defect (reproduced, worked around)

```
chromadb.errors.InternalError: Error executing plan: Internal error: Error finding id
```

Raised from Chroma 1.5.9's Rust query layer on **metadata-filtered** queries,
once several collections have been opened in the same process. Runs D and G — the
only two that use a `where` clause — vanished from the combined scorecard while
the other six completed.

**The failure mode is what made it dangerous.** It did not crash the run. The
evaluator caught the exception per-run and continued, so the output looked like a
valid eight-way comparison with two arms quietly missing.

Four hypotheses were tested and rejected before finding a workaround:

| Hypothesis | Test | Result |
|---|---|---|
| Interleaved access across collections | one client per collection | fails identically |
| The collection or its size | first filtered query per collection | all three succeed |
| `.get(where=...)` poisons the client | control vs. poisoned sequence | control passes |
| Number of client instances | 12 clients, filtered query each | no failure |

**Decision:** stop characterising the library bug and route around it.
`FilteredRetriever` now retrieves `k * 4` unfiltered and applies the year
predicate in Python. Same predicate, same ranking, one extra round of scoring,
and no dependency on a library path that fails unpredictably. The reproduction
case is preserved in the diagnostics history.

**Also changed:** the evaluator now prints a full traceback on a failed run, not
just the exception message. A bare message is what allowed two runs to disappear
without explanation.

### The instability (bounded, not explained)

Separately: running several configurations in one process yields different scores
than running each alone.

| Run | Isolated (3x, identical) | In an A–G batch |
|---|---|---|
| C (hybrid) | 75.0% | 81.2% |
| D (filtered) | 75.0% | 81.2% |

Bisected as far as usefulness allowed: `B2,C` gives 75.0%, `A,B,B2,C` gives
81.2%. All three preceding runs are needed to shift it. Each configuration is
perfectly deterministic in isolation — three consecutive runs of C returned
75.0% every time.

**Decision:** report the **isolated** figures throughout. They are the
conservative choice (lower for C and D), they do not depend on execution order,
and no conclusion in the writeup changes under either set. The instability is
disclosed in WRITEUP.md §5 rather than silently resolved in whichever direction
flatters the result.

**Why stop here.** Five hypotheses had already been tested and rejected across
the two problems. The remaining question moves one cell in one table and changes
no conclusion. Continuing to chase it would have been thoroughness spent where it
does not pay — and knowing when to bound a defect rather than solve it is part of
the engineering, not a lapse in it.
