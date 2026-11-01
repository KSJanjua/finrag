# Decision log

One entry per implementation decision: the problem, the options, the choice and the evidence.
Rejected ideas are recorded as well, so the reasoning behind what is *not* in the code is kept.

---

## D1 · Grow the corpus over time, not across companies

- **Problem:** 3 reports and 82 pages is too small to show multi-period questions, conflicting
  figures, freshness or incremental indexing.
- **Options:**
  1. More Tesla quarters.
  2. Add another EV maker (Rivian).
  3. Both.
- **Choice:** 11 Tesla quarterly updates, Q4 2022 – Q2 2025 (all public PDFs on Tesla's IR site).
- **Why:** each Tesla report repeats the last five quarters in its tables, so the same number
  appears in several reports. That creates real multi-period questions, duplicate evidence and
  freshness problems.
- **Checked, not assumed:** I compared every (metric, quarter) value in the financial and
  operational summary tables of all 11 reports. 460 of 581 values appear in 2+ reports, and
  **none were revised**. So "conflicting evidence" here means approximate chart readings vs exact
  table values, not restated numbers, and duplicates (the same fact in up to 5 reports) are the
  bigger practical issue. Rivian publishes its shareholder letters as HTML on
  SEC EDGAR, not as PDFs, so adding it would mostly mean a second parser without a new retrieval
  problem.
- **Revisit if:** cross-company comparison becomes a goal; the document registry already has a
  `company` field for it.

---

## D2 · A document registry instead of hard-coded file names

- **Problem:** document names, labels and URLs were hard-coded in three modules
  (`config`, `retriever`, `evaluate`), each assuming three 2023 reports.
- **Choice:** `data/documents.json` lists every source PDF with company, fiscal year, quarter,
  label, URL and publication date (taken from the PDF's own metadata). Labels, download URLs,
  the period-to-report lookup and evaluation codes (`2023Q3`) are all derived from it.
- **Why:** adding a quarter becomes a one-line data change; only registered documents are
  ingested (in chronological order), so a stray PDF in the folder can't enter the index; and
  every chunk now carries `fiscal_year` and `quarter` as fields, which later filtering needs.

---

## D3 · One vision model for all chart descriptions: gemini-3.1-flash-lite

- **Problem:** 165 chart-candidate pages across 11 reports. The Flash models allow about
  20 requests per day on the free tier (a week of runs); the existing descriptions came from
  three different Flash versions.
- **Options:**
  1. Keep the existing descriptions and add new ones with a Flash model over several days.
  2. Re-describe every page with flash-lite.
- **Check before choosing:** on three chart pages whose values were verified against the page
  images (cost per vehicle bars, Model Y price map, revenue-growth lines), flash-lite read every
  checked value within chart-reading tolerance (e.g. ~$39,600 for a bar just above $39,500),
  same as the Flash descriptions.
- **Choice:** option 2. One model makes chart chunks consistent in style and quality across
  quarters, which matters for a fair evaluation, and the job fits in one day of quota.
- **Mechanism:** `describe_charts.py` skips only pages already described by the *current*
  model, so changing the model redoes everything and an interrupted run resumes.

---

## D4 · flash-lite as the default answering model; a different model as judge

- **Choice:** chat and vision default to `gemini-3.1-flash-lite`; the LLM judge defaults to
  `gemini-3.8-flash`.
- **Why:** flash-lite's free quota is the only one large enough for evaluation runs (76+
  answers per run). The judge is a separate model so no answer is graded by the model that wrote
  it (self-preference bias), and it is called rarely enough for the smaller quota.

---

## D5 · Build the evaluation set before adding features

- **Problem:** v1's 38 questions cover three reports and only single facts. They can't show
  whether a feature helps with multi-period questions, re-worded periods or unanswerable questions.
- **Choice:** a 116-question set (`data/eval/questions.json`) measured against the unchanged v1
  pipeline first. Every later feature is judged against that baseline.

| Type | Count | Source |
|---|---|---|
| text | 32 | 14 from v1 + 18 quoted from each report's highlights page |
| table | 37 | 13 from v1 + 24 generated from extracted summary-table cells |
| chart | 15 | 11 from v1 + 4 new, values checked against the rendered page images |
| paraphrase | 10 | table facts with re-worded periods ("second quarter of 2024", "2Q-2024", "Q1 FY2025") |
| multi | 12 | two periods per question, several more than a year apart (two reports needed) |
| unanswerable | 10 | facts verified absent by searching every chunk |

- **Checks on the set itself:** `validate_questions.py` requires every evidence set to have its
  evidence on a gold page. While building the unanswerable questions, a search showed Q2 2021
  figures *are* in the corpus (reconciliation tables reach back to 2020), so that question was
  dropped.

---

## D6 · Graded relevance and NDCG

- **Problem:** the same figure appears in up to five reports. Any copy answers the question, but
  the quarter's own report is the better citation.
- **Choice:** gold pages carry a grade: 2 for the period's own report, 1 for a later report
  repeating the figure. Metrics:
  - `hit@k`: any relevant chunk in the top k (v1's "Recall@k").
  - `recall@k`: share of a question's evidence sets found (context recall; only differs from
    hit@k for multi-period questions).
  - `precision@k`: share of the top k that is relevant.
  - `ndcg@k`: rewards ranking the own report above later copies; the only metric that uses grades.
  - `MRR`.
- **Why not page-level relevance:** a page can hold a footnote-only chunk next to the chart that
  answers the question; a relevant chunk must contain the evidence string.

---

## D7 · Number grounding as a deterministic faithfulness check

- **Problem:** fact matching says whether the right number is present, not whether the answer
  also states numbers its sources don't contain.
- **Choice:** every number in an answer must be supported by one of the sources it cites. A source
  value supports it if, rounded to the precision the answer uses, it equals the answer's number
  ("$2.5B" ⇐ 2,531 in a $M table; "$26.1B" is not supported by 25,500). Small integers and bare
  years are ignored.
- **Rejected:** a flat relative tolerance (first version, 2.5%). It accepted "$26.1B" for 25,500
  and would reject correct roundings of small values ("$0.4B" for 441 is 9% off).
- **Why not only an LLM judge:** the judge needs another free-tier model with a 20-request daily
  quota, and is non-deterministic; number grounding runs on every answer for free and is
  reproducible. The judge remains available (`--judge`).

---

## D8 · A rule-based query parser for reporting periods

- **Problem (measured):** the report filter only recognised "Q3 2024". Re-worded periods
  ("third quarter of 2024", "2Q-2024", "Q1 FY2025", "quarter three, 2024") searched all 11
  reports (Hit@5 0.40 on those questions); "full-year 2024" questions had no filter at all; and
  "Q1, Q2 and Q3 2024" was narrowed to Q3 only.
- **Options:**
  1. Ask the LLM to extract periods (query rewriting / self-query).
  2. A rule-based parser.
- **Choice:** rules (`src/finrag/query.py`). Quarter mentions in the common formats, lists of
  quarters sharing a year, and full-year mentions mapped to that year's Q4 report (where annual
  results are published).
- **Why rules here:** the set of period formats in financial questions is small and enumerable,
  so a parser covers it deterministically, at zero latency and zero API cost (an LLM call per
  question would use the free-tier quota the answers need), and every case is unit-tested. An
  LLM rewriter would be the next step for open-ended wording, not for periods.
- **Result:** Hit@5 0.783 → 0.849 (re-worded 0.40 → 0.70, text 0.78 → 0.91 through the
  full-year rule).

---

## D9 · Query decomposition by period

- **Problem (measured):** a question comparing two periods (e.g. Q4 2022 vs Q4 2024) needs facts
  from two reports more than a year apart; one search over all reports rarely returns both
  (multi-period Recall@5 0.54).
- **Choice:** when a question names several periods, run one filtered search per period's report
  and merge the result lists round-robin in the order the periods were named, dropping
  duplicates. Periods without a report of their own get an unfiltered search.
- **Why round-robin:** scores from different filtered searches aren't comparable, and the goal is
  coverage, so every period gets a slot near the top.
- **Result:** multi-period Recall@5 0.54 → 0.63, Hit@5 0.849 → 0.868.
- **What it revealed:** in the remaining misses the right report *was* searched, but its big
  summary table was outranked inside that report, which is the table-size problem (D10).

---

## D10 · Split tables by the embedding model's token count

- **Problem (measured):** 97 of 165 table chunks exceeded the embedding model's 512-token input
  (numbers split into many tokens, so a 1,500-character table is ~630 tokens). Dense retrieval
  never saw their lower rows, and in multi-period questions the right report's summary table was
  outranked by shorter chunks.
- **Choice:** `split_table` measures size with the embedding model's own tokenizer and splits a
  table by rows (header repeated) so each part is at most 450 tokens, leaving room for the chunk's
  label line.
- **Compared (production configuration, same questions):**

| Table parts | Chunks | Over 512 tokens | Hit@5 | MRR | text | table | multi (Recall@5) |
|---|---|---|---|---|---|---|---|
| ≤ 1,500 characters (before) | 813 | 97 | 0.868 | 0.681 | 0.906 | 0.865 | 0.625 |
| **≤ 450 tokens** | 901 | 0 | **0.896** | **0.692** | 0.906 | 0.892 | **0.750** |
| ≤ 200 tokens | 1,340 | 0 | 0.830 | 0.678 | 0.719 | 0.919 | 0.667 |

- **Why not smaller:** 200-token row groups help table questions slightly, but 692 table fragments
  crowd text chunks out of the top results (text 0.91 → 0.72). Split only as much as the model
  needs.

---

## D11 · Rejected: stemming or truncating words for BM25

- **Problem (measured):** "How many vehicles did Tesla deliver in Q4 2022?" BM25 ranks the
  operational summary table ~30th, because the question says "deliver" and the table says
  "deliveries" (same for "produce" / "production"). Dense search ranks it 1st; after fusion it
  lands at rank 6–7.
- **Why not Porter stemming:** it doesn't join these pairs ("deliver" → "deliv" but "deliveries"
  → "deliveri"; "produce" → "produc" but "production" → "product"). The mismatch is between a
  verb and a noun, not between word endings.
- **Compared (production configuration):**

| BM25 words | Hit@5 | MRR | text | table | fixed | broke |
|---|---|---|---|---|---|---|
| **unchanged** | **0.896** | 0.692 | 0.906 | 0.892 | | |
| plural "s" removed | 0.906 | 0.713 | 0.844 | 0.973 | tb01, tb04, tb05 | tx02, tx08 |
| first 5 characters | 0.896 | 0.723 | 0.812 | 0.973 | mp07, tb01, tb04, tb05 | mp09, tx01, tx02, tx08 |
| plural removed + first 7 | 0.896 | 0.729 | 0.812 | 0.973 | tb01, tb04, tb05 | tx01, tx02, tx08 |

- **Decision:** keep BM25 words unchanged. Every variant moves table questions into the top 5 and
  pushes nearly as many text questions out (rank 5 → 6): the best one gains one question net
  (Hit@5 +0.01), which is within the noise of a 106-question set, and the MRR gain of 2–4 points is
  paid for with text questions. The evidence for these
  questions is at ranks 6–8, which is a context-size question (D12).

---

## D12 · Rejected: de-duplicating near-identical chunks in the context

- **Expected problem:** later reports repeat earlier figures, so near-identical chunks could fill
  the context.
- **Measured:** with the period filter each search stays inside one report. In the top 10 of all
  116 questions there are 4 near-duplicate chunks (token Jaccard ≥ 0.9) in 3 questions, and 13 at
  ≥ 0.8.
- **Decision:** not built. There is no measured problem to fix; it would matter if the filter
  were removed.

---

## D13 · Give the model 8 chunks instead of 5

- **Problem (measured):** all 11 wrong answers after M3 were refusals ("I could not find this"),
  not wrong numbers. For most of them the evidence was retrieved, just below rank 5.
- **Evidence recall by context size** (production retrieval, 116 questions):

| Chunks | Recall@k | Context tokens (mean / max) |
|---|---|---|
| 3 | 0.783 | 735 / 1,367 |
| **5 (before)** | 0.887 | 1,172 / 2,275 |
| 6 | 0.910 | 1,411 / 2,728 |
| **8** | **0.943** | 1,904 / 3,652 |
| 10 | 0.948 | 2,398 / 4,568 |
| 12 | 0.967 | 2,876 / 5,425 |

- **Choice:** 8. Recall rises 5.6 points from 5 to 8 and only 0.5 more from 8 to 10; beyond that
  each extra point costs ~500 tokens per request and adds more unrelated chunks to the prompt.
- **Measured on answers** (`gemini-3.1-flash-lite`, same questions):

| Context | Fact match | False refusals | Correct refusals (unanswerable) | Unsupported numbers |
|---|---|---|---|---|
| 5 chunks | 0.896 | 0.094 | 10/10 | 0.010 |
| **8 chunks** | **0.943** | **0.038** | 10/10 | 0.010 |

  Five answers became correct (tb01, tb05, pw05, mp06, c07) and none became wrong. More context
  did not make the model answer unanswerable questions or invent numbers. c07 already had its
  chart at rank 1, so that one is generation variance rather than context.
- **Cost:** ~730 more input tokens per question (+62%); well inside the free tier's limits.

---

## D14 · Not built in M4: answer-time number check with retry, chart-vs-table conflict handling

These were planned for M4. The measurements showed no problem for them to fix.
- **Runtime number check + corrective retry:** across the M3 and D13 answer runs, one answer had a
  number missing from its sources, and it was a correct subtraction of two cited figures. A retry
  loop would add an LLM call and only ever fire on correct answers. The deterministic check stays
  in the evaluation, where it guards against regressions.
- **Conflicting evidence (approximate chart values vs exact table values):** 3 answers to non-chart
  questions cited a chart chunk, and all 3 were correct; no answer used a chart estimate where a
  table had the exact figure. The system prompt's rule ("approximately" for chart values) is
  enough at this point.
- **Parent–child tables** (send the whole table when one part is retrieved): the table misses
  after M3 were ranking problems (D11), not parts that lacked context; each part already repeats
  the header and carries the page's section title. With 8 chunks those questions are answered.
- **Re-ordering the context** (best chunks at the start and end, against "lost in the middle"):
  no answer failed with its evidence in the context, so there is nothing for ordering to fix.
- **De-duplication:** see D12.
