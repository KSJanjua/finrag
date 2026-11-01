# Multimodal RAG for Financial Document Analysis

An end-to-end retrieval-augmented generation (RAG) system over Tesla's quarterly investor
updates. It answers questions whose evidence sits in **text, tables or charts**, cites the page it
used, and says "I could not find this" instead of guessing.

> **v2 in progress.** The corpus now covers 11 quarterly updates (Q4 2022 – Q2 2025, 339 pages,
> listed in `data/documents.json`), with a 116-question evaluation set. The results below are the
> v2 baseline; improvements are tracked in [docs/ROADMAP.md](docs/ROADMAP.md) and every design
> decision in [docs/DECISIONS.md](docs/DECISIONS.md).

```
PDF ──► PyMuPDF ──┬─ text blocks ─► clean ─► recursive chunks (800 chars / 150 overlap) ─┐
                  ├─ tables ─► markdown (row-rebuild fallback) ─► split by rows ──────────┤ typed chunks
                  └─ page PNG ─► Gemini vision ─► chart descriptions ─────────────────────┘ {doc, page, type}

question ─► "Q3 2023"? ─► report filter
        ─┬─ BM25 top-50 ───────────────┐
         └─ FAISS (bge-small) top-50 ──┴─► RRF ─► top-5 ─► LLM ─► answer with [n] page citations
                                (optional: cross-encoder rerank of the top-20 - evaluated, not the default)
```

Chunking, the BM25 tokenizer, RRF, the retrieval funnel, the table fallback and every metric are
implemented by hand (no LangChain/LlamaIndex), so each step can be inspected and explained.

## Results

### v2 baseline (11 reports, 116 questions)

These numbers measure the v1 pipeline, unchanged, on the larger corpus and the harder v2 question
set. They are the baseline that the v2 retrieval and generation work is measured against.

The question set has 106 answerable questions (32 text, 37 table, 15 chart, 10 with re-worded
periods, 12 multi-period) and 10 that the documents can't answer. A retrieved chunk is relevant if
it comes from a gold page and contains the answer evidence. Grade 2 means the period's own report;
grade 1 means a later report repeating the figure.

| Corpus | Retriever | Hit@1 | Hit@5 | Recall@5 | NDCG@10 | MRR | ms/query (CPU) |
|---|---|---|---|---|---|---|---|
| text-only | hybrid | 0.151 | 0.236 | 0.236 | 0.212 | 0.192 | 57 |
| text + tables | hybrid | 0.340 | 0.575 | 0.557 | 0.452 | 0.441 | 57 |
| text + tables + charts | hybrid (RRF) | 0.396 | 0.642 | 0.637 | 0.512 | 0.510 | 59 |
| text + tables + charts | hybrid + cross-encoder rerank | 0.302 | 0.575 | 0.561 | 0.450 | 0.420 | ~2300 |
| **text + tables + charts** | **hybrid + report filter (default)** | **0.528** | **0.783** | **0.778** | **0.603** | **0.639** | 61 |

Hit@5 by question type for the default configuration (Recall@5 for multi-period):

| text | table | chart | re-worded periods | multi-period |
|---|---|---|---|---|
| 0.781 | 0.865 | 1.000 | **0.400** | **0.542** |

Answers (default configuration, `gemini-3.1-flash-lite`, one question per request):

| Fact match | False refusals | Correct refusals on unanswerable | Answers with unsupported numbers |
|---|---|---|---|
| 0.840 | 0.132 | 1.000 (10/10) | 0.011 |

Where the baseline fails ([results/retrieval_misses.md](results/retrieval_misses.md)):
- **Re-worded periods** ("third quarter of 2024", "2Q-2024", "Q1 FY2025"): the report filter only
  recognises "Q3 2024", so these questions search all 11 reports and 6 of 10 miss.
- **Multi-period questions** spanning reports more than a year apart: one search rarely brings back
  both periods (Recall@5 0.54).
- **Full-year questions** ("full-year 2024"): annual figures live in the Q4 report, but nothing
  maps the year to it.
- **Large tables:** 97 of 165 table chunks exceed the embedding model's 512-token input, and
  several table facts only appear at ranks 6–8.
- The cross-encoder still lowers ranking quality (MRR 0.639 → 0.538 with the report filter), as it did in v1.

Full grid: [results/retrieval_results.md](results/retrieval_results.md); answers:
[results/answer_results.md](results/answer_results.md). The one flagged "unsupported number" is a
correctly computed difference ($14.4B growth); computed figures are a known limitation of the
number-grounding check.

### v1 (tag `v1.0`, 3 reports, 38 questions)

On the original corpus the default configuration reached Hit@5 0.974, MRR 0.912 and 0.974 answer
fact match (0.395 text-only). The chunk-size, reranker and FAISS experiments in `results/`
(`chunk_sweep.md`, `reranker_sweep.md`, `faiss_benchmark.txt`) are from v1.

## Setup

Requires Python 3.10+ (verified on Python 3.14, Windows 11, CPU only).

```bash
pip install -r requirements.txt          # or: pip install -r requirements-lock.txt (exact verified versions)
python scripts/build_index.py            # downloads the 3 PDFs, parses, chunks, embeds (~2 min on CPU)
python scripts/validate_questions.py     # checks every eval label against the real chunks
python -m pytest -q tests                # 52 tests (unit, LLM paths with a mocked API, integration, API)
```

Retrieval, evaluation of retrieval, the API's `/search` endpoint and `ask.py --no-llm` need **no API key**.

### LLM setup (free)

Answer generation, chart description and the LLM judge use any OpenAI-compatible API. The default is
**Google Gemini's free tier** (no credit card):

1. Create a key at https://aistudio.google.com/apikey
2. `setx GEMINI_API_KEY "your-key"` (Windows; on macOS/Linux: `export GEMINI_API_KEY=...`), then open a new terminal
3. `python scripts/check_llm.py` - lists your models and runs one text and one vision test

Free-tier facts that shape the code: **20 requests per day per model** (aliases such as `gemini-3.5-flash`
share another model's quota), and free-tier requests may be used by Google to improve its products
(fine for these public PDFs). All LLM jobs therefore throttle, retry on HTTP 429, save after every request
and **resume** where they stopped when re-run.

| Variable | Default | Purpose |
|---|---|---|
| `FINRAG_LLM_PROVIDER` | `gemini` | `gemini`, `openai` (needs `OPENAI_API_KEY`, paid) or `custom` |
| `FINRAG_CHAT_MODEL` / `FINRAG_VISION_MODEL` / `FINRAG_JUDGE_MODEL` | `gemini-3.1-flash-lite` (chat, vision), `gemini-3.8-flash` (judge) | model per role |
| `FINRAG_BASE_URL`, `FINRAG_API_KEY` | - | for `custom`, e.g. a local Ollama at `http://localhost:11434/v1` (not tested here) |
| `FINRAG_MIN_INTERVAL` | 4 s (gemini) | minimum seconds between requests |

## Usage

```bash
python scripts/ask.py "What was Tesla's operating margin in Q3 2023?"
python scripts/ask.py "Model Y price in Colorado after subsidies?" --no-llm    # retrieval only
python scripts/ask.py "Supercharger connectors in Q3 2023" --doc all          # disable the report filter
python scripts/ask.py "..." --mode hybrid_rerank                              # use the cross-encoder

uvicorn finrag.api:app --app-dir src --port 8000                              # REST API
```

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | - | chunk count, LLM provider, whether its key is set |
| `POST /search` | `{"question", "mode"?, "top_k"?, "doc"?}` | ranked chunks with doc, page, type, score, text, and the report filter applied (no LLM) |
| `POST /ask` | same | `answer`, `citations` (doc, page, type), `sources`; 503 if no LLM key |

`mode`: `hybrid` (default), `bm25`, `dense`, `hybrid_rerank`. `doc`: `"auto"` (default, infer from the
question), `null` (all reports) or a PDF file name. Interactive docs: http://localhost:8000/docs

## Evaluation

```bash
python scripts/run_eval.py --check-repro                 # retrieval grid -> results/, compares with the last run
python scripts/run_eval.py --answers-only --answers      # answer evaluation (116 LLM requests, resumable)
python scripts/run_eval.py --answers-only --answers --text-only   # + the text-only corpus for comparison
python scripts/run_eval.py --answers-only --answers --judge       # + LLM-as-judge
python scripts/validate_questions.py                     # every evidence set must be findable in the index
python scripts/describe_charts.py                        # (re)describe chart pages - resumable, quota-aware
python scripts/reranker_sweep.py | chunk_sweep.py | faiss_benchmark.py   # experiments -> results/
```

## Project status

| Item | Status |
|---|---|
| Ingestion, index, retrieval, API `/search`, CLI | Working, tested |
| Retrieval evaluation | v2 baseline on 116 questions; deterministic (identical ranks across runs) |
| Chart descriptions | All 164 candidate pages of the 11 reports described by `gemini-3.1-flash-lite` (model recorded per entry); chart answer labels checked against the rendered page images |
| LLM code paths (generation, vision, judge, JSON parsing, per-provider parameters) | Tested against the real OpenAI SDK with a mocked HTTP transport; generation and vision also verified live against Gemini |
| Answer evaluation | v2 baseline: 116 answers on the one-question production path (`gemini-3.1-flash-lite`); fact match, refusals and number grounding; LLM judge optional, not run |

## Project layout

```
src/finrag/
  config.py      paths, document registry, models, hyper-parameters, defaults
  schema.py      Chunk (doc, page, type, section, text)
  ingest.py      PyMuPDF text/tables, cleaning, row-rebuild table fallback, chunk building
  chunking.py    recursive splitter with overlap; table splitting with repeated header
  charts.py      chart-page filter, page rendering, vision-LLM chart descriptions
  llm.py         provider layer (Gemini / OpenAI / custom), throttling, retries, tolerant JSON parsing
  sparse.py      BM25 + number-aware tokenizer
  dense.py       bge-small embeddings + exact FAISS index
  hybrid.py      Reciprocal Rank Fusion
  rerank.py      optional cross-encoder reranker
  retriever.py   the funnel, modality subsets, query -> report filter, persistence
  generate.py    grounded prompt with [n] citations and abstention
  evaluate.py    evidence-level relevance, Recall@k, MRR, whole-number fact matching, LLM judge
  api.py         FastAPI: /health, /search, /ask
scripts/         build_index, check_llm, describe_charts, ask, run_eval, validate_questions,
                 reranker_sweep, chunk_sweep, faiss_benchmark
data/documents.json        document registry (period, label, URL, publication date)
data/eval/questions.json   116 labelled questions     data/cache/   chart descriptions
results/                   generated reports          tests/        52 tests
docs/ARCHITECTURE.md       architecture diagrams (Mermaid, generated from docs/architecture/*.mmd)
```

## Limitations

- 116 questions is still a small test set (1 question ≈ 0.9 points of Hit@5), written by the author;
  table questions are generated from extracted table cells, so they share one phrasing style.
- Chart values come from a vision model and are approximate, so chart-based answer labels were
  checked against the rendered pages.
- Statement tables with merged cells are rebuilt heuristically from word positions.
- The report filter only understands "Qn 2023"-style mentions, and it narrows to a single report
  when only one quarter carries the year (e.g. "Q1, Q2 and Q3 2023").
- Large tables can exceed the embedding model's 512-token input; the overflow is only visible to BM25.
- Number grounding flags correctly computed figures (e.g. a difference between two quarters) as
  unsupported, because computed values don't appear in any source.

## Next steps

See [docs/ROADMAP.md](docs/ROADMAP.md): retrieval fixes for the baseline's failure modes (query
parsing, query decomposition, table chunks that fit the embedding model), then context engineering,
grounded generation checks, production concerns and prompt-injection tests.
