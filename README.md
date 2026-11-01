# Multimodal RAG for Financial Document Analysis

An end-to-end retrieval-augmented generation (RAG) system over Tesla's quarterly investor
updates. It answers questions whose evidence sits in **text, tables or charts**, cites the page it
used, and says "I could not find this" instead of guessing.

> **v2 in progress.** The corpus now covers 11 quarterly updates (Q4 2022 – Q2 2025, 339 pages,
> listed in `data/documents.json`). The results below are from v1 (tag `v1.0`, three 2023
> reports) until the v2 evaluation is in place. Plan: [docs/ROADMAP.md](docs/ROADMAP.md);
> design decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

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

**Retrieval** (38 hand-labelled questions: 14 text, 13 table, 11 chart). A retrieved chunk counts as a
hit only if it comes from a gold page **and** contains the answer evidence. Reproducible: two
consecutive runs give identical per-question ranks (`run_eval.py --check-repro`).

| Corpus | Retriever | Recall@1 | Recall@5 | MRR | ms/query (CPU) |
|---|---|---|---|---|---|
| text-only | hybrid | 0.342 | 0.421 | 0.377 | 25 |
| text + tables | hybrid | 0.579 | 0.711 | 0.634 | 32 |
| text + tables + charts | BM25 | 0.658 | 0.868 | 0.751 | 1 |
| text + tables + charts | dense (FAISS) | 0.737 | 0.895 | 0.793 | 31 |
| text + tables + charts | hybrid (RRF) | 0.789 | 0.947 | 0.854 | 32 |
| text + tables + charts | hybrid + cross-encoder rerank | 0.684 | 0.947 | 0.779 | ~1400 |
| **text + tables + charts** | **hybrid + report filter (default)** | **0.868** | **0.974** | **0.912** | 31 |

By question type (Recall@5): text-only retrieval finds **0 %** of chart answers and 23 % of table answers;
the multimodal index finds **100 %** of both. Full grid: [results/retrieval_results.md](results/retrieval_results.md).

What the experiments showed:
- **Charts and tables are what make the difference**: Recall@5 goes 0.42 → 0.71 → 0.97 as tables, then charts, are indexed.
- **Hybrid beats either retriever alone** (0.947 vs 0.868 BM25 / 0.895 dense at Recall@5).
- **The cross-encoder reranker made results worse here** (MRR 0.854 → 0.779) and costs ~1.4 s/query, so it is
  available as `--mode hybrid_rerank` but not the default. See [results/reranker_sweep.md](results/reranker_sweep.md).
- **Filtering to the report named in the question** ("Q3 2023") fixes confusion between the three reports'
  near-identical tables (MRR 0.854 → 0.912).
- **Chunk size**: [results/chunk_sweep.md](results/chunk_sweep.md). **FAISS at scale** (100k vectors, exact vs
  HNSW / IVF / IVF-PQ): [results/faiss_benchmark.txt](results/faiss_benchmark.txt).

**Answer correctness** (38 questions per corpus, one question per request, generator `gemini-3.1-flash-lite`,
default retrieval). An answer is correct if it contains every required fact (numbers matched whole).

| Corpus | Fact match | text | table | chart | Abstained |
|---|---|---|---|---|---|
| text-only | 0.395 | 0.929 | 0.154 | 0.000 | 0.526 |
| **text + tables + charts** | **0.974** | 0.929 | 1.000 | 1.000 | 0.000 |

- **Stable:** two full runs gave identical per-question correctness (76/76 pairs agree).
- **Error analysis:** the single multimodal miss (t02) is a retrieval miss: the list of operating-income drivers sits
  in a chunk outside the top 5, and the model answered with the nearby *revenue* drivers instead. Text-only mostly
  abstained when evidence was missing, but once (c09) it reported "8 million" FSD miles, which is the page
  number printed under the chart title, a hallucination the chart descriptions prevent.
- The optional LLM judge was not run (free-tier quota); fact-match results were checked by reading the answers.
  Details: [results/answer_results.md](results/answer_results.md), [results/answer_records.json](results/answer_records.json).

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
python scripts/run_eval.py --answers-only --answers      # answer correctness (76 LLM requests, resumable)
python scripts/run_eval.py --answers-only --answers --judge   # + LLM-as-judge (another 76)
python scripts/describe_charts.py                        # (re)describe chart pages - resumable, quota-aware
python scripts/reranker_sweep.py | chunk_sweep.py | faiss_benchmark.py   # experiments -> results/
```

## Project status

| Item | Status |
|---|---|
| Ingestion, index, retrieval, API `/search`, CLI | Working, tested |
| Retrieval evaluation | Complete and reproducible (identical ranks across runs) |
| Chart descriptions | Complete: all 37 candidate pages described by Gemini (`gemini-3.8/3.7/3.6-flash`, model recorded per entry); chart answers c06/c07 checked against the rendered page images |
| LLM code paths (generation, vision, judge, JSON parsing, per-provider parameters) | Tested against the real OpenAI SDK with a mocked HTTP transport; generation and vision also verified live against Gemini |
| Answer-correctness evaluation | Complete: 76 answers on the one-question production path (`gemini-3.1-flash-lite`), two runs with identical per-question correctness; LLM judge optional, not run |

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
data/eval/questions.json   38 labelled questions      data/cache/   chart descriptions
results/                   generated reports          tests/        52 tests
docs/ARCHITECTURE.md       architecture diagrams (Mermaid, generated from docs/architecture/*.mmd)
```

## Limitations

- 38 questions is a small test set (1 question = 2.6 recall points), written by the author.
- Chart values come from a vision model and are approximate, so chart-based answer labels were
  checked against the rendered pages.
- Statement tables with merged cells are rebuilt heuristically from word positions.
- The report filter only understands "Qn 2023"-style mentions, and it narrows to a single report
  when only one quarter carries the year (e.g. "Q1, Q2 and Q3 2023").
- Large tables can exceed the embedding model's 512-token input; the overflow is only visible to BM25.

## Next steps

- A larger evaluation set with multi-period and unanswerable questions.
- A query parser for quarter and year mentions in any wording.
- Smaller table chunks so every chunk fits the embedding model.
- Per-report indexes or a vector store with metadata filters once the corpus grows.
