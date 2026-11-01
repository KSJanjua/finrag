# Architecture

Diagrams of the financial multimodal RAG system, derived from the code in `src/finrag/` and `scripts/`.
Each diagram's source is an editable Mermaid file in [`docs/architecture/`](architecture/); this page is
generated from them by `python docs/architecture/build_markdown.py`.

**Not implemented** (do not read these into the diagrams): authentication, a database, a web UI,
streaming responses, Docker/deployment, background job queues. Planned work is listed under
*Next steps* in the README.

## System context

01 · System context: who uses the system, what it stores, which external services it calls.  
_Source: [`architecture/01-system-context.mmd`](architecture/01-system-context.mmd)_ · Source of truth: scripts/*.py, src/finrag/api.py, src/finrag/llm.py, src/finrag/config.py

```mermaid
%% 01 · System context: who uses the system, what it stores, which external services it calls.
%% Source of truth: scripts/*.py, src/finrag/api.py, src/finrag/llm.py, src/finrag/config.py
flowchart LR
    dev(["Developer<br/>(terminal)"])
    client(["HTTP client<br/>(curl, browser /docs)"])

    subgraph app["finrag (Python 3.10+, runs locally, CPU only)"]
        direction TB
        cli["CLI scripts<br/>build_index · describe_charts · ask<br/>run_eval · validate_questions · check_llm<br/>reranker_sweep · chunk_sweep · faiss_benchmark"]
        api["FastAPI app (src/finrag/api.py)<br/>GET /health · POST /search · POST /ask"]
        core["finrag package<br/>ingest · chunking · charts · sparse · dense · query<br/>hybrid · rerank · retriever · generate · evaluate · llm"]
        cli --> core
        api --> core
    end

    subgraph files["Local files (no database)"]
        direction TB
        reg[("data/documents.json<br/>registry: file, period, label, url")]
        pdfs[("data/pdfs/*.pdf<br/>11 quarterly reports, 339 pages")]
        cache[("data/cache/chart_descriptions.json")]
        index[("data/index/chunks.jsonl<br/>+ embeddings.npy (901 × 384)")]
        evalq[("data/eval/questions.json<br/>116 labelled questions")]
        results[("results/*.md, *.json")]
    end

    tesla["digitalassets.tesla.com<br/>(PDF download, HTTPS)"]
    hf["Hugging Face Hub<br/>(model weights, first run)"]
    gemini["Gemini API · OpenAI-compatible<br/>generativelanguage.googleapis.com<br/>(default LLM, free tier)"]
    openai["OpenAI API (optional, paid)"]
    custom["Custom OpenAI-compatible server<br/>(optional, e.g. local Ollama, untested)"]

    dev -- "python scripts/…" --> cli
    client -- "HTTP JSON" --> api
    core <-- "read / write" --> files
    cli -- "urllib GET if PDFs missing" --> tesla
    core -- "download bge-small, MiniLM cross-encoder" --> hf
    core -- "chat.completions (text + images)" --> gemini
    core -. "FINRAG_LLM_PROVIDER=openai" .-> openai
    core -. "FINRAG_LLM_PROVIDER=custom" .-> custom
```

## Offline ingestion

02 · Offline ingestion: PDF pages -> typed chunks -> embeddings on disk.  
_Source: [`architecture/02-ingestion-pipeline.mmd`](architecture/02-ingestion-pipeline.mmd)_ · Source of truth: scripts/build_index.py, scripts/describe_charts.py, src/finrag/ingest.py, chunking.py, charts.py, dense.py, retriever.py (save_index)

```mermaid
%% 02 · Offline ingestion: PDF pages -> typed chunks -> embeddings on disk.
%% Source of truth: scripts/build_index.py, scripts/describe_charts.py,
%%                  src/finrag/ingest.py, chunking.py, charts.py, dense.py, retriever.py (save_index)
flowchart TB
    reg[("data/documents.json")] --> pdf[("data/pdfs/*.pdf")]
    pdf --> page["For each page: PyMuPDF (fitz)"]

    subgraph tables["Tables"]
        ft["page.find_tables()"] --> ok{"table_is_usable?<br/>≤40% empty cells,<br/>first column ≥50% unique"}
        ok -- yes --> md["to_markdown()"]
        ok -- "no (merged cells)" --> rw["rows_from_words():<br/>group words by y (±3pt),<br/>split cells on x-gaps > 8pt"]
        md --> st["split_table(): ≤450 tokens (embedding tokenizer),<br/>header repeated in every part"]
        rw --> st
    end

    subgraph text["Text"]
        blocks["text blocks outside table boxes<br/>(avoids indexing tables twice)"] --> clean["clean_text(): un-space headings,<br/>drop runs of ≥4 number-only lines"]
        clean --> rs["recursive_split(800, overlap 150)<br/>separators ¶ → line → '. ' → ' ' → char"]
        rs --> drop["drop pieces < 40 chars"]
    end

    subgraph charts["Charts (scripts/describe_charts.py, needs LLM key)"]
        cand{"chart_candidate(page)?<br/>axis-tick run ≥4 lines<br/>or > 4 raster images"} -- "yes (164 of 339 pages)" --> png["render_page_png(150 dpi)"]
        png --> vlm["Vision LLM via llm.chat()<br/>prompt: describe every chart as JSON"]
        vlm --> pj["parse_json(): tolerate ```json fences"]
        pj --> cachef[("data/cache/chart_descriptions.json<br/>{doc, page, description, source}<br/>null description = 'no charts'")]
    end

    page --> ft
    page --> blocks
    page --> cand
    st --> tc["table chunks (253)"]
    drop --> xc["text chunks (513)"]
    cachef --> cc["chart chunks (135)"]

    tc --> chunk["Chunk(chunk_id, doc, doc_label, page, type, section, text,<br/>fiscal_year, quarter from the registry)"]
    xc --> chunk
    cc --> chunk
    chunk --> it["index_text() = 'doc_label | page N | section | type' + text"]
    it --> emb["bge-small-en-v1.5 encode_docs()<br/>L2-normalised, float32"]
    chunk --> jl[("data/index/chunks.jsonl")]
    emb --> npy[("data/index/embeddings.npy  (901 × 384)")]
```

## Online query

03 · Online query: question -> retrieval funnel -> grounded answer with citations.  
_Source: [`architecture/03-query-pipeline.mmd`](architecture/03-query-pipeline.mmd)_ · Source of truth: src/finrag/query.py (parse_periods), retriever.py (Retriever.retrieve, report_filters), sparse.py, dense.py, hybrid.py, rerank.py, generate.py, config.py

```mermaid
%% 03 · Online query: question -> retrieval funnel -> grounded answer with citations.
%% Source of truth: src/finrag/query.py (parse_periods), retriever.py (Retriever.retrieve, report_filters),
%%                  sparse.py, dense.py, hybrid.py, rerank.py, generate.py, config.py
flowchart LR
    q(["question"]) --> pp["parse_periods(): 'Q3 2024', '3Q-2024',<br/>'third quarter of 2024', 'Q1, Q2 and Q3 2024',<br/>full-year 2024 → Q4 2024 report"]
    pp --> f{"periods named<br/>(doc = 'auto')"}
    f -- "one, with a report" --> k1a["k1 = all chunks, keep only that report"]
    f -- "none" --> k1b["k1 = 50, all reports"]
    f -- "several" --> dec["decomposition: one search per period's report<br/>(merged round-robin, duplicates dropped)"]
    dec --> k1a

    k1a --> bm25
    k1b --> bm25
    k1a --> dense
    k1b --> dense

    bm25["BM25Okapi (k1 1.5, b 0.75)<br/>number-aware tokenizer:<br/>'7.6%', '51105' stay whole"] --> rrf
    dense["FAISS IndexFlatIP (exact, cosine)<br/>query = BGE prefix + question"] --> rrf
    rrf["Reciprocal Rank Fusion<br/>score = Σ 1 / (60 + rank)"] --> mode{"mode"}

    mode -- "hybrid (default)" --> top5["top-8 RetrievedChunk<br/>(chunk, score, ranks)"]
    mode -. "hybrid_rerank (optional,<br/>lowered MRR 0.692→0.636)" .-> ce["cross-encoder ms-marco-MiniLM-L-6-v2<br/>scores top-20, batch 4, max 512 tokens<br/>(loaded lazily on first use)"]
    ce -.-> top5

    top5 --> ctx["build_context():<br/>[n] (doc_label, page, type) + text"]
    ctx --> llm["llm.chat(): system prompt rules<br/>only sources · cite [n] · exact numbers<br/>chart values 'approximately' · abstain"]
    llm --> pc["parse_citations(): [2], [1, 3], [1][2]<br/>drop out-of-range numbers"]
    pc --> out(["{answer, citations: [{n, doc, page, type}]}"])
    llm -. "no answer in sources" .-> abst(["'I could not find this in the provided documents.'"])
```

## REST API

04 · REST API: request/response flow with error paths.  
_Source: [`architecture/04-api-sequence.mmd`](architecture/04-api-sequence.mmd)_ · Source of truth: src/finrag/api.py, retriever.py, llm.py, generate.py

```mermaid
%% 04 · REST API: request/response flow with error paths.
%% Source of truth: src/finrag/api.py, retriever.py, llm.py, generate.py
sequenceDiagram
    autonumber
    participant C as HTTP client
    participant A as FastAPI app
    participant R as Retriever (in memory)
    participant L as llm.py (OpenAI SDK)
    participant G as Gemini API

    Note over A,R: Startup (lifespan): load_retriever() reads chunks.jsonl + embeddings.npy,<br/>builds BM25 + FAISS once. FileNotFoundError if the index was never built.

    C->>A: POST /ask {question, mode?, top_k?, doc?}
    alt body fails pydantic validation (question 3–500 chars, top_k 1–20)
        A-->>C: 422 Unprocessable Entity
    else mode not in MODES, or unknown doc file name
        A-->>C: 400 Bad Request
    else valid request
        A->>R: retrieve(question, mode, top_k, doc)
        R-->>A: top-k RetrievedChunk list
        A->>L: get_client()
        alt LLM key not set for the provider
            L-->>A: RuntimeError
            A-->>C: 503 Service Unavailable (names the missing key)
        else key set
            A->>L: answer(): chat(model, system + user prompt)
            Note over L: throttle: ≥4 s between calls (Gemini)
            L->>G: POST /v1beta/openai/chat/completions (HTTPS)
            alt HTTP 429 or 5xx
                G-->>L: error
                Note over L: SDK retries with exponential backoff (max_retries = 6)
            end
            G-->>L: completion text
            L-->>A: text → parse_citations()
            A-->>C: 200 {answer, citations, doc_filters, sources}
        end
    end

    C->>A: POST /search {question, …}
    A->>R: retrieve(...)
    A-->>C: 200 {doc_filters, results: [n, doc, page, type, score, text]}  (no LLM call)

    C->>A: GET /health
    A-->>C: 200 {status, chunks, llm_provider, llm_configured}
```

## LLM provider layer

05 · LLM provider layer: one OpenAI-compatible client, three providers.  
_Source: [`architecture/05-llm-provider.mmd`](architecture/05-llm-provider.mmd)_ · Source of truth: src/finrag/llm.py

```mermaid
%% 05 · LLM provider layer: one OpenAI-compatible client, three providers.
%% Source of truth: src/finrag/llm.py
flowchart LR
    env["Environment<br/>FINRAG_LLM_PROVIDER · FINRAG_CHAT/VISION/JUDGE_MODEL<br/>FINRAG_BASE_URL · FINRAG_MIN_INTERVAL · *_API_KEY"] --> s["settings()<br/>provider defaults + env overrides"]

    s --> gem["gemini (default)<br/>key GEMINI_API_KEY<br/>model gemini-3.8-flash<br/>reasoning_effort = low, no token cap<br/>min interval 4 s"]
    s --> oai["openai<br/>key OPENAI_API_KEY<br/>gpt-4o-mini / gpt-4o<br/>temperature 0, max_completion_tokens"]
    s --> cus["custom<br/>FINRAG_BASE_URL + FINRAG_CHAT_MODEL<br/>temperature 0, max_completion_tokens"]

    gem --> gc["get_client(): OpenAI(api_key, base_url, max_retries=6)"]
    oai --> gc
    cus --> gc
    gc --> chat["chat(client, model, messages)<br/>wait for min interval → create() → content or ''"]

    chat --> u1["generate.answer()"]
    chat --> u2["charts.describe_page()"]
    chat --> u3["evaluate.judge_correct()"]
    u2 --> pj["parse_json(): strip ```json fences,<br/>fall back to first {...} block"]
    u3 --> pj

    gc -. "key missing" .-> err["RuntimeError → API 503 / CLI message"]
```

## Evaluation workflow

06 · Evaluation workflow: labels -> retrieval grid -> answer correctness -> reports.  
_Source: [`architecture/06-evaluation.mmd`](architecture/06-evaluation.mmd)_ · Source of truth: scripts/validate_questions.py, scripts/run_eval.py, src/finrag/evaluate.py

```mermaid
%% 06 · Evaluation workflow: labels -> retrieval grid -> answer correctness -> reports.
%% Source of truth: scripts/validate_questions.py, scripts/run_eval.py, src/finrag/evaluate.py
flowchart TB
    qs[("questions.json<br/>116 questions: 32 text · 37 table · 15 chart<br/>10 paraphrase · 12 multi-period · 10 unanswerable<br/>evidence sets: graded gold pages + evidence strings")] --> val["validate_questions.py<br/>every evidence set found on a gold page?"]
    idx[("chunks.jsonl + embeddings.npy")] --> val

    qs --> grid
    idx --> grid
    subgraph grid["run_eval.py · retrieval grid (no API key)"]
        direction TB
        corp["3 corpora: text-only · text+tables · multimodal<br/>(Retriever.subset by chunk type)"] --> modes["4 modes: bm25 · dense · hybrid · hybrid_rerank<br/>+ 2 report-filter variants = 14 configurations"]
        modes --> rel["score_ranking(top-10):<br/>relevant = gold page AND contains evidence<br/>grade 2 = own report, 1 = later copy"]
        rel --> met["Hit@k · Recall@k · Precision@k<br/>NDCG@k (graded) · MRR"]
    end
    met --> rr[("results/retrieval_results.md<br/>retrieval_metrics.json · retrieval_misses.md")]
    rr -. "--check-repro: compare per-question ranks with last run" .-> met

    qs --> ans
    subgraph ans["run_eval.py --answers (LLM key, resumable)"]
        direction TB
        gen["per question (116 requests):<br/>hybrid + period filter → generate.answer()"] --> fm["facts_correct(): every fact group present,<br/>numbers matched whole ('5%' ≠ '35%')"]
        fm --> abst["abstained = answer contains refusal sentence<br/>(correct only for unanswerable questions)"]
        gen --> un["unsupported_numbers(): answer numbers<br/>not found in the cited chunks"]
        un --> ar
        gen -. "--judge (optional)" .-> judge["judge_correct(): LLM-as-judge"]
    end
    abst --> ar[("results/answer_results.md · answer_records.json")]
    judge -.-> ar

    sweeps["reranker_sweep.py · chunk_sweep.py · faiss_benchmark.py"] --> sr[("results/reranker_sweep.md · chunk_sweep.md<br/>faiss_benchmark.txt")]
```

## Data model

07 · Data model: in-memory types and on-disk records (there is no database).  
_Source: [`architecture/07-data-model.mmd`](architecture/07-data-model.mmd)_ · Source of truth: src/finrag/schema.py, retriever.py, ingest.py, charts.py, evaluate.py

```mermaid
%% 07 · Data model: in-memory types and on-disk records (there is no database).
%% Source of truth: src/finrag/schema.py, retriever.py, ingest.py, charts.py, evaluate.py
classDiagram
    class Chunk {
        +str chunk_id   "doc:pN:text0 / table0.1 / chart0"
        +str doc        "PDF file name"
        +str doc_label  "human-readable report name"
        +int page       "1-based, used in citations"
        +str type       "text | table | chart"
        +str section    "page heading"
        +str text
        +int fiscal_year "from the registry"
        +int quarter
        +index_text() str
        +citation() str
    }
    class RetrievedChunk {
        +Chunk chunk
        +float score
        +dict ranks   "bm25, dense, rrf, period"
    }
    class ChartCacheEntry {
        +str doc
        +int page
        +str description   "null = page checked, no charts"
        +str source        "vision model that produced it"
    }
    class Question {
        +str id
        +str type          "text | table | chart | paraphrase | multi | unanswerable"
        +str question
        +list evidence_sets "{gold: [period, page, grade], evidence}"
        +list answer_facts "groups of alternatives"
    }
    class AnswerRecord {
        +str modality
        +str id
        +str model
        +str answer
        +list citations
        +bool abstained
        +bool fact_match   "null for unanswerable"
        +list unsupported_numbers
        +bool judge        "optional"
    }
    RetrievedChunk --> Chunk : wraps
    ChartCacheEntry ..> Chunk : becomes a chart chunk
    Question ..> RetrievedChunk : is_relevant()
    AnswerRecord ..> Question : scored against answer_facts
```
