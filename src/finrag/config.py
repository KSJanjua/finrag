"""Central configuration: paths, model names and retrieval hyper-parameters."""
import os
from pathlib import Path

# Hugging Face prints a symlink warning on Windows without Developer Mode; it is harmless.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = ROOT / "results"

CHART_CACHE = CACHE_DIR / "chart_descriptions.json"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
QUESTIONS_FILE = EVAL_DIR / "questions.json"

# Human-readable document names. They are added to every chunk so that a query
# like "Q2 2023 capacity" can match the right report lexically and semantically.
DOC_LABELS = {
    "TSLA-Q1-2023-Update.pdf": "Tesla Q1 2023 Quarterly Update",
    "TSLA-Q2-2023-Update.pdf": "Tesla Q2 2023 Quarterly Update",
    "TSLA-Q3-2023-Update-3.pdf": "Tesla Q3 2023 Quarterly Update",
}
PDF_URLS = {
    name: f"https://digitalassets.tesla.com/tesla-contents/image/upload/IR/{name}"
    for name in DOC_LABELS
}

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 150       # characters shared between neighbouring chunks
TABLE_MAX_CHARS = 1500    # long tables are split by rows, header repeated

# --- Embeddings / FAISS -----------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim bi-encoder
# BGE models are trained with an instruction prefix on the *query* side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- BM25 ---------------------------------------------------------------------
BM25_K1 = 1.5    # term-frequency saturation
BM25_B = 0.75    # document-length normalisation

# --- Retrieval funnel ---------------------------------------------------------
FIRST_STAGE_K = 50        # candidates taken from each of BM25 and FAISS
RRF_K = 60                # constant in 1 / (k + rank)
RERANK_CANDIDATES = 20    # fused candidates sent to the cross-encoder
FINAL_K = 5               # chunks given to the LLM
DEFAULT_DOC_FILTER = "auto"   # CLI/API default: infer the report from "Q3 2023" in the question
# Default retrieval mode. The cross-encoder ("hybrid_rerank") was evaluated and lowered
# MRR on this corpus while adding ~2 s/query (results/retrieval_results.md,
# results/reranker_sweep.md), so plain hybrid RRF is the default.
DEFAULT_MODE = "hybrid"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH_SIZE = 4     # small batches = less padding (see rerank.py)

# --- LLMs --------------------------------------------------------------------
# Provider, models and keys are configured in llm.py (FINRAG_LLM_PROVIDER etc.).
