"""Central configuration: paths, document registry, model names and retrieval hyper-parameters."""
import json
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

REGISTRY_FILE = DATA_DIR / "documents.json"


def _load_registry():
    """Source documents keyed by file name, in chronological order."""
    docs = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))["documents"]
    docs.sort(key=lambda d: (d["fiscal_year"], d["quarter"]))
    return {d["file"]: d for d in docs}


DOCUMENTS = _load_registry()
# Labels are added to every chunk so that a query like "Q2 2023 capacity" can
# match the right report lexically and semantically.
DOC_LABELS = {name: d["label"] for name, d in DOCUMENTS.items()}
PDF_URLS = {name: d["url"] for name, d in DOCUMENTS.items()}
DOC_BY_PERIOD = {(d["fiscal_year"], d["quarter"]): name for name, d in DOCUMENTS.items()}

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 150       # characters shared between neighbouring chunks
# Long tables are split by rows (header repeated) so that every part fits the
# embedding model's 512-token input, leaving room for the chunk's label line.
TABLE_MAX_TOKENS = 450

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
# A question naming several periods ("Q4 2022 vs Q4 2024") is answered from one
# search per period's report instead of one search over all reports.
DECOMPOSE_PERIODS = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH_SIZE = 4     # small batches = less padding (see rerank.py)

# --- LLMs --------------------------------------------------------------------
# Provider, models and keys are configured in llm.py (FINRAG_LLM_PROVIDER etc.).
