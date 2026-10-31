"""FastAPI service.

    uvicorn finrag.api:app --app-dir src --port 8000
    open http://localhost:8000/docs

Endpoints
    GET  /health   index size, LLM provider and whether its key is configured
    POST /search   retrieval only (no API key needed)
    POST /ask      retrieval + grounded answer with citations (needs an LLM key, see README)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .llm import llm_configured, settings
from .retriever import MODES, infer_doc, load_retriever

state = {}


@asynccontextmanager
async def lifespan(app):
    state["retriever"] = load_retriever()  # load models once, not per request (reranker: on first use)
    yield
    state.clear()


app = FastAPI(title="Financial Multimodal RAG", version="1.0", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    mode: str = Field(config.DEFAULT_MODE, description=f"one of {MODES}")
    top_k: int = Field(config.FINAL_K, ge=1, le=20)
    doc: str | None = Field(config.DEFAULT_DOC_FILTER,
                            description='null = all reports, "auto" = infer from the question, or a PDF file name')


def _retrieve(q: Query):
    if q.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    try:
        hits = state["retriever"].retrieve(q.question, mode=q.mode, top_k=q.top_k, doc=q.doc)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    applied = infer_doc(q.question) if q.doc == "auto" else q.doc
    return hits, applied


def _source(n, h):
    c = h.chunk
    return {"n": n, "doc": c.doc_label, "page": c.page, "type": c.type,
            "score": round(h.score, 4), "text": c.text}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks),
            "llm_provider": settings()["name"], "llm_configured": llm_configured()}


@app.post("/search")
def search(q: Query):
    """Retrieval only - no LLM call."""
    hits, applied = _retrieve(q)
    return {"doc_filter": applied, "results": [_source(n, h) for n, h in enumerate(hits, 1)]}


@app.post("/ask")
def ask(q: Query):
    from .generate import answer
    from .llm import get_client
    hits, applied = _retrieve(q)
    try:
        client = get_client()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    out = answer(client, q.question, hits)
    return {**out, "doc_filter": applied, "sources": [_source(n, h) for n, h in enumerate(hits, 1)]}
