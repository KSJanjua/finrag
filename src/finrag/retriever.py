"""The retrieval funnel.

    query ─┬─ BM25  top-50 ─┐
           └─ FAISS top-50 ─┴─ RRF ─ top-5 ─> LLM          (default: "hybrid")
                                  └─ top-20 ─ cross-encoder ─ top-5   ("hybrid_rerank")

Modes (used for ablations): "bm25", "dense", "hybrid", "hybrid_rerank".
The cross-encoder is optional: it was evaluated and did not help on this corpus.
"""
import json
import re

import numpy as np

from . import config
from .dense import DenseRetriever, Embedder
from .hybrid import reciprocal_rank_fusion
from .rerank import CrossEncoderReranker
from .schema import Chunk, RetrievedChunk
from .sparse import BM25Retriever

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

# "Q3 2023", "Q3-2023", "Q3'23" -> the report for that quarter
QUARTER_PATTERN = re.compile(r"\bQ([1-3])[\s\-']*(?:20)?23\b", re.IGNORECASE)
QUARTER_DOCS = {"1": "TSLA-Q1-2023-Update.pdf", "2": "TSLA-Q2-2023-Update.pdf",
                "3": "TSLA-Q3-2023-Update-3.pdf"}


def infer_doc(query):
    """Metadata extraction from the query: which report is it asking about?

    The three reports share near-identical tables, so "connectors at the end
    of Q3 2023" otherwise matches the Q1 and Q2 tables just as well. Returns
    None when the query names no quarter (or several), i.e. no filter.
    """
    quarters = set(QUARTER_PATTERN.findall(query))
    return QUARTER_DOCS[quarters.pop()] if len(quarters) == 1 else None


class Retriever:
    def __init__(self, chunks, embeddings, embedder, reranker=None):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings - rebuild the index")
        self.chunks = chunks
        self.embeddings = embeddings
        texts = [c.index_text() for c in chunks]
        self.bm25 = BM25Retriever(texts)
        self.dense = DenseRetriever(embeddings, embedder)
        self.reranker = reranker

    def subset(self, types):
        """New retriever restricted to some chunk types (for modality ablations)."""
        keep = [i for i, c in enumerate(self.chunks) if c.type in types]
        return Retriever([self.chunks[i] for i in keep], self.embeddings[keep],
                         self.dense.embedder, self.reranker)

    def retrieve(self, query, mode=config.DEFAULT_MODE, top_k=config.FINAL_K, doc=None):
        """doc: None (no filter), a PDF file name, or "auto" (infer from the query)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if doc == "auto":
            doc = infer_doc(query)
        elif doc is not None and doc not in config.DOC_LABELS:
            raise ValueError(f"unknown doc {doc!r}; use None, 'auto' or one of {list(config.DOC_LABELS)}")
        # With a document filter we score everything and filter afterwards;
        # fine for ~200 chunks (at scale: per-document indexes or FAISS IDSelector).
        k1 = len(self.chunks) if doc else config.FIRST_STAGE_K
        allowed = (lambda i: self.chunks[i].doc == doc) if doc else (lambda i: True)

        bm25_hits = [(i, s) for i, s in self.bm25.search(query, k1) if allowed(i)] \
            if mode != "dense" else []
        dense_hits = [(i, s) for i, s in self.dense.search(query, k1) if allowed(i)] \
            if mode != "bm25" else []
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_hits, 1)}
        dense_rank = {i: r for r, (i, _) in enumerate(dense_hits, 1)}

        if mode == "bm25":
            ranked = bm25_hits
        elif mode == "dense":
            ranked = dense_hits
        else:
            ranked = reciprocal_rank_fusion([[i for i, _ in bm25_hits], [i for i, _ in dense_hits]])

        if mode == "hybrid_rerank":
            if self.reranker is None:  # loaded on first use, so the default path stays light
                self.reranker = CrossEncoderReranker()
            candidates = [i for i, _ in ranked[:config.RERANK_CANDIDATES]]
            scores = self.reranker.score(query, [self.chunks[i].index_text() for i in candidates])
            fused_rank = {i: r for r, i in enumerate(candidates, 1)}
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        else:
            fused_rank = {}

        results = []
        for i, score in ranked[:top_k]:
            ranks = {"bm25": bm25_rank.get(i), "dense": dense_rank.get(i)}
            if fused_rank:
                ranks["rrf"] = fused_rank.get(i)
            results.append(RetrievedChunk(self.chunks[i], float(score), ranks))
        return results


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_index(chunks, embeddings, index_dir=config.INDEX_DIR):
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / config.CHUNKS_FILE.name, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    # The embeddings are the persisted artefact; the exact FAISS index is rebuilt
    # from them at load time (a memory copy, milliseconds for ~200 vectors).
    np.save(index_dir / config.EMBEDDINGS_FILE.name, embeddings)


def load_chunks(index_dir=config.INDEX_DIR):
    with open(index_dir / config.CHUNKS_FILE.name, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def load_retriever(use_reranker=False, index_dir=config.INDEX_DIR):
    """use_reranker=True loads the cross-encoder now; otherwise it loads lazily if needed."""
    if not (index_dir / config.CHUNKS_FILE.name).exists():
        raise FileNotFoundError("No index found - run `python scripts/build_index.py` first.")
    chunks = load_chunks(index_dir)
    embeddings = np.load(index_dir / config.EMBEDDINGS_FILE.name)
    reranker = CrossEncoderReranker() if use_reranker else None
    return Retriever(chunks, embeddings, Embedder(), reranker)
