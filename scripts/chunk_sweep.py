"""Chunk-size ablation: how does text chunk size affect retrieval?

Small chunks = precise embeddings but context gets cut (e.g. a list separated
from the sentence introducing it). Large chunks = more context per chunk but
the embedding averages over several topics. Table and chart chunks are not
affected (tables are split by rows, charts are one chunk each).

    python scripts/chunk_sweep.py
"""
import _bootstrap  # noqa: F401

from finrag import config
from finrag.dense import Embedder
from finrag.evaluate import first_relevant_rank, load_questions, retrieval_metrics
from finrag.ingest import build_chunks
from finrag.rerank import CrossEncoderReranker
from finrag.retriever import Retriever

SETTINGS = [(400, 75), (800, 150), (1200, 200), (1600, 250)]


def main():
    questions = load_questions()
    embedder, reranker = Embedder(), CrossEncoderReranker()
    lines = ["# Chunk-size sweep (multimodal corpus)", "",
             "| chunk size / overlap | #text chunks | mode | Recall@1 | Recall@5 | MRR |",
             "|---|---|---|---|---|---|"]
    for size, overlap in SETTINGS:
        chunks = build_chunks(chunk_size=size, overlap=overlap)
        emb = embedder.encode_docs([c.index_text() for c in chunks])
        retriever = Retriever(chunks, emb, embedder, reranker)
        n_text = sum(c.type == "text" for c in chunks)
        for mode in ("hybrid", "hybrid_rerank"):
            ranks = [first_relevant_rank(retriever.retrieve(q["question"], mode=mode, top_k=10), q)
                     for q in questions]
            m = retrieval_metrics(ranks)
            print(size, overlap, mode, m)
            lines.append(f"| {size} / {overlap} | {n_text} | {mode} | {m['recall@1']:.3f} | "
                         f"{m['recall@5']:.3f} | {m['mrr']:.3f} |")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "chunk_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
