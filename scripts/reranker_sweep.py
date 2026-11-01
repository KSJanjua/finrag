"""Is the cross-encoder worth its latency? Compare reranker settings on the eval set.

    python scripts/reranker_sweep.py

Rows: no reranker (hybrid RRF only) vs ms-marco-MiniLM-L-6-v2 with different
batch sizes / max input lengths, each with and without the query -> report
filter. Writes results/reranker_sweep.md. Latency is CPU wall-clock.
"""
import _bootstrap  # noqa: F401

import time

from finrag import config
from finrag.evaluate import aggregate, ideal_grades, load_questions, score_ranking
from finrag.rerank import CrossEncoderReranker
from finrag.retriever import load_retriever

SETTINGS = [  # (max_length, batch_size)
    (512, 32),
    (512, 4),    # production setting
    (256, 4),
]


class Reranker(CrossEncoderReranker):
    def __init__(self, max_length, batch_size):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=max_length)
        self.batch_size = batch_size

    def score(self, query, texts):
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=self.batch_size)]


def main():
    questions = [q for q in load_questions() if q["sets"]]
    retriever = load_retriever(use_reranker=False)
    ideal = {q["id"]: ideal_grades(retriever.chunks, q) for q in questions}
    lines = [f"# Reranker sweep (multimodal corpus, {len(questions)} answerable questions)", "",
             "| Reranker | Report filter | Hit@1 | Hit@5 | MRR | ms/query |", "|---|---|---|---|---|---|"]

    def run(label, mode):
        for doc in (None, "auto"):
            t0 = time.perf_counter()
            m = aggregate([score_ranking(retriever.retrieve(q["question"], mode=mode, top_k=10, doc=doc), q,
                                         ideal[q["id"]]) for q in questions])
            ms = 1000 * (time.perf_counter() - t0) / len(questions)
            line = (f"| {label} | {'auto' if doc else 'none'} | {m['hit@1']:.3f} | {m['hit@5']:.3f} | "
                    f"{m['mrr']:.3f} | {ms:.0f} |")
            print(line, flush=True)
            lines.append(line)

    run("none (hybrid RRF only)", "hybrid")
    for max_length, batch_size in SETTINGS:
        retriever.reranker = Reranker(max_length, batch_size)
        run(f"MiniLM-L6 max_len={max_length} batch={batch_size}", "hybrid_rerank")
    lines += ["", "Not included: `mixedbread-ai/mxbai-rerank-xsmall-v1` did not finish 38 queries in 45 "
              "minutes on this CPU (> 70 s/query), and `BAAI/bge-reranker-base` (278M parameters) is "
              "larger still - both are impractical without a GPU."]
    (config.RESULTS_DIR / "reranker_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {config.RESULTS_DIR / 'reranker_sweep.md'}")


if __name__ == "__main__":
    main()
