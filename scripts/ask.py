"""Ask a question from the command line.

    python scripts/ask.py "What was Tesla's operating margin in Q3 2023?"
    python scripts/ask.py "Model Y price in Colorado?" --no-llm     # show retrieved chunks only
    python scripts/ask.py "Shanghai capacity" --doc TSLA-Q2-2023-Update.pdf --mode hybrid_rerank
    python scripts/ask.py "Supercharger connectors in Q3 2023" --doc all   # no report filter
"""
import _bootstrap  # noqa: F401

import argparse

from finrag import config
from finrag.retriever import MODES, load_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default=config.DEFAULT_MODE, choices=MODES)
    ap.add_argument("--k", type=int, default=config.FINAL_K)
    ap.add_argument("--doc", default=config.DEFAULT_DOC_FILTER,
                    help='"auto" (default) infers the report from "Q3 2023" in the question; '
                         '"all" disables filtering; or give a PDF file name')
    ap.add_argument("--no-llm", action="store_true", help="only print retrieved chunks")
    args = ap.parse_args()

    retriever = load_retriever()
    doc = None if args.doc == "all" else args.doc
    hits = retriever.retrieve(args.question, mode=args.mode, top_k=args.k, doc=doc)

    print(f"\nTop {len(hits)} chunks ({args.mode}):")
    for n, h in enumerate(hits, 1):
        c = h.chunk
        preview = " ".join(c.text.split())[:160]
        print(f"[{n}] {c.citation()} | {c.type} | score={h.score:.3f} ranks={h.ranks}\n     {preview}")

    if not args.no_llm:
        from finrag.generate import answer
        from finrag.llm import get_client
        out = answer(get_client(), args.question, hits)
        print(f"\nAnswer:\n{out['answer']}\n\nSources:")
        for c in out["citations"]:
            print(f"  [{c['n']}] {c['doc']}, p.{c['page']} ({c['type']})")


if __name__ == "__main__":
    main()
