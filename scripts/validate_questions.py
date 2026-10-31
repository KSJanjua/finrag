"""Sanity-check the evaluation set against the actual chunks.

For each question: does at least one chunk on a gold page contain the evidence?
Also lists *other* pages that contain the evidence, so missing gold pages can
be added. A broken label would silently corrupt every metric, so run this
after editing questions.json or re-generating chart descriptions.
"""
import _bootstrap  # noqa: F401

from finrag.evaluate import contains, is_relevant, load_questions
from finrag.retriever import load_chunks


def main():
    chunks = load_chunks()
    problems = 0
    for q in load_questions():
        hits = [c for c in chunks if is_relevant(c, q)]
        elsewhere = sorted({(c.doc[:7], c.page, c.type) for c in chunks
                            if (c.doc, c.page) not in q["gold_pages"]
                            and any(contains(c.text, e) for e in q["evidence"])})
        status = "OK " if hits else "MISSING"
        problems += not hits
        types = sorted({c.type for c in hits})
        print(f"{status} {q['id']} ({q['type']}) evidence in: {types}")
        if elsewhere:
            print(f"     also found on non-gold pages: {elsewhere[:6]}")
    print(f"\n{problems} question(s) without evidence on a gold page.")


if __name__ == "__main__":
    main()
