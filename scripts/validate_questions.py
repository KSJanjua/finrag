"""Sanity-check the evaluation set against the actual chunks.

For every evidence set of every answerable question: does at least one chunk
on one of its gold pages contain the evidence? Also lists *other* pages that
contain the evidence, so missing gold pages can be added. A broken label would
silently corrupt every metric, so run this after editing questions.json or
re-generating chart descriptions.
"""
import _bootstrap  # noqa: F401

from collections import Counter

from finrag.evaluate import contains, load_questions, satisfied_sets
from finrag.retriever import load_chunks


def main():
    chunks = load_chunks()
    questions = load_questions()
    problems = 0
    for q in questions:
        if not q["sets"]:
            continue
        found = {i for c in chunks for i, _ in satisfied_sets(c, q)}
        for i, s in enumerate(q["sets"]):
            ok = i in found
            problems += not ok
            elsewhere = sorted({(c.doc[5:12], c.page, c.type) for c in chunks
                                if (c.doc, c.page) not in s["pages"]
                                and any(contains(c.text, e) for e in s["evidence"])})
            label = f"{q['id']}" + (f"[{i}]" if len(q["sets"]) > 1 else "")
            print(f"{'OK     ' if ok else 'MISSING'} {label:9s} ({q['type']})"
                  + (f"  also on non-gold pages: {elsewhere[:5]}" if elsewhere else ""))
    counts = Counter(q["type"] for q in questions)
    print(f"\n{len(questions)} questions {dict(counts)}; {problems} evidence set(s) without evidence on a gold page.")


if __name__ == "__main__":
    main()
