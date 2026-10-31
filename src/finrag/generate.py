"""Grounded answer generation with page citations."""
import re

from .llm import chat, settings

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about Tesla's 2023 quarterly updates.
Rules:
1. Use ONLY the numbered sources provided. Do not use outside knowledge.
2. Cite every fact with its source number in square brackets, e.g. [2].
3. Quote numbers exactly as written in the sources, with units and the period they refer to.
4. Sources marked "chart" are descriptions of charts produced by a vision model; their values may be approximate, so say "approximately" when you use them.
5. If the sources do not contain the answer, reply exactly: "I could not find this in the provided documents."
Keep answers short (1-4 sentences)."""

NOT_FOUND = "I could not find this in the provided documents."


def build_context(retrieved):
    blocks = []
    for n, r in enumerate(retrieved, start=1):
        c = r.chunk
        blocks.append(f"[{n}] ({c.doc_label}, page {c.page}, {c.type})\n{c.text}")
    return "\n\n".join(blocks)


def parse_citations(text, n_sources):
    """Source numbers cited as [2], [1, 3] or [1][2]; out-of-range numbers are dropped."""
    nums = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        nums.update(int(n) for n in group.split(","))
    return sorted(n for n in nums if 1 <= n <= n_sources)


def _with_citations(text, retrieved):
    return {
        "answer": text,
        "citations": [
            {"n": n, "doc": retrieved[n - 1].chunk.doc_label, "page": retrieved[n - 1].chunk.page,
             "type": retrieved[n - 1].chunk.type}
            for n in parse_citations(text, len(retrieved))
        ],
    }


def answer(client, question, retrieved, model=None):
    """Answer one question from its retrieved chunks (the path the API and CLI use)."""
    model = model or settings()["chat_model"]
    user = f"Sources:\n\n{build_context(retrieved)}\n\nQuestion: {question}"
    text = chat(client, model, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])
    return _with_citations(text.strip(), retrieved)
