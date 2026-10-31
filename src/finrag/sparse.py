"""Sparse (lexical) retrieval with BM25."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

STOPWORDS = set("""
a an and are as at be by did do does for from had has have how in is it its of on or
our so than that the their there these this to was were what when which who why will
with we you your
""".split())

# Keeps numbers such as 7.6%, 51,105 and $23.4b together as single tokens.
TOKEN = re.compile(r"\$?\d+(?:[.,]\d+)*%?|[a-z]+(?:[/'-][a-z0-9]+)*")


def tokenize(text):
    text = text.lower()
    tokens = []
    for tok in TOKEN.findall(text):
        tok = tok.lstrip("$")
        if tok[:1].isdigit():
            tok = tok.replace(",", "")      # "51,105" and "51105" should match
        if tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


class BM25Retriever:
    def __init__(self, texts, k1=config.BM25_K1, b=config.BM25_B):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query, k):
        """Return [(position, bm25_score)] for the top-k chunks with score > 0."""
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
