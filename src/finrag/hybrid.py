"""Reciprocal Rank Fusion (RRF).

BM25 scores (unbounded) and cosine scores (-1..1) are on different scales, so
instead of mixing scores we fuse *ranks*:

    RRF(d) = sum over retrievers r of 1 / (k + rank_r(d))

k (=60 in the original paper) damps the advantage of the very top ranks, so a
document ranked well by both retrievers beats one ranked first by only one.
"""
from . import config


def reciprocal_rank_fusion(ranked_lists, k=config.RRF_K):
    """ranked_lists: list of [doc_id, ...] best-first. Returns [(doc_id, score)]."""
    scores = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
