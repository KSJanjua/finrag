"""Second-stage reranking with a cross-encoder.

A bi-encoder embeds query and chunk separately and compares vectors - fast,
but the two texts never "see" each other. A cross-encoder feeds the pair
"[CLS] query [SEP] chunk" through one transformer, so attention runs across
both texts and the relevance score is much more accurate. It is too slow to
run on every chunk, so it only reorders the top candidates from stage one.
"""
from . import config


class CrossEncoderReranker:
    def __init__(self, model_name=config.RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def score(self, query, texts):
        if not texts:
            return []
        # Every pair in a batch is padded to the batch's longest pair (up to 512
        # tokens), so one long table chunk would make 31 short chunks pay for 512
        # tokens too. Small batches cut that waste: identical scores, ~25-30% faster on
        # CPU (results/reranker_sweep.md: 1.5 s vs 2.1 s per query with batch 32).
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs, batch_size=config.RERANK_BATCH_SIZE)]
