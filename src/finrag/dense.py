"""Dense retrieval: bi-encoder embeddings + FAISS."""
import faiss
import numpy as np

from . import config


class Embedder:
    """Wraps a sentence-transformers bi-encoder.

    Documents and queries are encoded independently (that is what makes it a
    bi-encoder), so document vectors can be pre-computed and indexed.
    """

    def __init__(self, model_name=config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_docs(self, texts, batch_size=32):
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 64)
        return np.asarray(emb, dtype="float32")

    def encode_query(self, query):
        emb = self.model.encode([config.QUERY_PREFIX + query], normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")


def make_index(embeddings):
    """Exact inner-product index over L2-normalised vectors (inner product == cosine).

    With ~200 chunks, brute-force search takes well under a millisecond and has
    perfect recall, so an approximate index (HNSW / IVF / PQ) would only lose
    accuracy. scripts/faiss_benchmark.py shows when those become worthwhile.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index


class DenseRetriever:
    def __init__(self, embeddings, embedder):
        self.embedder = embedder
        self.index = make_index(embeddings)

    def search(self, query, k):
        """Return [(position, cosine_score)] for the top-k chunks."""
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(self.embedder.encode_query(query), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
