"""Exact vs approximate FAISS indexes on a larger synthetic corpus.

The project corpus has ~200 chunks, so exact IndexFlatIP search is the right
choice (sub-millisecond, perfect recall). This script shows what changes at
scale: it builds Flat, HNSW, IVF-Flat and IVF-PQ indexes over N clustered
random vectors and reports build time, query latency, memory and Recall@10
against the exact result.

    python scripts/faiss_benchmark.py --n 100000
"""
import _bootstrap  # noqa: F401

import argparse
import time

import faiss
import numpy as np


def clustered_vectors(n, d, rng, n_clusters=200, latent_dim=32):
    """Clustered points on a low-dimensional subspace, projected to d dims.

    Real sentence embeddings have low intrinsic dimensionality; isotropic
    Gaussian noise in 384-d would make all neighbours nearly equidistant and
    is an unrealistically hard (and meaningless) test for ANN indexes.
    """
    centers = rng.normal(size=(n_clusters, latent_dim))
    labels = rng.integers(0, n_clusters, size=n)
    latent = centers[labels] + 0.5 * rng.normal(size=(n, latent_dim))
    x = (latent @ PROJECTION[:latent_dim, :d] + 0.01 * rng.normal(size=(n, d))).astype("float32")
    faiss.normalize_L2(x)
    return x


PROJECTION = np.random.default_rng(42).normal(size=(64, 1024))


def recall_at_k(found, truth, k):
    return np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--queries", type=int, default=500)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    n, d, k = args.n, args.d, 10

    xb = clustered_vectors(n, d, rng)
    xq = clustered_vectors(args.queries, d, rng)
    nlist = int(4 * np.sqrt(n))

    # (name, constructor, search-time knob, values to sweep)
    def set_ef(idx, v): idx.hnsw.efSearch = v
    def set_nprobe(idx, v): idx.nprobe = v
    configs = [
        ("Flat (exact)", lambda: faiss.IndexFlatIP(d), None, [None]),
        ("HNSW M=32", lambda: faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT),
         ("efSearch", set_ef), [16, 64, 256]),
        (f"IVF-Flat nlist={nlist}",
         lambda: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT),
         ("nprobe", set_nprobe), [4, 16, 64]),
        # m=48 sub-vectors of 8 dims, 8 bits each -> 48 bytes/vector vs 1536 for float32
        (f"IVF-PQ nlist={nlist} m=48",
         lambda: faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT),
         ("nprobe", set_nprobe), [4, 16, 64]),
    ]

    truth = None
    print(f"N={n:,} d={d} queries={args.queries}\n")
    print(f"{'index':28s} {'knob':>14s} {'build s':>8s} {'ms/query':>9s} {'MB':>7s} {'R@10':>6s}")
    for name, make, knob, values in configs:
        idx = make()
        t0 = time.perf_counter()
        if not idx.is_trained:  # IVF k-means / PQ codebooks are learned from a sample
            idx.train(xb[rng.choice(n, size=min(n, 50_000), replace=False)])
        idx.add(xb)
        build = time.perf_counter() - t0
        mb = len(faiss.serialize_index(idx)) / 1e6
        for v in values:
            if knob:
                knob[1](idx, v)
            t0 = time.perf_counter()
            _, ids = idx.search(xq, k)
            ms = 1000 * (time.perf_counter() - t0) / len(xq)
            if truth is None:
                truth = ids
            label = f"{knob[0]}={v}" if knob else "-"
            print(f"{name:28s} {label:>14s} {build:8.2f} {ms:9.3f} {mb:7.1f} "
                  f"{recall_at_k(ids, truth, k):6.3f}")


if __name__ == "__main__":
    main()
