"""Parse the PDFs, build typed chunks, embed them and save the index.

    python scripts/build_index.py            # downloads PDFs if missing
"""
import _bootstrap  # noqa: F401

import time
import urllib.request
from collections import Counter

from finrag import config
from finrag.dense import Embedder
from finrag.ingest import build_chunks
from finrag.retriever import save_index


def download_pdfs():
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in config.PDF_URLS.items():
        path = config.PDF_DIR / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)


def main():
    download_pdfs()
    t0 = time.time()
    chunks = build_chunks()
    counts = Counter(c.type for c in chunks)
    print(f"built {len(chunks)} chunks {dict(counts)} in {time.time() - t0:.1f}s")
    if not counts.get("chart"):
        print("note: no chart descriptions found - run scripts/describe_charts.py first")

    t0 = time.time()
    embeddings = Embedder().encode_docs([c.index_text() for c in chunks])
    print(f"embedded with {config.EMBED_MODEL} -> {embeddings.shape} in {time.time() - t0:.1f}s")

    save_index(chunks, embeddings)
    print(f"saved index to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
