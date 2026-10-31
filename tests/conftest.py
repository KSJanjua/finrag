import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finrag import config  # noqa: E402


@pytest.fixture(scope="session")
def retriever():
    """The real index + models (loaded once). Skips if the index hasn't been built."""
    if not config.CHUNKS_FILE.exists():
        pytest.skip("index not built - run `python scripts/build_index.py`")
    from finrag.retriever import load_retriever
    return load_retriever(use_reranker=True)
