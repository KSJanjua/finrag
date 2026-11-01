"""Integration tests on the real index and models (skipped if the index isn't built)."""
import pytest
from fastapi.testclient import TestClient

from finrag import config
from finrag.evaluate import load_questions, satisfied_sets
from finrag.retriever import load_chunks


def test_index_has_all_modalities():
    if not config.CHUNKS_FILE.exists():
        pytest.skip("index not built")
    chunks = load_chunks()
    types = {c.type for c in chunks}
    assert types == {"text", "table", "chart"}
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk ids must be unique"
    assert all(c.page >= 1 and c.doc in config.DOC_LABELS for c in chunks)


def test_every_question_has_evidence_in_the_index():
    if not config.CHUNKS_FILE.exists():
        pytest.skip("index not built")
    chunks = load_chunks()
    missing = []
    for q in load_questions():
        found = {i for c in chunks for i, _ in satisfied_sets(c, q)}
        missing += [f"{q['id']}[{i}]" for i in range(len(q["sets"])) if i not in found]
    assert not missing, f"evidence sets without evidence: {missing}"


def test_chart_only_fact_is_retrieved_first(retriever):
    hits = retriever.retrieve("What is the Model Y starting price in Colorado after subsidies?", top_k=3)
    top = hits[0].chunk
    assert (top.type, top.page) == ("chart", 12) and "31,490" in top.text


def test_doc_filter_picks_the_named_report(retriever):
    q = "How many Supercharger connectors did Tesla have at the end of Q3 2023?"
    hits = retriever.retrieve(q, doc="auto", top_k=3)
    assert all(h.chunk.doc == "TSLA-Q3-2023-Update-3.pdf" for h in hits)
    assert "51,105" in hits[0].chunk.text


def test_text_only_subset_contains_no_charts(retriever):
    sub = retriever.subset({"text"})
    assert {c.type for c in sub.chunks} == {"text"}
    assert len(sub.chunks) == len(sub.embeddings)


def test_invalid_arguments(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="nope")
    with pytest.raises(ValueError):
        retriever.retrieve("x", doc="missing.pdf")


@pytest.fixture(scope="module")
def api(retriever, monkeypatch_module):
    import finrag.api as api_module
    monkeypatch_module.setattr(api_module, "load_retriever", lambda use_reranker=True: retriever)
    monkeypatch_module.delenv("FINRAG_LLM_PROVIDER", raising=False)   # default provider: gemini
    monkeypatch_module.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(api_module.app) as client:
        yield client


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_api_health(api):
    body = api.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0
    assert body["llm_provider"] == "gemini" and body["llm_configured"] is False


def test_api_search(api):
    r = api.post("/search", json={"question": "Supercharger connectors at the end of Q3 2023?", "top_k": 2})
    body = r.json()
    assert r.status_code == 200 and body["doc_filters"] == ["TSLA-Q3-2023-Update-3.pdf"]
    assert body["results"][0]["page"] == 6 and body["results"][0]["type"] == "table"


def test_api_validation(api):
    assert api.post("/search", json={"question": "abc", "mode": "nope"}).status_code == 400
    assert api.post("/search", json={"question": "abc", "doc": "missing.pdf"}).status_code == 400
    assert api.post("/search", json={"question": "ab"}).status_code == 422
    assert api.post("/search", json={"question": "abc", "top_k": 99}).status_code == 422


def test_api_ask_without_key_is_503(api):
    r = api.post("/ask", json={"question": "Operating margin in Q3 2023?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
