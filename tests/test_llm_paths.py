"""LLM code paths, tested through the real OpenAI SDK with a mocked HTTP transport.

No API key or network is used: the SDK builds the real request, our handler
checks it and returns a canned chat-completion response, and the SDK parses it.
The same client code serves Gemini (free), OpenAI and custom servers.
"""
import base64
import json

import pytest
from openai import OpenAI

try:  # openai>=3 ships its own fork of httpx; older versions use httpx itself
    import httpx2 as hx
except ImportError:  # pragma: no cover
    import httpx as hx

from finrag import llm
from finrag.charts import describe_page
from finrag.evaluate import judge_correct
from finrag.generate import NOT_FOUND, answer
from finrag.schema import Chunk, RetrievedChunk


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    for var in ("FINRAG_LLM_PROVIDER", "FINRAG_CHAT_MODEL", "FINRAG_VISION_MODEL",
                "FINRAG_JUDGE_MODEL", "FINRAG_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FINRAG_MIN_INTERVAL", "0")


def fake_client(reply, seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return hx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
        })
    return OpenAI(api_key="test-key", http_client=hx.Client(transport=hx.MockTransport(handler)))


def hit(n, page, type_="text", text="Operating margin was 7.6% in Q3."):
    return RetrievedChunk(Chunk(f"id{n}", "TSLA-Q3-2023-Update-3.pdf", "Tesla Q3 2023 Quarterly Update",
                                page, type_, text), score=1.0)


# --------------------------------------------------------------------------
# Provider configuration
# --------------------------------------------------------------------------
def test_gemini_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = llm.settings()
    assert s["name"] == "gemini" and "generativelanguage.googleapis.com" in s["base_url"]
    assert not llm.llm_configured()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.get_client()


def test_gemini_client_uses_compat_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    client = llm.get_client()
    assert "generativelanguage.googleapis.com/v1beta/openai" in str(client.base_url)
    assert llm.llm_configured()


def test_env_overrides_models(monkeypatch):
    monkeypatch.setenv("FINRAG_CHAT_MODEL", "gemini-3.6-flash")
    assert llm.settings()["chat_model"] == "gemini-3.6-flash"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("FINRAG_LLM_PROVIDER", "nope")
    with pytest.raises(RuntimeError):
        llm.settings()


def test_custom_provider_needs_url_and_model(monkeypatch):
    monkeypatch.setenv("FINRAG_LLM_PROVIDER", "custom")
    with pytest.raises(RuntimeError, match="FINRAG_BASE_URL"):
        llm.get_client()
    monkeypatch.setenv("FINRAG_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("FINRAG_CHAT_MODEL", "some-local-model")
    assert "localhost:11434" in str(llm.get_client().base_url)


@pytest.mark.parametrize("provider, expect_present, expect_absent", [
    ("gemini", {"reasoning_effort": "low"}, ["max_completion_tokens", "temperature"]),
    ("openai", {"temperature": 0, "max_completion_tokens": 800}, ["reasoning_effort"]),
])
def test_request_parameters_per_provider(monkeypatch, provider, expect_present, expect_absent):
    monkeypatch.setenv("FINRAG_LLM_PROVIDER", provider)
    seen = []
    llm.chat(fake_client("ok", seen), "m", [{"role": "user", "content": "hi"}])
    for k, v in expect_present.items():
        assert seen[0][k] == v
    for k in expect_absent:
        assert k not in seen[0]
    assert "response_format" not in seen[0]   # JSON is parsed from text for every provider


@pytest.mark.parametrize("text", [
    '{"charts": ["a"]}',
    '```json\n{"charts": ["a"]}\n```',
    'Here you go:\n{"charts": ["a"]}\nHope this helps.',
])
def test_parse_json_tolerates_fences_and_prose(text):
    assert llm.parse_json(text) == {"charts": ["a"]}


# --------------------------------------------------------------------------
# Generation, chart description, judge
# --------------------------------------------------------------------------
def test_answer_builds_grounded_prompt_and_maps_citations():
    seen = []
    client = fake_client("The operating margin was 7.6% [2].", seen)
    out = answer(client, "Operating margin Q3 2023?", [hit(1, 5), hit(2, 3, "table")])

    request = seen[0]
    assert request["model"] == llm.settings()["chat_model"]
    assert request["messages"][0]["role"] == "system" and "ONLY" in request["messages"][0]["content"]
    user = request["messages"][1]["content"]
    assert "[1] (Tesla Q3 2023 Quarterly Update, page 5, text)" in user
    assert "[2] (Tesla Q3 2023 Quarterly Update, page 3, table)" in user
    assert out["citations"] == [{"n": 2, "doc": "Tesla Q3 2023 Quarterly Update", "page": 3, "type": "table"}]


def test_answer_abstention_has_no_citations():
    out = answer(fake_client(NOT_FOUND, []), "Unknown?", [hit(1, 5)])
    assert out["answer"] == NOT_FOUND and out["citations"] == []


def test_describe_page_sends_image_and_parses_json():
    seen = []
    client = fake_client('```json\n{"charts": ["Bar chart: deliveries rising", {"title": "map"}, ""]}\n```', seen)
    charts = describe_page(client, b"\x89PNG fake bytes")

    content = seen[0]["messages"][0]["content"]
    image = [c for c in content if c["type"] == "image_url"][0]["image_url"]["url"]
    assert image == "data:image/png;base64," + base64.b64encode(b"\x89PNG fake bytes").decode()
    assert seen[0]["model"] == llm.settings()["vision_model"]
    assert charts == ["Bar chart: deliveries rising", '{"title": "map"}']


def test_describe_page_no_charts():
    assert describe_page(fake_client('{"charts": []}', []), b"png") == []


@pytest.mark.parametrize("reply, expected", [
    ('{"correct": true, "reason": "matches"}', True),
    ('```json\n{"correct": true, "reason": "ok"}\n```', True),
    ('{"correct": false, "reason": "wrong number"}', False),
    ('{"correct": "yes"}', False),      # only a real boolean true counts
    ("not json", False),
])
def test_judge_parsing(reply, expected):
    ok, reason = judge_correct(fake_client(reply, []), "q", "a", [["7.6%"]])
    assert ok is expected and isinstance(reason, str)
