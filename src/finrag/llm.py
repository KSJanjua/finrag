"""LLM access through the OpenAI-compatible Chat Completions API.

Google Gemini (free tier), OpenAI and local servers such as Ollama all speak
this protocol, so one client works for every provider - only the base URL,
API key and model names change. Pick one with FINRAG_LLM_PROVIDER:

    gemini  (default, free)  key: GEMINI_API_KEY   https://aistudio.google.com/apikey
    openai  (paid)           key: OPENAI_API_KEY
    custom  any OpenAI-compatible server: FINRAG_BASE_URL (+ FINRAG_API_KEY if needed)
"""
import json
import os
import re
import time

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        # flash-lite has by far the largest free-tier daily quota, which bulk jobs
        # (chart descriptions, evaluation runs) need. The judge uses a different
        # model so answers are never graded by the model that wrote them.
        "chat_model": "gemini-3.1-flash-lite",
        "vision_model": "gemini-3.1-flash-lite",
        "judge_model": "gemini-3.8-flash",
        # Gemini 3 models always "think"; thinking tokens count against a token cap,
        # so a small cap can leave an empty answer. Keep thinking low and don't cap.
        "extra": {"reasoning_effort": "low"},
        "token_cap": False,
        "min_interval_s": 4.0,   # stay under free-tier requests-per-minute limits
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "chat_model": "gpt-4o-mini",
        "vision_model": "gpt-4o",
        "judge_model": "gpt-4o-mini",
        "extra": {"temperature": 0},
        "token_cap": True,
        "min_interval_s": 0.0,
    },
    "custom": {
        "base_url": None,          # taken from FINRAG_BASE_URL
        "key_env": "FINRAG_API_KEY",
        "chat_model": None,        # set FINRAG_CHAT_MODEL / FINRAG_VISION_MODEL
        "vision_model": None,
        "judge_model": None,
        "extra": {"temperature": 0},
        "token_cap": True,
        "min_interval_s": 0.0,
    },
}


def provider_name():
    name = os.getenv("FINRAG_LLM_PROVIDER", "gemini").lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"FINRAG_LLM_PROVIDER must be one of {list(PROVIDERS)}, got {name!r}")
    return name


def settings():
    """Resolved provider settings; FINRAG_* environment variables override defaults."""
    name = provider_name()
    p = dict(PROVIDERS[name], name=name)
    for role in ("chat", "vision", "judge"):
        p[f"{role}_model"] = os.getenv(f"FINRAG_{role.upper()}_MODEL") or p[f"{role}_model"]
    p["base_url"] = os.getenv("FINRAG_BASE_URL") or p["base_url"]
    p["min_interval_s"] = float(os.getenv("FINRAG_MIN_INTERVAL", p["min_interval_s"]))
    return p


def llm_configured():
    p = settings()
    if p["name"] == "custom":
        return bool(p["base_url"] and p["chat_model"])
    return bool(os.getenv(p["key_env"]))


def get_client():
    p = settings()
    key = os.getenv(p["key_env"]) if p["key_env"] else None
    if p["name"] == "custom":
        if not (p["base_url"] and p["chat_model"]):
            raise RuntimeError("custom provider needs FINRAG_BASE_URL and FINRAG_CHAT_MODEL")
        key = key or "not-needed"   # local servers such as Ollama ignore the key
    elif not key:
        raise RuntimeError(
            f"{p['key_env']} is not set (provider: {p['name']}). Retrieval works without it, "
            "but chart description, answer generation and LLM-judge evaluation need it. "
            "See README > 'LLM setup (free)'."
        )
    from openai import OpenAI
    # max_retries: the SDK retries 429 (rate limit) and 5xx errors with exponential backoff.
    return OpenAI(api_key=key, base_url=p["base_url"], max_retries=6)


_last_call = [0.0]


def chat(client, model, messages, max_tokens=800):
    """One chat completion; returns the reply text ('' if the model returned none)."""
    p = settings()
    wait = p["min_interval_s"] - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    kwargs = dict(p["extra"])
    if p["token_cap"]:
        kwargs["max_completion_tokens"] = max_tokens
    resp = client.chat.completions.create(model=model, messages=messages, **kwargs)
    _last_call[0] = time.monotonic()
    return resp.choices[0].message.content or ""


def parse_json(text):
    """Parse a JSON object from a model reply, tolerating ```json fences or extra prose.

    JSON mode isn't supported the same way by every provider, so prompts ask
    for JSON and this function extracts it.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
