"""Check that the LLM provider is set up correctly (run this first after adding a key).

    python scripts/check_llm.py

1. shows the provider, base URL and models in use
2. lists the models your key can access (to spot renamed / retired models)
3. sends one short text request
4. sends one vision request with a real page (the Model Y price map, Q3 p.12)
"""
import _bootstrap  # noqa: F401

import sys

from finrag import config
from finrag.charts import describe_page, render_page_png
from finrag.llm import chat, get_client, settings


def main():
    s = settings()
    print(f"provider : {s['name']}")
    print(f"base_url : {s['base_url'] or 'https://api.openai.com/v1'}")
    print(f"models   : chat={s['chat_model']}  vision={s['vision_model']}  judge={s['judge_model']}")
    try:
        client = get_client()
    except RuntimeError as exc:
        print(f"\nNOT CONFIGURED: {exc}")
        sys.exit(1)

    try:
        ids = sorted(m.id.removeprefix("models/") for m in client.models.list())
        flash = [i for i in ids if "flash" in i or "gpt" in i][:15]
        print(f"\n{len(ids)} models available to this key, e.g.: {', '.join(flash or ids[:15])}")
        for role in ("chat_model", "vision_model"):
            if s[role] not in ids:
                print(f"WARNING: {role} '{s[role]}' is not in the list - set FINRAG_{role.split('_')[0].upper()}_MODEL")
    except Exception as exc:  # listing is optional; some servers don't support it
        print(f"\n(could not list models: {exc})")

    try:
        reply = chat(client, s["chat_model"], [{"role": "user", "content": "Reply with exactly: OK"}])
        print(f"\ntext test   : {reply.strip()[:80]!r}")
    except Exception as exc:
        print(f"\ntext test FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)

    pdf = config.PDF_DIR / "TSLA-Q3-2023-Update-3.pdf"
    if not pdf.exists():
        print("vision test : skipped (run scripts/build_index.py first to download the PDFs)")
        return
    try:
        charts = describe_page(client, render_page_png(pdf, 12))
        print(f"vision test : {len(charts)} chart(s) found on Q3 p.12")
        for c in charts:
            print(f"  - {c[:300]}")
    except Exception as exc:
        print(f"vision test FAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print("\nAll checks passed - next: python scripts/describe_charts.py")


if __name__ == "__main__":
    main()
