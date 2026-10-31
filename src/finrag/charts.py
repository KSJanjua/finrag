"""Chart -> text with a vision LLM.

Charts in investor decks often hold facts that never appear in the text layer
(e.g. the Model Y price map, the Dojo compute projection). We render each page
to an image, ask a vision LLM (default: Gemini) to describe every chart on it,
and store the descriptions as ordinary text chunks so BM25 and FAISS can find
them. Results are cached per (document, page) so the API is only called once.

To save API quota, only *candidate* pages are sent (see chart_candidate). On 19
pages that were also scanned without the filter, it kept all 4 pages that had
charts.
"""
import base64
import json

import fitz

from . import config
from .ingest import NUMERIC_LINE, extract_tables, page_text_outside
from .llm import chat, parse_json, settings

TEMPLATE_IMAGES = 4   # every page of these decks carries 4 decorative template images

PROMPT = """You are analysing one page of a company's quarterly investor update.
Find every chart, graph, or map on the page (ignore plain tables and photos).
For each one, write a self-contained description that includes:
- the title and chart type
- axes / legend / series names and the time period covered
- the specific values you can read (approximate if needed, e.g. "~$39,000")
- the main trend or takeaway
Return JSON only, no markdown: {"charts": ["<description 1>", "<description 2>"]}.
If the page has no charts, return {"charts": []}."""


def chart_candidate(page, min_ticks=4):
    """Why this page might contain a chart: "axis-ticks", "images" or None.

    - axis-ticks: a run of numbers-only lines outside tables, i.e. the tick
      labels of a vector chart's axis (0 / 100 / 200 / ...)
    - images: raster images beyond the template's, e.g. a map or chart screenshot
    """
    text = page_text_outside(page, [r for r, _ in extract_tables(page)])
    run = 0
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        run = run + 1 if NUMERIC_LINE.match(line) else 0
        if run >= min_ticks:
            return "axis-ticks"
    if len(page.get_images()) > TEMPLATE_IMAGES:
        return "images"
    return None


def render_page_png(pdf_path, page_no, dpi=150):
    with fitz.open(pdf_path) as pdf:
        return pdf[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def describe_page(client, png_bytes, model=None):
    model = model or settings()["vision_model"]
    image_b64 = base64.b64encode(png_bytes).decode()
    content = chat(client, model, [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }], max_tokens=1200)
    charts = parse_json(content).get("charts", [])
    return [c if isinstance(c, str) else json.dumps(c) for c in charts if c]


def load_cache(path=config.CHART_CACHE):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_cache(entries, path=config.CHART_CACHE):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: (e["doc"], e["page"]))
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
