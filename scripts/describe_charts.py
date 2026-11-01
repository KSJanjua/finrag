"""Describe the charts on each PDF page with a vision LLM and cache the results.

    python scripts/describe_charts.py                  # candidate pages not yet described by this model
    python scripts/describe_charts.py --all-pages      # skip the chart-candidate filter
    python scripts/describe_charts.py --force          # redo pages even if this model described them
    python scripts/describe_charts.py --max-pages 15   # cap the number of API calls for this run
    python scripts/describe_charts.py --pages 8 12 --doc TSLA-Q3-2023-Update-3.pdf

A page's cache entries are replaced only when its new result arrives, so an
interrupted run (e.g. a free-tier quota error) can simply be restarted.
Afterwards, rebuild the index with build_index.py and re-check the evaluation
labels with validate_questions.py, since chart wording may change.
"""
import _bootstrap  # noqa: F401

import argparse

import fitz
from tqdm import tqdm

from finrag import config
from finrag.charts import chart_candidate, describe_page, load_cache, render_page_png, save_cache
from finrag.llm import get_client, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--doc", default=None)
    ap.add_argument("--pages", type=int, nargs="*")
    ap.add_argument("--all-pages", action="store_true", help="skip the chart-candidate filter")
    ap.add_argument("--max-pages", type=int, default=None, help="describe at most N pages this run")
    args = ap.parse_args()

    client = get_client()
    vision_model = settings()["vision_model"]
    cache = load_cache()
    jobs = []
    for pdf_path in sorted(config.PDF_DIR.glob("*.pdf")):
        if args.doc and pdf_path.name != args.doc:
            continue
        with fitz.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                if args.pages and page_no not in args.pages:
                    continue
                reason = "requested" if args.all_pages else chart_candidate(page)
                if reason:
                    jobs.append((pdf_path, page_no, reason))

    if not args.force:
        # Pages described by a different model are redone, so switching models keeps
        # every chart chunk consistent, and a restarted run picks up where it stopped.
        done = {(e["doc"], e["page"]) for e in cache if e.get("source") == vision_model}
        jobs = [j for j in jobs if (j[0].name, j[1]) not in done]
    # Pages with chart axes first: they are the most likely to hold chart-only facts.
    jobs.sort(key=lambda j: j[2] != "axis-ticks")
    jobs = jobs[:args.max_pages]
    print(f"{len(jobs)} pages to describe with {vision_model}")

    failures = 0
    for pdf_path, page_no, _reason in tqdm(jobs):
        try:
            charts = describe_page(client, render_page_png(pdf_path, page_no))
            failures = 0
        except Exception as exc:  # keep going; one bad page shouldn't stop the run
            failures += 1
            print(f"skip {pdf_path.name} p{page_no}: {type(exc).__name__}: {str(exc)[:200]}")
            if failures >= 3:  # e.g. free-tier daily quota used up
                print("3 failures in a row - stopping. Progress is saved; run again later to resume.")
                break
            continue
        # Replace this page's old entries only now that the new result has arrived.
        cache = [e for e in cache if (e["doc"], e["page"]) != (pdf_path.name, page_no)]
        # A null description records "checked, no charts" so the page isn't re-sent next run.
        for desc in charts or [None]:
            cache.append({"doc": pdf_path.name, "page": page_no,
                          "description": desc, "source": vision_model})
        save_cache(cache)  # save as we go so an interruption loses nothing
    n_charts = sum(1 for e in cache if e.get("description"))
    print(f"cache now has {n_charts} chart descriptions -> {config.CHART_CACHE}")


if __name__ == "__main__":
    main()
