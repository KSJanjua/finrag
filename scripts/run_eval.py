"""Run the evaluation and write results/ tables.

    python scripts/run_eval.py                    # retrieval metrics (no API key needed)
    python scripts/run_eval.py --check-repro      # ... and compare with the previous run
    python scripts/run_eval.py --answers          # + answer correctness (needs an LLM key)
    python scripts/run_eval.py --answers --limit 5   # quick, cheap trial on 5 questions
    python scripts/run_eval.py --answers-only --answers   # skip the retrieval grid
    python scripts/run_eval.py --answers --judge  # + LLM-as-judge

Ablation grid:
  modality  : text-only | text+tables | multimodal (text+tables+charts)
  retriever : bm25 | dense | hybrid (RRF) | hybrid_rerank
  multimodal also runs with the query -> report filter (doc="auto").
"""
import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time

from finrag import config
from finrag.evaluate import (facts_correct, first_relevant_rank, judge_correct,
                             load_questions, retrieval_metrics)
from finrag.generate import NOT_FOUND
from finrag.llm import settings
from finrag.retriever import MODES, load_retriever

MODALITIES = {
    "text-only": {"text"},
    "text+tables": {"text", "table"},
    "multimodal": {"text", "table", "chart"},
}
QTYPES = ("text", "table", "chart")
# The configuration the CLI and API use by default.
PRODUCTION = ("multimodal", "hybrid+doc_filter", "hybrid", "auto")
METRICS_FILE = config.RESULTS_DIR / "retrieval_metrics.json"


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def eval_retrieval(retrievers, questions, k=10):
    results, latency, per_question = {}, {}, {}
    for modality, retriever in retrievers.items():
        variants = [(m, m, None) for m in MODES]
        if modality == "multimodal":
            variants += [("hybrid+doc_filter", "hybrid", "auto"),
                         ("hybrid_rerank+doc_filter", "hybrid_rerank", "auto")]
        for label, mode, doc in variants:
            ranks, t0 = [], time.perf_counter()
            for q in questions:
                hits = retriever.retrieve(q["question"], mode=mode, top_k=k, doc=doc)
                rank = first_relevant_rank(hits, q)
                ranks.append(rank)
                per_question[(modality, label, q["id"])] = {
                    "rank": rank, "top3": [h.chunk.chunk_id for h in hits[:3]]}
            key = f"{modality}|{label}"
            latency[key] = 1000 * (time.perf_counter() - t0) / len(questions)
            row = {"all": retrieval_metrics(ranks),
                   "ranks": {q["id"]: r for q, r in zip(questions, ranks)}}
            for t in QTYPES:
                row[t] = retrieval_metrics([r for r, q in zip(ranks, questions) if q["type"] == t])
            results[key] = row
            print(f"{modality:12s} {label:25s} R@5={row['all']['recall@5']:.3f} "
                  f"MRR={row['all']['mrr']:.3f} ({latency[key]:.0f} ms/q)")
    return results, latency, per_question


def retrieval_report(results, latency, questions):
    n_by_type = {t: sum(q["type"] == t for q in questions) for t in QTYPES}
    lines = ["# Retrieval evaluation", "",
             f"{len(questions)} questions ({', '.join(f'{n_by_type[t]} {t}' for t in QTYPES)}). "
             "A hit = a chunk from a gold page that contains the answer evidence. "
             "`doc_filter` = restrict to the report named in the question (e.g. \"Q3 2023\"). "
             "Latency is CPU wall-clock per query and varies between runs; the metrics do not.", "",
             "## All questions", "",
             "| Modality | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms/query |",
             "|---|---|---|---|---|---|---|---|"]
    for key, row in results.items():
        modality, label = key.split("|")
        a = row["all"]
        lines.append(f"| {modality} | {label} | {fmt(a['recall@1'])} | {fmt(a['recall@3'])} | "
                     f"{fmt(a['recall@5'])} | {fmt(a['recall@10'])} | {fmt(a['mrr'])} | {latency[key]:.0f} |")
    lines += ["", "## Recall@5 by question type", "",
              "| Modality | Retriever | text | table | chart |", "|---|---|---|---|---|"]
    for key, row in results.items():
        modality, label = key.split("|")
        lines.append(f"| {modality} | {label} | " + " | ".join(fmt(row[t]["recall@5"]) for t in QTYPES) + " |")
    return "\n".join(lines) + "\n"


def misses_report(per_question, questions):
    modality, label = PRODUCTION[:2]
    lines = [f"# Misses at k=5 ({modality}, {label})", "",
             "Questions whose first relevant chunk is not in the top 5 - start error analysis here.", ""]
    for q in questions:
        info = per_question[(modality, label, q["id"])]
        if info["rank"] is None or info["rank"] > 5:
            lines.append(f"- **{q['id']}** ({q['type']}) {q['question']}  \n"
                         f"  first relevant rank: {info['rank']}; top-3: {', '.join(info['top3'])}")
    if len(lines) == 4:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def check_reproducibility(results):
    """Compare per-question ranks with the previous run's metrics file."""
    if not METRICS_FILE.exists():
        print("repro check: no previous results to compare against")
        return True
    old = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    diffs = [k for k in results if old.get(k, {}).get("ranks") != results[k]["ranks"]]
    if diffs:
        print(f"repro check: FAILED - ranks differ for {diffs}")
        return False
    print(f"repro check: OK - identical per-question ranks for all {len(results)} configurations")
    return True


ANSWERS_FILE = config.RESULTS_DIR / "answer_records.json"
CORPORA = ("text-only", "multimodal")


def _save(records):
    ordered = sorted(records.values(), key=lambda r: (CORPORA.index(r["modality"]), r["order"]))
    ANSWERS_FILE.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    return ordered


def eval_answers(retrievers, questions, use_judge):
    """Generate (one question per request, the production path) and score answers
    for the text-only and multimodal corpora.

    Resumable: records are saved after every request and a re-run reuses answers
    from the same model, so a free-tier daily quota can spread the run over days.
    """
    from finrag.generate import answer
    from finrag.llm import get_client
    client = get_client()
    s = settings()
    chat_model, judge_model = s["chat_model"], s["judge_model"]
    _, _, mode, doc = PRODUCTION
    order = {q["id"]: i for i, q in enumerate(questions)}
    by_id = {q["id"]: q for q in questions}
    previous = json.loads(ANSWERS_FILE.read_text(encoding="utf-8")) if ANSWERS_FILE.exists() else []
    records = {(r["modality"], r["id"]): r for r in previous
               if r.get("model") == chat_model and r["id"] in by_id}

    for modality in CORPORA:
        pending = [q for q in questions if (modality, q["id"]) not in records]
        for i, q in enumerate(pending, 1):
            hits = retrievers[modality].retrieve(q["question"], mode=mode, doc=doc)
            out = answer(client, q["question"], hits)
            records[(modality, q["id"])] = {
                "modality": modality, "id": q["id"], "order": order[q["id"]], "type": q["type"],
                "model": chat_model, "question": q["question"], "answer": out["answer"],
                "citations": out["citations"], "fact_match": facts_correct(out["answer"], q["answer_facts"])}
            _save(records)
            print(f"  {modality} {i}/{len(pending)} {q['id']} fact_match={records[(modality, q['id'])]['fact_match']}")

    if use_judge:  # ideally a different model than the generator
        for r in records.values():
            if r.get("judge_model") != judge_model:
                r["judge"], r["judge_reason"] = judge_correct(
                    client, r["question"], r["answer"], by_id[r["id"]]["answer_facts"])
                r["judge_model"] = judge_model
                _save(records)

    ordered = _save(records)
    summary = {}
    for modality in CORPORA:
        mine = [r for r in ordered if r["modality"] == modality]
        row = {"n": len(mine), "fact_match": mean(r["fact_match"] for r in mine)}
        for t in QTYPES:
            row[f"fact_match_{t}"] = mean(r["fact_match"] for r in mine if r["type"] == t)
        if use_judge:
            row["judge"] = mean(r["judge"] for r in mine)
        row["abstained"] = mean(NOT_FOUND in r["answer"] for r in mine)  # may add an explanation after it
        summary[modality] = row
        print(modality, {k: (fmt(v) if k != "n" else v) for k, v in row.items()})
    return ordered, summary


def answers_report(summary, use_judge):
    s = settings()
    lines = [f"# Answer correctness ({PRODUCTION[2]}, doc filter={PRODUCTION[3]}, top-{config.FINAL_K} context)", "",
             f"- Generator: `{s['chat_model']}`, one question per request (the production path)",
             "- Fact match: every required fact appears in the answer (numbers matched whole)",
             f"- LLM judge: `{s['judge_model']}`" if use_judge else "- LLM judge: not run",
             f"- Abstained: answer contains \"{NOT_FOUND}\"", "",
             "| Corpus | n | Fact match (all) | text | table | chart |" + (" LLM judge |" if use_judge else "") + " Abstained |",
             "|---|---|---|---|---|---|" + ("---|" if use_judge else "") + "---|"]
    for modality, row in summary.items():
        line = (f"| {modality} | {row['n']} | {fmt(row['fact_match'])} | {fmt(row['fact_match_text'])} | "
                f"{fmt(row['fact_match_table'])} | {fmt(row['fact_match_chart'])} |")
        if use_judge:
            line += f" {fmt(row['judge'])} |"
        line += f" {fmt(row['abstained'])} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also evaluate generated answers")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scoring (with --answers)")
    ap.add_argument("--check-repro", action="store_true", help="compare ranks with the previous run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N questions (e.g. to try --answers on a free tier first)")
    ap.add_argument("--answers-only", action="store_true", help="skip the retrieval grid")
    args = ap.parse_args()

    questions = load_questions()[:args.limit]
    full = load_retriever(use_reranker=True)
    retrievers = {name: full.subset(types) for name, types in MODALITIES.items()}

    config.RESULTS_DIR.mkdir(exist_ok=True)
    reproducible = True
    if args.answers_only:
        pass
    elif args.limit:  # partial run: don't overwrite the full results or the repro baseline
        eval_retrieval(retrievers, questions)
        print("--limit given: retrieval results printed above, not written to results/")
    else:
        results, latency, per_question = eval_retrieval(retrievers, questions)
        reproducible = check_reproducibility(results) if args.check_repro else True
        (config.RESULTS_DIR / "retrieval_results.md").write_text(
            retrieval_report(results, latency, questions), encoding="utf-8")
        (config.RESULTS_DIR / "retrieval_misses.md").write_text(
            misses_report(per_question, questions), encoding="utf-8")
        METRICS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {config.RESULTS_DIR / 'retrieval_results.md'}")

    if args.answers:
        _, summary = eval_answers(retrievers, questions, args.judge)
        (config.RESULTS_DIR / "answer_results.md").write_text(answers_report(summary, args.judge), encoding="utf-8")
        print(f"wrote {config.RESULTS_DIR / 'answer_results.md'}")

    sys.exit(0 if reproducible else 1)


if __name__ == "__main__":
    main()
