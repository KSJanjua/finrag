"""Run the evaluation and write results/ tables.

    python scripts/run_eval.py                       # retrieval grid (no API key needed)
    python scripts/run_eval.py --check-repro         # ... and compare ranks with the previous run
    python scripts/run_eval.py --answers-only --answers                # answers, production config
    python scripts/run_eval.py --answers-only --answers --text-only    # + text-only corpus for comparison
    python scripts/run_eval.py --answers-only --answers --judge        # + LLM-as-judge
    python scripts/run_eval.py --answers --limit 5   # quick trial on 5 questions

Retrieval grid:
  corpus    : text-only | text+tables | multimodal (text+tables+charts)
  retriever : bm25 | dense | hybrid (RRF) | hybrid_rerank
  the multimodal corpus also runs with the query -> report filter (doc="auto").
Unanswerable questions are only used for the answer evaluation.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import sys
import time

from finrag import config
from finrag.evaluate import (aggregate, facts_correct, ideal_grades, judge_correct,
                             load_questions, score_ranking, unsupported_numbers)
from finrag.generate import NOT_FOUND
from finrag.llm import settings
from finrag.retriever import MODES, load_retriever

MODALITIES = {
    "text-only": {"text"},
    "text+tables": {"text", "table"},
    "multimodal": {"text", "table", "chart"},
}
QTYPES = ("text", "table", "chart", "paraphrase", "multi")
# The configuration the CLI and API use by default.
PRODUCTION = ("multimodal", "hybrid+doc_filter", "hybrid", "auto")
METRICS_FILE = config.RESULTS_DIR / "retrieval_metrics.json"
ANSWERS_FILE = config.RESULTS_DIR / "answer_records.json"
TOP_K = 10


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def eval_retrieval(retrievers, questions):
    answerable = [q for q in questions if q["sets"]]
    results, latency, per_question = {}, {}, {}
    for modality, retriever in retrievers.items():
        ideal = {q["id"]: ideal_grades(retriever.chunks, q) for q in answerable}
        variants = [(m, m, None) for m in MODES]
        if modality == "multimodal":
            variants += [("hybrid+doc_filter", "hybrid", "auto"),
                         ("hybrid_rerank+doc_filter", "hybrid_rerank", "auto")]
        for label, mode, doc in variants:
            scores, t0 = {}, time.perf_counter()
            for q in answerable:
                hits = retriever.retrieve(q["question"], mode=mode, top_k=TOP_K, doc=doc)
                scores[q["id"]] = score_ranking(hits, q, ideal[q["id"]])
                per_question[(modality, label, q["id"])] = {
                    "rank": scores[q["id"]]["rank"], "top3": [h.chunk.chunk_id for h in hits[:3]]}
            key = f"{modality}|{label}"
            latency[key] = 1000 * (time.perf_counter() - t0) / len(answerable)
            row = {"all": aggregate(list(scores.values())),
                   "ranks": {qid: [s["rank"], s["recall@10"]] for qid, s in scores.items()}}
            for t in QTYPES:
                row[t] = aggregate([scores[q["id"]] for q in answerable if q["type"] == t])
            results[key] = row
            a = row["all"]
            print(f"{modality:12s} {label:25s} hit@5={a['hit@5']:.3f} recall@5={a['recall@5']:.3f} "
                  f"ndcg@10={a['ndcg@10']:.3f} MRR={a['mrr']:.3f} ({latency[key]:.0f} ms/q)")
    return results, latency, per_question


def retrieval_report(results, latency, questions):
    answerable = [q for q in questions if q["sets"]]
    counts = ", ".join(f"{sum(q['type'] == t for q in answerable)} {t}" for t in QTYPES)
    lines = ["# Retrieval evaluation", "",
             f"{len(answerable)} answerable questions ({counts}). A chunk is relevant if it comes from a gold page "
             "and contains the answer evidence; grade 2 = the period's own report, grade 1 = a later report "
             "repeating the figure. `recall@k` is the share of a question's evidence sets found in the top k "
             "(differs from `hit@k` only for multi-period questions). `doc_filter` = restrict to the report named "
             "in the question. Latency is CPU wall-clock per query and varies between runs; the metrics do not.", "",
             "## All answerable questions", "",
             "| Corpus | Retriever | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Precision@5 | NDCG@10 | MRR | ms/query |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, row in results.items():
        corpus, label = key.split("|")
        a = row["all"]
        lines.append(f"| {corpus} | {label} | {fmt(a['hit@1'])} | {fmt(a['hit@5'])} | {fmt(a['recall@5'])} | "
                     f"{fmt(a['recall@10'])} | {fmt(a['precision@5'])} | {fmt(a['ndcg@10'])} | {fmt(a['mrr'])} | "
                     f"{latency[key]:.0f} |")
    lines += ["", "## Hit@5 by question type (Recall@5 for multi-period)", "",
              "| Corpus | Retriever | " + " | ".join(QTYPES) + " |", "|---|---|" + "---|" * len(QTYPES)]
    for key, row in results.items():
        corpus, label = key.split("|")
        cells = [fmt(row[t].get("recall@5" if t == "multi" else "hit@5")) for t in QTYPES]
        lines.append(f"| {corpus} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def misses_report(per_question, questions):
    corpus, label = PRODUCTION[:2]
    lines = [f"# Misses at k=5 ({corpus}, {label})", "",
             "Answerable questions whose first relevant chunk is not in the top 5.", ""]
    for q in questions:
        info = per_question.get((corpus, label, q["id"]))
        if info and (info["rank"] is None or info["rank"] > 5):
            lines.append(f"- **{q['id']}** ({q['type']}) {q['question']}  \n"
                         f"  first relevant rank: {info['rank']}; top-3: {', '.join(info['top3'])}")
    if len(lines) == 4:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def check_reproducibility(results):
    """Compare per-question ranks and recall with the previous run's metrics file."""
    if not METRICS_FILE.exists():
        print("repro check: no previous results to compare against")
        return True
    old = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    diffs = [k for k in results if old.get(k, {}).get("ranks") != results[k]["ranks"]]
    if diffs:
        print(f"repro check: FAILED - ranks differ for {diffs}")
        return False
    print(f"repro check: OK - identical per-question results for all {len(results)} configurations")
    return True


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------
def _save(records):
    ordered = sorted(records.values(), key=lambda r: (r["modality"], r["order"]))
    ANSWERS_FILE.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    return ordered


def eval_answers(retrievers, questions, corpora, use_judge, context_k=config.FINAL_K):
    """Generate and score answers on the production path (one question per request).

    Resumable: records are saved after every request and a re-run reuses answers
    from the same model, so a free-tier daily quota can spread a run over days.
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
               if r.get("model") == chat_model and r.get("context_k") == context_k
               and r["id"] in by_id and r["modality"] in corpora}

    for corpus in corpora:
        pending = [q for q in questions if (corpus, q["id"]) not in records]
        for i, q in enumerate(pending, 1):
            hits = retrievers[corpus].retrieve(q["question"], mode=mode, top_k=context_k, doc=doc)
            out = answer(client, q["question"], hits)
            cited = [hits[c["n"] - 1].chunk.index_text() for c in out["citations"]]
            abstained = NOT_FOUND in out["answer"]
            records[(corpus, q["id"])] = {
                "modality": corpus, "id": q["id"], "order": order[q["id"]], "type": q["type"],
                "model": chat_model, "context_k": context_k, "question": q["question"], "answer": out["answer"],
                "citations": out["citations"], "abstained": abstained,
                "fact_match": facts_correct(out["answer"], q["answer_facts"]) if q["sets"] else None,
                "unsupported_numbers": [] if abstained else unsupported_numbers(out["answer"], cited)}
            _save(records)
            print(f"  {corpus} {i}/{len(pending)} {q['id']}")

    # Reused answers are rescored so that label corrections apply without new LLM calls.
    for r in records.values():
        q = by_id[r["id"]]
        r["fact_match"] = facts_correct(r["answer"], q["answer_facts"]) if q["sets"] else None

    if use_judge:  # ideally a different model than the generator
        for r in records.values():
            if r["fact_match"] is not None and r.get("judge_model") != judge_model:
                r["judge"], r["judge_reason"] = judge_correct(
                    client, r["question"], r["answer"], by_id[r["id"]]["answer_facts"])
                r["judge_model"] = judge_model
                _save(records)

    ordered = _save(records)
    summary = {}
    for corpus in corpora:
        mine = [r for r in ordered if r["modality"] == corpus]
        ans = [r for r in mine if r["fact_match"] is not None]
        una = [r for r in mine if r["fact_match"] is None]
        spoke = [r for r in mine if not r["abstained"]]
        row = {"n_answerable": len(ans), "n_unanswerable": len(una),
               "fact_match": mean(r["fact_match"] for r in ans),
               "false_refusal": mean(r["abstained"] for r in ans),
               "correct_refusal": mean(r["abstained"] for r in una),
               "unsupported_numbers": mean(bool(r["unsupported_numbers"]) for r in spoke)}
        for t in QTYPES:
            row[f"fact_match_{t}"] = mean(r["fact_match"] for r in ans if r["type"] == t)
        if use_judge:
            row["judge"] = mean(r["judge"] for r in ans if "judge" in r)
        summary[corpus] = row
        print(corpus, {k: (fmt(v) if isinstance(v, float) or v is None else v) for k, v in row.items()})
    return ordered, summary


def answers_report(summary, use_judge, context_k):
    s = settings()
    lines = [f"# Answer evaluation ({PRODUCTION[2]}, doc filter={PRODUCTION[3]}, top-{context_k} context)", "",
             f"- Generator: `{s['chat_model']}`, one question per request (the production path)",
             "- Fact match: every required fact appears in the answer (numbers matched whole)",
             f"- Refusal: the answer contains \"{NOT_FOUND}\"",
             "- Unsupported numbers: share of non-refusing answers with a number that none of its cited sources "
             "contains (rounding and $M/$B scale allowed)",
             f"- LLM judge: `{s['judge_model']}`" if use_judge else "- LLM judge: not run", "",
             "| Corpus | Fact match | False refusals | Correct refusals (unanswerable) | Unsupported numbers |"
             + (" LLM judge |" if use_judge else ""),
             "|---|---|---|---|---|" + ("---|" if use_judge else "")]
    for corpus, row in summary.items():
        line = (f"| {corpus} | {fmt(row['fact_match'])} | {fmt(row['false_refusal'])} | "
                f"{fmt(row['correct_refusal'])} | {fmt(row['unsupported_numbers'])} |")
        if use_judge:
            line += f" {fmt(row.get('judge'))} |"
        lines.append(line)
    lines += ["", "## Fact match by question type", "",
              "| Corpus | " + " | ".join(QTYPES) + " |", "|---|" + "---|" * len(QTYPES)]
    for corpus, row in summary.items():
        lines.append(f"| {corpus} | " + " | ".join(fmt(row[f'fact_match_{t}']) for t in QTYPES) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also evaluate generated answers")
    ap.add_argument("--text-only", action="store_true", help="also answer from the text-only corpus")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scoring (with --answers)")
    ap.add_argument("--check-repro", action="store_true", help="compare ranks with the previous run")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N questions (e.g. to try --answers on a free tier first)")
    ap.add_argument("--answers-only", action="store_true", help="skip the retrieval grid")
    ap.add_argument("--context-k", type=int, default=config.FINAL_K,
                    help="chunks given to the LLM per question (answer evaluation)")
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
        corpora = ["multimodal"] + (["text-only"] if args.text_only else [])
        _, summary = eval_answers(retrievers, questions, corpora, args.judge, args.context_k)
        (config.RESULTS_DIR / "answer_results.md").write_text(answers_report(summary, args.judge, args.context_k), encoding="utf-8")
        print(f"wrote {config.RESULTS_DIR / 'answer_results.md'}")

    sys.exit(0 if reproducible else 1)


if __name__ == "__main__":
    main()
