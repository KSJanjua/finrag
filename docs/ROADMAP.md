# Roadmap (v2)

v1 (tag `v1.0`) answers questions about three Tesla quarterly reports and measured its own
weaknesses. v2 extends it to more documents, harder questions, and the production concerns that
come with them.

Every feature has to fix a measured problem and move a metric on the v2 evaluation set.
Features that don't are recorded as rejected in [DECISIONS.md](DECISIONS.md).

## Milestones (in order)

| # | Milestone | Why it comes here | Proves |
|---|---|---|---|
| M0 | Re-run the v1 evaluation on the v2 code base | A refactor must start from known-good numbers | identical per-question ranks in all 14 configurations |
| M1 | Corpus: 11 Tesla quarterly updates (Q4 2022 – Q2 2025) with a document registry (company, fiscal year, quarter, URL) | Time is where the hard problems are (repeated tables, revised numbers, new quarters) | Ingestion works on ~300 pages, not 82 |
| M2 | **Evaluation v2 before any new feature**: ~100 questions incl. multi-period, unanswerable and re-worded ones; Precision@k, NDCG@k, context recall, abstention accuracy, number-grounding faithfulness | You can't claim an improvement without a baseline | The v1 pipeline's real failure modes at scale |
| M3 | Retrieval fixes, one at a time: token-aware chunking + parent–child tables (the 512-token truncation), query parser for periods (the "Q1, Q2 and Q3" bug), query decomposition for multi-period questions | These are the failures v1 already showed | Each one's metric delta |
| M4 | Context engineering + grounded generation: de-duplication, token budget, ordering, conflicting evidence, runtime citation check, corrective retry / early abstention | Better retrieval is wasted if the context or the answer is sloppy | Faithfulness and abstention numbers |
| M5 | Production: vector store with stable IDs, incremental indexing, caching, request tracing, latency/cost report | Only matters once retrieval is right | Adding a quarter re-embeds only new chunks; per-request traces |
| M6 | Security: indirect prompt-injection test documents and mitigations | Retrieved text is untrusted input to the LLM | Injection success rate before/after |
| M7 | Documentation: decision log, architecture, README | Every choice must be traceable | — |

## Constraints that shape decisions

- **Free Gemini tier:** ~20 requests/day for the Flash models; `gemini-3.1-flash-lite` allowed 150+.
  Every LLM call is cached, LLM-heavy steps must justify their calls, and evaluations may run over
  several days (all jobs are resumable).
- **CPU only:** models must be small (bge-small, MiniLM); no GPU-only components.
- **Explainability:** prefer a simple rule that works and can be tested over a clever component
  that can't be explained.
