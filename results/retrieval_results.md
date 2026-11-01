# Retrieval evaluation

106 answerable questions (32 text, 37 table, 15 chart, 10 paraphrase, 12 multi). A chunk is relevant if it comes from a gold page and contains the answer evidence; grade 2 = the period's own report, grade 1 = a later report repeating the figure. `recall@k` is the share of a question's evidence sets found in the top k (differs from `hit@k` only for multi-period questions). `doc_filter` = restrict to the report named in the question. Latency is CPU wall-clock per query and varies between runs; the metrics do not.

## All answerable questions

| Corpus | Retriever | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Precision@5 | NDCG@10 | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| text-only | bm25 | 0.123 | 0.226 | 0.226 | 0.245 | 0.051 | 0.178 | 0.164 | 3 |
| text-only | dense | 0.104 | 0.217 | 0.217 | 0.264 | 0.049 | 0.166 | 0.149 | 59 |
| text-only | hybrid | 0.151 | 0.236 | 0.236 | 0.302 | 0.055 | 0.212 | 0.192 | 57 |
| text-only | hybrid_rerank | 0.160 | 0.283 | 0.283 | 0.302 | 0.066 | 0.225 | 0.209 | 1330 |
| text+tables | bm25 | 0.226 | 0.443 | 0.429 | 0.533 | 0.142 | 0.334 | 0.320 | 3 |
| text+tables | dense | 0.311 | 0.575 | 0.557 | 0.684 | 0.162 | 0.406 | 0.427 | 53 |
| text+tables | hybrid | 0.340 | 0.575 | 0.557 | 0.703 | 0.172 | 0.452 | 0.441 | 57 |
| text+tables | hybrid_rerank | 0.236 | 0.547 | 0.533 | 0.703 | 0.162 | 0.398 | 0.370 | 2295 |
| multimodal | bm25 | 0.264 | 0.462 | 0.458 | 0.623 | 0.136 | 0.390 | 0.369 | 4 |
| multimodal | dense | 0.368 | 0.651 | 0.632 | 0.741 | 0.179 | 0.467 | 0.483 | 55 |
| multimodal | hybrid | 0.396 | 0.642 | 0.637 | 0.788 | 0.185 | 0.512 | 0.510 | 59 |
| multimodal | hybrid_rerank | 0.302 | 0.575 | 0.561 | 0.769 | 0.155 | 0.450 | 0.420 | 2272 |
| multimodal | hybrid+doc_filter | 0.528 | 0.783 | 0.778 | 0.873 | 0.219 | 0.603 | 0.639 | 61 |
| multimodal | hybrid_rerank+doc_filter | 0.377 | 0.783 | 0.769 | 0.854 | 0.200 | 0.538 | 0.538 | 2144 |

## Hit@5 by question type (Recall@5 for multi-period)

| Corpus | Retriever | text | table | chart | paraphrase | multi |
|---|---|---|---|---|---|---|
| text-only | bm25 | 0.719 | 0.027 | 0.000 | 0.000 | 0.000 |
| text-only | dense | 0.656 | 0.054 | 0.000 | 0.000 | 0.000 |
| text-only | hybrid | 0.719 | 0.054 | 0.000 | 0.000 | 0.000 |
| text-only | hybrid_rerank | 0.844 | 0.081 | 0.000 | 0.000 | 0.000 |
| text+tables | bm25 | 0.688 | 0.405 | 0.067 | 0.200 | 0.458 |
| text+tables | dense | 0.656 | 0.757 | 0.067 | 0.300 | 0.500 |
| text+tables | hybrid | 0.688 | 0.676 | 0.067 | 0.400 | 0.583 |
| text+tables | hybrid_rerank | 0.906 | 0.459 | 0.067 | 0.100 | 0.708 |
| multimodal | bm25 | 0.594 | 0.324 | 0.867 | 0.200 | 0.208 |
| multimodal | dense | 0.531 | 0.757 | 0.933 | 0.300 | 0.417 |
| multimodal | hybrid | 0.625 | 0.649 | 0.867 | 0.400 | 0.542 |
| multimodal | hybrid_rerank | 0.750 | 0.405 | 0.933 | 0.100 | 0.458 |
| multimodal | hybrid+doc_filter | 0.781 | 0.865 | 1.000 | 0.400 | 0.542 |
| multimodal | hybrid_rerank+doc_filter | 0.844 | 0.865 | 1.000 | 0.200 | 0.458 |
