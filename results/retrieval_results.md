# Retrieval evaluation

106 answerable questions (32 text, 37 table, 15 chart, 10 paraphrase, 12 multi). A chunk is relevant if it comes from a gold page and contains the answer evidence; grade 2 = the period's own report, grade 1 = a later report repeating the figure. `recall@k` is the share of a question's evidence sets found in the top k (differs from `hit@k` only for multi-period questions). `doc_filter` = restrict to the report named in the question. Latency is CPU wall-clock per query and varies between runs; the metrics do not.

## All answerable questions

| Corpus | Retriever | Hit@1 | Hit@5 | Recall@5 | Recall@10 | Precision@5 | NDCG@10 | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| text-only | bm25 | 0.123 | 0.226 | 0.226 | 0.245 | 0.051 | 0.178 | 0.164 | 2 |
| text-only | dense | 0.104 | 0.217 | 0.217 | 0.264 | 0.049 | 0.166 | 0.149 | 62 |
| text-only | hybrid | 0.151 | 0.236 | 0.236 | 0.302 | 0.055 | 0.212 | 0.192 | 60 |
| text-only | hybrid_rerank | 0.160 | 0.283 | 0.283 | 0.302 | 0.066 | 0.225 | 0.209 | 1370 |
| text+tables | bm25 | 0.311 | 0.547 | 0.528 | 0.604 | 0.172 | 0.418 | 0.416 | 3 |
| text+tables | dense | 0.330 | 0.613 | 0.594 | 0.689 | 0.172 | 0.406 | 0.448 | 51 |
| text+tables | hybrid | 0.425 | 0.642 | 0.632 | 0.741 | 0.211 | 0.501 | 0.520 | 59 |
| text+tables | hybrid_rerank | 0.264 | 0.651 | 0.627 | 0.741 | 0.194 | 0.452 | 0.416 | 2304 |
| multimodal | bm25 | 0.311 | 0.594 | 0.580 | 0.679 | 0.166 | 0.454 | 0.440 | 4 |
| multimodal | dense | 0.387 | 0.679 | 0.660 | 0.764 | 0.183 | 0.469 | 0.503 | 61 |
| multimodal | hybrid | 0.443 | 0.717 | 0.703 | 0.816 | 0.211 | 0.554 | 0.560 | 52 |
| multimodal | hybrid_rerank | 0.330 | 0.679 | 0.651 | 0.825 | 0.177 | 0.507 | 0.468 | 2100 |
| multimodal | hybrid+doc_filter | 0.557 | 0.896 | 0.887 | 0.948 | 0.230 | 0.643 | 0.692 | 72 |
| multimodal | hybrid_rerank+doc_filter | 0.481 | 0.877 | 0.863 | 0.958 | 0.221 | 0.621 | 0.636 | 2562 |

## Hit@5 by question type (Recall@5 for multi-period)

| Corpus | Retriever | text | table | chart | paraphrase | multi |
|---|---|---|---|---|---|---|
| text-only | bm25 | 0.719 | 0.027 | 0.000 | 0.000 | 0.000 |
| text-only | dense | 0.656 | 0.054 | 0.000 | 0.000 | 0.000 |
| text-only | hybrid | 0.719 | 0.054 | 0.000 | 0.000 | 0.000 |
| text-only | hybrid_rerank | 0.844 | 0.081 | 0.000 | 0.000 | 0.000 |
| text+tables | bm25 | 0.688 | 0.622 | 0.067 | 0.200 | 0.667 |
| text+tables | dense | 0.594 | 0.838 | 0.067 | 0.500 | 0.583 |
| text+tables | hybrid | 0.656 | 0.811 | 0.067 | 0.600 | 0.750 |
| text+tables | hybrid_rerank | 0.875 | 0.703 | 0.067 | 0.300 | 0.708 |
| multimodal | bm25 | 0.594 | 0.541 | 0.867 | 0.200 | 0.625 |
| multimodal | dense | 0.500 | 0.811 | 0.933 | 0.500 | 0.417 |
| multimodal | hybrid | 0.594 | 0.757 | 0.867 | 0.600 | 0.708 |
| multimodal | hybrid_rerank | 0.750 | 0.622 | 0.933 | 0.100 | 0.583 |
| multimodal | hybrid+doc_filter | 0.906 | 0.892 | 1.000 | 0.800 | 0.750 |
| multimodal | hybrid_rerank+doc_filter | 0.938 | 0.865 | 1.000 | 0.600 | 0.708 |
