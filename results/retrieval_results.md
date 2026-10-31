# Retrieval evaluation

38 questions (14 text, 13 table, 11 chart). A hit = a chunk from a gold page that contains the answer evidence. `doc_filter` = restrict to the report named in the question (e.g. "Q3 2023"). Latency is CPU wall-clock per query and varies between runs; the metrics do not.

## All questions

| Modality | Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|---|---|
| text-only | bm25 | 0.289 | 0.368 | 0.421 | 0.421 | 0.339 | 0 |
| text-only | dense | 0.316 | 0.342 | 0.421 | 0.474 | 0.353 | 62 |
| text-only | hybrid | 0.342 | 0.421 | 0.421 | 0.421 | 0.377 | 40 |
| text-only | hybrid_rerank | 0.368 | 0.421 | 0.421 | 0.447 | 0.398 | 1012 |
| text+tables | bm25 | 0.474 | 0.605 | 0.632 | 0.711 | 0.549 | 1 |
| text+tables | dense | 0.579 | 0.632 | 0.684 | 0.711 | 0.622 | 68 |
| text+tables | hybrid | 0.579 | 0.684 | 0.711 | 0.711 | 0.634 | 48 |
| text+tables | hybrid_rerank | 0.500 | 0.632 | 0.711 | 0.737 | 0.580 | 2790 |
| multimodal | bm25 | 0.658 | 0.789 | 0.868 | 0.974 | 0.751 | 2 |
| multimodal | dense | 0.737 | 0.816 | 0.895 | 0.921 | 0.793 | 76 |
| multimodal | hybrid | 0.789 | 0.895 | 0.947 | 0.974 | 0.854 | 81 |
| multimodal | hybrid_rerank | 0.684 | 0.816 | 0.947 | 0.974 | 0.779 | 2788 |
| multimodal | hybrid+doc_filter | 0.868 | 0.974 | 0.974 | 0.974 | 0.912 | 77 |
| multimodal | hybrid_rerank+doc_filter | 0.763 | 0.921 | 0.947 | 0.974 | 0.842 | 2826 |

## Recall@5 by question type

| Modality | Retriever | text | table | chart |
|---|---|---|---|---|
| text-only | bm25 | 0.929 | 0.231 | 0.000 |
| text-only | dense | 0.857 | 0.231 | 0.091 |
| text-only | hybrid | 0.929 | 0.231 | 0.000 |
| text-only | hybrid_rerank | 0.929 | 0.231 | 0.000 |
| text+tables | bm25 | 0.857 | 0.846 | 0.091 |
| text+tables | dense | 0.857 | 0.923 | 0.182 |
| text+tables | hybrid | 0.929 | 1.000 | 0.091 |
| text+tables | hybrid_rerank | 0.929 | 1.000 | 0.091 |
| multimodal | bm25 | 0.786 | 0.846 | 1.000 |
| multimodal | dense | 0.857 | 0.846 | 1.000 |
| multimodal | hybrid | 0.929 | 0.923 | 1.000 |
| multimodal | hybrid_rerank | 0.929 | 0.923 | 1.000 |
| multimodal | hybrid+doc_filter | 0.929 | 1.000 | 1.000 |
| multimodal | hybrid_rerank+doc_filter | 0.929 | 0.923 | 1.000 |
