# Reranker sweep (multimodal corpus, 38 questions)

| Reranker | Report filter | Recall@1 | Recall@5 | MRR | ms/query |
|---|---|---|---|---|---|
| none (hybrid RRF only) | none | 0.789 | 0.947 | 0.854 | 74 |
| none (hybrid RRF only) | auto | 0.868 | 0.974 | 0.912 | 75 |
| MiniLM-L6 max_len=512 batch=32 | none | 0.684 | 0.947 | 0.779 | 2090 |
| MiniLM-L6 max_len=512 batch=32 | auto | 0.763 | 0.947 | 0.842 | 2091 |
| MiniLM-L6 max_len=512 batch=4 | none | 0.684 | 0.947 | 0.779 | 1562 |
| MiniLM-L6 max_len=512 batch=4 | auto | 0.763 | 0.947 | 0.842 | 1446 |
| MiniLM-L6 max_len=256 batch=4 | none | 0.632 | 0.868 | 0.730 | 974 |
| MiniLM-L6 max_len=256 batch=4 | auto | 0.737 | 0.868 | 0.794 | 1016 |

Not included: `mixedbread-ai/mxbai-rerank-xsmall-v1` did not finish 38 queries in 45 minutes on this CPU (> 70 s/query), and `BAAI/bge-reranker-base` (278M parameters) is larger still - both are impractical without a GPU.
