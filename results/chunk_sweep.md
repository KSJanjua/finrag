# Chunk-size sweep (multimodal corpus)

| chunk size / overlap | #text chunks | mode | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|---|
| 400 / 75 | 242 | hybrid | 0.737 | 0.947 | 0.817 |
| 400 / 75 | 242 | hybrid_rerank | 0.684 | 0.921 | 0.779 |
| 800 / 150 | 127 | hybrid | 0.789 | 0.947 | 0.854 |
| 800 / 150 | 127 | hybrid_rerank | 0.684 | 0.947 | 0.779 |
| 1200 / 200 | 93 | hybrid | 0.711 | 0.895 | 0.800 |
| 1200 / 200 | 93 | hybrid_rerank | 0.632 | 0.947 | 0.738 |
| 1600 / 250 | 84 | hybrid | 0.711 | 0.895 | 0.798 |
| 1600 / 250 | 84 | hybrid_rerank | 0.553 | 0.921 | 0.687 |
