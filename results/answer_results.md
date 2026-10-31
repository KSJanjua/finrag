# Answer correctness (hybrid, doc filter=auto, top-5 context)

- Generator: `gemini-3.1-flash-lite`, one question per request (the production path)
- Fact match: every required fact appears in the answer (numbers matched whole)
- LLM judge: not run
- Abstained: answer contains "I could not find this in the provided documents."

| Corpus | n | Fact match (all) | text | table | chart | Abstained |
|---|---|---|---|---|---|---|
| text-only | 38 | 0.395 | 0.929 | 0.154 | 0.000 | 0.526 |
| multimodal | 38 | 0.974 | 0.929 | 1.000 | 1.000 | 0.000 |
