# Answer evaluation (hybrid, doc filter=auto, top-8 context)

- Generator: `gemini-3.1-flash-lite`, one question per request (the production path)
- Fact match: every required fact appears in the answer (numbers matched whole)
- Refusal: the answer contains "I could not find this in the provided documents."
- Unsupported numbers: share of non-refusing answers with a number that none of its cited sources contains (rounding and $M/$B scale allowed)
- LLM judge: not run

| Corpus | Fact match | False refusals | Correct refusals (unanswerable) | Unsupported numbers |
|---|---|---|---|---|
| multimodal | 0.943 | 0.038 | 1.000 | 0.010 |

## Fact match by question type

| Corpus | text | table | chart | paraphrase | multi |
|---|---|---|---|---|---|
| multimodal | 0.906 | 0.973 | 1.000 | 0.900 | 0.917 |
