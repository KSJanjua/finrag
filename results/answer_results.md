# Answer evaluation (hybrid, doc filter=auto, top-5 context)

- Generator: `gemini-3.1-flash-lite`, one question per request (the production path)
- Fact match: every required fact appears in the answer (numbers matched whole)
- Refusal: the answer contains "I could not find this in the provided documents."
- Unsupported numbers: share of non-refusing answers with a number that none of its cited sources contains (rounding and $M/$B scale allowed)
- LLM judge: not run

| Corpus | Fact match | False refusals | Correct refusals (unanswerable) | Unsupported numbers |
|---|---|---|---|---|
| multimodal | 0.840 | 0.132 | 1.000 | 0.011 |

## Fact match by question type

| Corpus | text | table | chart | paraphrase | multi |
|---|---|---|---|---|---|
| multimodal | 0.812 | 0.946 | 1.000 | 0.500 | 0.667 |
