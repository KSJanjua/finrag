"""Evaluation: retrieval metrics (Recall@k, MRR) and answer correctness."""
import json
import re

from . import config
from .llm import chat, parse_json, settings

def doc_for_code(code):
    """'2023Q3' -> the registered PDF for that period."""
    m = re.fullmatch(r"(\d{4})Q([1-4])", code)
    if not m or (int(m[1]), int(m[2])) not in config.DOC_BY_PERIOD:
        raise ValueError(f"unknown document code {code!r}; expected e.g. '2023Q3'")
    return config.DOC_BY_PERIOD[(int(m[1]), int(m[2]))]


def load_questions(path=config.QUESTIONS_FILE):
    data = json.loads(path.read_text(encoding="utf-8"))
    for q in data["questions"]:
        q["gold_pages"] = {(doc_for_code(d), p) for d, p in q["gold"]}
    return data["questions"]


def normalize(text):
    """Lower-case, drop thousands separators and '$', collapse whitespace."""
    text = text.lower().replace("$", "")
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return re.sub(r"\s+", " ", text)


def contains(text, phrase):
    """Does normalised ``phrase`` occur in normalised ``text``?

    Numbers must match whole: "5%" is not found inside "35%", and "37,500"
    is not found inside "375,000". Plain substring matching would make both
    of those checks pass by accident.
    """
    phrase = normalize(phrase).strip()
    left = r"(?<![\d.])" if phrase[:1].isdigit() else ""
    right = r"(?![\d])" if phrase[-1:].isdigit() else ""
    return re.search(left + re.escape(phrase) + right, normalize(text)) is not None


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def is_relevant(chunk, question):
    """A chunk is relevant if it comes from a gold page AND contains the evidence.

    Page-level matching alone would be too generous: the Model Y price page
    has a text chunk that is just a footnote, which should not count as
    retrieving the answer.
    """
    if (chunk.doc, chunk.page) not in question["gold_pages"]:
        return False
    return any(contains(chunk.text, e) for e in question["evidence"])


def first_relevant_rank(retrieved, question):
    for rank, r in enumerate(retrieved, start=1):
        if is_relevant(r.chunk, question):
            return rank
    return None


def retrieval_metrics(ranks, ks=(1, 3, 5, 10)):
    """ranks: first-relevant rank per question (None = not found in top-K).

    Recall@k here is the fraction of questions with at least one relevant
    chunk in the top k (every question has one answer to find, so this equals
    Hit Rate@k). MRR = mean of 1/rank of the first relevant chunk (0 if missing).
    """
    n = len(ranks)
    if n == 0:
        return {**{f"recall@{k}": None for k in ks}, "mrr": None}
    out = {f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


# --------------------------------------------------------------------------
# Answer correctness
# --------------------------------------------------------------------------
def facts_correct(answer, fact_groups):
    """Deterministic check: every fact group must have one alternative in the answer."""
    return all(any(contains(answer, alt) for alt in group) for group in fact_groups)


JUDGE_PROMPT = """You are grading a question-answering system on financial documents.
Question: {question}
Facts a correct answer must contain: {facts}
System answer: {answer}

Is the system answer correct and consistent with the required facts (approximate values for charts are fine)?
Return JSON only, no markdown: {{"correct": true/false, "reason": "<one sentence>"}}"""


def judge_correct(client, question, answer, fact_groups, model=None):
    """LLM-as-judge: more tolerant of paraphrase than string matching."""
    model = model or settings()["judge_model"]
    facts = "; ".join(" OR ".join(g) for g in fact_groups)
    raw = chat(client, model, [{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, facts=facts, answer=answer)}], max_tokens=150)
    try:
        verdict = parse_json(raw)
    except ValueError:  # json.JSONDecodeError is a ValueError
        return False, f"unparseable judge output: {raw[:100]!r}"
    return verdict.get("correct") is True, verdict.get("reason", "")
