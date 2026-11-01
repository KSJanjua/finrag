"""Evaluation: retrieval metrics, answer correctness and number grounding."""
import json
import math
import re

from . import config
from .llm import chat, parse_json, settings

KS = (1, 3, 5, 10)


def doc_for_code(code):
    """'2023Q3' -> the registered PDF for that period."""
    m = re.fullmatch(r"(\d{4})Q([1-4])", code)
    if not m or (int(m[1]), int(m[2])) not in config.DOC_BY_PERIOD:
        raise ValueError(f"unknown document code {code!r}; expected e.g. '2023Q3'")
    return config.DOC_BY_PERIOD[(int(m[1]), int(m[2]))]


def load_questions(path=config.QUESTIONS_FILE):
    """Questions with their evidence sets resolved to {(pdf, page): grade} maps.

    Accepts the current format (``evidence_sets``) and the original one
    (``gold`` + ``evidence`` at the top level, every page grade 2).
    Unanswerable questions have no evidence sets.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    for q in data["questions"]:
        sets = q.get("evidence_sets")
        if sets is None:
            sets = [{"gold": q["gold"], "evidence": q["evidence"]}]
        q["sets"] = [{"pages": {(doc_for_code(g[0]), g[1]): (g[2] if len(g) > 2 else 2) for g in s["gold"]},
                      "evidence": s["evidence"]} for s in sets]
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
def satisfied_sets(chunk, question):
    """[(set index, grade)] for every evidence set this chunk satisfies.

    A chunk satisfies a set if it comes from one of the set's gold pages and
    contains one of its evidence strings. Page-level matching alone would be too
    generous: a page can hold a footnote-only text chunk next to the chart that
    actually answers the question.
    """
    out = []
    for i, s in enumerate(question["sets"]):
        grade = s["pages"].get((chunk.doc, chunk.page), 0)
        if grade and any(contains(chunk.text, e) for e in s["evidence"]):
            out.append((i, grade))
    return out


def relevance(chunk, question):
    """Graded relevance: 2 = the period's own report, 1 = a later report repeating the figure."""
    return max((g for _, g in satisfied_sets(chunk, question)), default=0)


def is_relevant(chunk, question):
    return relevance(chunk, question) > 0


def first_relevant_rank(retrieved, question):
    for rank, r in enumerate(retrieved, start=1):
        if is_relevant(r.chunk, question):
            return rank
    return None


def ideal_grades(chunks, question):
    """Grades of all relevant chunks in the searched corpus, best first (NDCG's ideal ranking)."""
    return sorted((g for g in (relevance(c, question) for c in chunks) if g), reverse=True)


def score_ranking(retrieved, question, ideal, ks=KS):
    """Retrieval scores of one ranked list for one answerable question.

    hit@k        any relevant chunk in the top k
    recall@k     share of the question's evidence sets covered in the top k (context recall;
                 differs from hit@k only for multi-period questions)
    precision@k  share of the top k that is relevant
    ndcg@k       graded ranking quality: own-report chunks above later copies
    rr           reciprocal rank of the first relevant chunk
    """
    satisfied = [satisfied_sets(r.chunk, question) for r in retrieved]
    grades = [max((g for _, g in s), default=0) for s in satisfied]
    first = next((i + 1 for i, g in enumerate(grades) if g), None)
    out = {"rank": first, "rr": 1 / first if first else 0.0}
    for k in ks:
        covered = {i for s in satisfied[:k] for i, _ in s}
        dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))
        idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        out[f"hit@{k}"] = float(any(grades[:k]))
        out[f"recall@{k}"] = len(covered) / len(question["sets"])
        out[f"precision@{k}"] = sum(1 for g in grades[:k] if g) / k
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out


def aggregate(scores):
    """Mean of each per-question score; the mean reciprocal rank is reported as MRR."""
    if not scores:
        return {}
    out = {k: sum(s[k] for s in scores) / len(scores) for k in scores[0] if k != "rank"}
    out["mrr"] = out.pop("rr")
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


# --------------------------------------------------------------------------
# Number grounding (a deterministic faithfulness check)
# --------------------------------------------------------------------------
_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_NUMBER = re.compile(r"(?<![\w.])\$?\(?-?\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    """(raw, value) for every number in ``text``; citation markers like [2] are ignored."""
    out = []
    for raw in _NUMBER.findall(_CITATION.sub(" ", text)):
        digits = raw.strip("$(").replace(",", "")
        try:
            out.append((raw, abs(float(digits))))
        except ValueError:
            continue
    return out


def unsupported_numbers(answer, sources):
    """Numbers in an answer that none of its sources support.

    A number is supported if some source value, rounded to the precision the
    answer uses, equals it: "$2.5B" is supported by "(2,531)" in a $M table
    (2.531 -> 2.5), "$26.1B" is not supported by "25,500". Values are compared
    as-is and scaled by 1,000 either way (millions vs billions). Small integers
    and bare years are skipped: they are counts in the wording ("three
    reports") or period names ("2024") rather than figures.
    """
    source_values = [v for text in sources for _, v in _numbers(text)]
    missing = []
    for raw, v in _numbers(answer):
        if (v < 10 and v == int(v)) or re.fullmatch(r"20\d\d", raw):
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not any(abs(round(s * scale, decimals) - v) < 1e-9
                   for s in source_values for scale in (1, 1000, 0.001)):
            missing.append(raw)
    return missing
