"""Unit tests for the pieces implemented from scratch (no models needed)."""
import math

import pytest

from finrag.chunking import recursive_split, split_table
from finrag.evaluate import aggregate, contains, facts_correct, normalize, score_ranking, unsupported_numbers
from finrag.schema import Chunk, RetrievedChunk
from finrag.generate import parse_citations
from finrag.hybrid import reciprocal_rank_fusion
from finrag.ingest import clean_text, rows_to_markdown, table_is_usable
from finrag.query import parse_periods
from finrag.retriever import infer_doc, report_filters
from finrag.sparse import tokenize

TEXT = ("Revenue grew 9% YoY to $23.4B.\n\n" + "Operating margin was 7.6%. " * 30 +
        "\n\nFree cash flow was $0.8B in Q3.")


def test_chunks_respect_size():
    chunks = recursive_split(TEXT, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_cover_all_text():
    chunks = recursive_split(TEXT, chunk_size=200, overlap=40)
    joined = " ".join(chunks)
    for fact in ["$23.4B", "7.6%", "$0.8B"]:
        assert fact in joined


def test_neighbouring_chunks_overlap():
    chunks = recursive_split(TEXT, chunk_size=200, overlap=40)
    shared = [a[-20:].strip() in b for a, b in zip(chunks, chunks[1:])]
    assert any(shared)


def test_short_text_is_one_chunk():
    assert recursive_split("short text", 100, 10) == ["short text"]


def test_split_table_repeats_header():
    md = "|Metric|Q3|\n|---|---|\n" + "\n".join(f"|row{i}|{i}|" for i in range(100))
    parts = split_table(md, max_size=200)
    assert len(parts) > 1
    assert all(p.startswith("|Metric|Q3|\n|---|---|") for p in parts)
    assert all(len(p) <= 200 for p in parts)


def test_split_table_uses_the_given_size_measure():
    md = "|Metric|Q3|\n|---|---|\n" + "\n".join(f"|row {i}|{i}|" for i in range(40))
    words = lambda text: len(text.split())          # stand-in for a token counter
    parts = split_table(md, max_size=20, size=words)
    assert len(parts) > 1 and all(words(p) <= 20 for p in parts)
    assert sum(p.count("|row ") for p in parts) == 40   # no row lost or duplicated


def test_rrf_rewards_agreement():
    # "b" is 2nd in both lists, "a" is 1st in one list only -> b wins
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]], k=60)
    assert fused[0][0] == "b"


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["x"], ["x"]], k=60))
    assert abs(fused["x"] - 2 / 61) < 1e-12


def test_tokenizer_keeps_numbers():
    toks = tokenize("Operating margin of 7.6% and 51,105 connectors")
    assert "7.6%" in toks and "51105" in toks and "of" not in toks


def test_clean_text_drops_axis_ticks():
    text = "Energy Storage deployments (GWh)\n0\n1\n2\n\n3\n4\nSolar fell to 49 MW"
    cleaned = clean_text(text)
    assert "Solar fell" in cleaned and "\n2\n" not in cleaned


def test_clean_text_unspaces_headings():
    assert clean_text("C O R E   T E C H N O L O G Y") == "CORE TECHNOLOGY"


def _hit(doc, text):
    return RetrievedChunk(Chunk("id", doc, doc, 4, "table", text), 1.0)


def test_ranking_scores_for_a_multi_period_question():
    # Two evidence sets; A.pdf is the own report (grade 2), B.pdf a later copy (grade 1).
    q = {"sets": [{"pages": {("A.pdf", 4): 2, ("B.pdf", 4): 1}, "evidence": ["443,956"]},
                  {"pages": {("A.pdf", 4): 2}, "evidence": ["386,810"]}]}
    ranked = [_hit("C.pdf", "noise"), _hit("B.pdf", "443,956"), _hit("A.pdf", "386,810 and 443,956")]
    s = score_ranking(ranked, q, ideal=[2, 1], ks=(1, 2, 3))
    assert s["rank"] == 2 and s["rr"] == 0.5
    assert s["hit@1"] == 0 and s["hit@2"] == 1
    assert s["recall@2"] == 0.5 and s["recall@3"] == 1.0      # second set only found at rank 3
    assert abs(s["precision@3"] - 2 / 3) < 1e-12
    dcg = 1 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 1 / math.log2(3)
    assert abs(s["ndcg@3"] - dcg / idcg) < 1e-12
    assert aggregate([s, {**s, "rr": 1.0}])["mrr"] == 0.75


@pytest.mark.parametrize("answer, unsupported", [
    ("Revenue was $25.5 billion in Q2 2024 [1].", []),          # $B vs a table in $M, rounding allowed
    ("Revenue was $26.1 billion [1].", ["$26.1"]),               # not in any source
    ("Revenue was about $21.3 billion [1].", []),                # 21,301 rounds to 21.3
    ("It covers three reports and 2024 [1].", []),               # small counts and years are not claims
])
def test_unsupported_numbers(answer, unsupported):
    sources = ["Tesla Q2 2024 Quarterly Update | page 4\n|Total revenues|21,301|25,500|"]
    assert unsupported_numbers(answer, sources) == unsupported


def test_fact_matching_normalises_numbers():
    assert normalize("$51,105") == "51105"
    assert facts_correct("Tesla had 51105 connectors [1].", [["51,105"]])
    assert not facts_correct("Revenue was 23.4B", [["23.4"], ["9%"]])


def test_number_matching_is_whole_number():
    assert not contains("approx. 35% growth", "5%")
    assert not contains("capacity of 375,000", "37,500")
    assert contains("cost was ~$37,500 per vehicle", "37,500")
    assert contains("margin of 7.6%.", "7.6%")


def test_fact_groups_need_every_group():
    # c07-style check: "5%" must not be satisfied by "35%"
    assert not facts_correct("Tesla grew ~35%.", [["35%"], ["5%"]])
    assert facts_correct("Tesla ~35% vs industry ~5%.", [["35%"], ["5%"]])


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        recursive_split("x" * 100, chunk_size=10, overlap=10)


def test_citation_parsing():
    assert parse_citations("Revenue was $23.4B [1]. Margin 7.6% [2, 3][3].", 5) == [1, 2, 3]
    assert parse_citations("See [9] and [0].", 5) == []


@pytest.mark.parametrize("query, expected", [
    ("Supercharger connectors at the end of Q3 2023?", "TSLA-Q3-2023-Update-3.pdf"),
    ("Total production in Q2-2023", "TSLA-Q2-2023-Update.pdf"),
    ("Energy storage deployed in Q4 2024", "TSLA-Q4-2024-Update.pdf"),
    ("Operating margin in Q1'25", "TSLA-Q1-2025-Update.pdf"),
    ("Compare Q1 2023 and Q2 2023", None),      # two quarters -> no filter
    ("Revenue in Q1 2021", None),               # not in the registry
    ("Model Y price in Colorado", None),
])
def test_infer_doc(query, expected):
    assert infer_doc(query) == expected


def test_registry_is_chronological_and_complete():
    from finrag import config
    periods = [(d["fiscal_year"], d["quarter"]) for d in config.DOCUMENTS.values()]
    assert periods == sorted(periods) and len(periods) == len(set(periods))
    assert all(name in config.PDF_URLS for name in config.DOCUMENTS)


def test_doc_codes_resolve_through_registry():
    from finrag.evaluate import doc_for_code
    assert doc_for_code("2023Q3") == "TSLA-Q3-2023-Update-3.pdf"
    with pytest.raises(ValueError):
        doc_for_code("Q3")


@pytest.mark.parametrize("line, expected", [
    ("C O R E   T E C H N O L O G Y", "CORE TECHNOLOGY"),
    ("M O D E L   Y   S T A R T I N G   P R I C E   S U B S I D I E S*", "MODEL Y STARTING PRICE SUBSIDIES*"),
    ("P H O T O S   &   C H A R T S", "PHOTOS & CHARTS"),
    ("Revenue grew 9%", "Revenue grew 9%"),
    ("Q3-2023", "Q3-2023"),
])
def test_letter_spaced_headings(line, expected):
    assert clean_text(line) == expected


def test_table_quality_check():
    good = [["Metric", "Q1", "Q2"], ["Revenue", "10", "12"], ["Margin", "5%", "6%"]]
    garbled = [["REVENUES", None, None], ["REVENUES", None, "18,582"], ["REVENUES", None, "554"]]
    assert table_is_usable(good)
    assert not table_is_usable(garbled)


def test_rows_to_markdown():
    md = rows_to_markdown([["Metric", "Q3"], ["R&D", "1,161"]])
    assert md.splitlines()[0] == "|Metric|Q3|" and "|R&D|1,161|" in md


@pytest.mark.parametrize("query, quarters, full_years", [
    ("Operating margin in Q3 2024", [(2024, 3)], []),
    ("Revenue in Q4'24 and Q1-2025", [(2024, 4), (2025, 1)], []),
    ("Total production in 2Q-2024", [(2024, 2)], []),
    ("Revenue in Q1 FY2025", [(2025, 1)], []),
    ("Margin for the third quarter of 2024", [(2024, 3)], []),
    ("Cash at the end of quarter three, 2024", [(2024, 3)], []),
    ("Connectors at the end of Quarter 1 2025", [(2025, 1)], []),
    ("Deliveries in Q1, Q2 and Q3 2024", [(2024, 1), (2024, 2), (2024, 3)], []),
    ("Operating margin for full-year 2022", [], [2022]),
    ("AI training compute growth in 2024", [], [2024]),
    ("Model Y price in Colorado", [], []),
    ("Over 400 vehicles in Q3 results", [], []),
])
def test_parse_periods(query, quarters, full_years):
    p = parse_periods(query)
    assert p.quarters == quarters and p.full_years == full_years


def test_report_filters():
    assert report_filters("Deliveries in Q1, Q2 and Q3 2024") == [
        "TSLA-Q1-2024-Update.pdf", "TSLA-Q2-2024-Update.pdf", "TSLA-Q3-2024-Update.pdf"]
    assert report_filters("Free cash flow for full-year 2024") == ["TSLA-Q4-2024-Update.pdf"]
    assert report_filters("Deliveries in Q3 2025") == [None]      # named, but no report of its own: search all
    assert report_filters("Model Y price in Colorado") == []
    assert infer_doc("Deliveries in Q1, Q2 and Q3 2024") is None  # several periods: no single filter
