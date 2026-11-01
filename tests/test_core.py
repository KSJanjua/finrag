"""Unit tests for the pieces implemented from scratch (no models needed)."""
import pytest

from finrag.chunking import recursive_split, split_table
from finrag.evaluate import contains, facts_correct, normalize, retrieval_metrics
from finrag.generate import parse_citations
from finrag.hybrid import reciprocal_rank_fusion
from finrag.ingest import clean_text, rows_to_markdown, table_is_usable
from finrag.retriever import infer_doc
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
    parts = split_table(md, max_chars=200)
    assert len(parts) > 1
    assert all(p.startswith("|Metric|Q3|\n|---|---|") for p in parts)


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


def test_metrics():
    m = retrieval_metrics([1, 3, None, 6], ks=(1, 5))
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5
    assert abs(m["mrr"] - (1 + 1 / 3 + 1 / 6) / 4) < 1e-12


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
