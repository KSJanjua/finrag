"""PDF ingestion: page text, tables and chart descriptions -> typed chunks.

Every chunk keeps (doc, page, type) metadata so answers can cite pages and
retrieval can be filtered by document or modality.
"""
import json
import re

import fitz  # PyMuPDF

from . import config
from .chunking import recursive_split, split_table
from .dense import token_counter
from .schema import Chunk

fitz.TOOLS.mupdf_display_errors(False)  # these PDFs trigger harmless structure-tree warnings

NUMERIC_LINE = re.compile(r"^[\s\d.,%$()\-–—+/xX]*$")
# One letter-spaced word: single characters separated by single spaces ("S U M M A R Y"),
# possibly a single character ("Y" in "M O D E L   Y", "&"), optionally ending in a
# footnote marker ("S U B S I D I E S*").
LETTER_SPACED = re.compile(r"^(?:\S )*\S[^\w\s]*$")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _unspace_heading(line):
    """'C O R E   T E C H N O L O G Y' -> 'CORE TECHNOLOGY'."""
    words = re.split(r"\s{2,}", line.strip())
    if all(LETTER_SPACED.match(w) for w in words):
        return " ".join(w.replace(" ", "") for w in words)
    return line


def clean_text(text, min_numeric_run=4):
    """Normalise headings and drop runs of numbers-only lines.

    Runs like '0 / 100 / 200 / 300' are chart axis ticks (or table cells that
    lost their row/column labels). They carry no meaning on their own and only
    add noise to embeddings and BM25.
    """
    lines = [_unspace_heading(l.strip()) for l in text.splitlines()]
    out, run = [], []
    for line in lines:
        if not line and run:  # blank lines between text blocks don't end a run
            continue
        if line and NUMERIC_LINE.match(line):
            run.append(line)
            continue
        if 0 < len(run) < min_numeric_run:
            out.extend(run)
        run = []
        out.append(line)
    if 0 < len(run) < min_numeric_run:
        out.extend(run)
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def page_section(page):
    """Return the page's letter-spaced headings, e.g. 'OPERATIONAL SUMMARY'."""
    heads = []
    for line in page.get_text().splitlines():
        line = line.strip()
        if len(line) > 3 and line != _unspace_heading(line):
            heads.append(_unspace_heading(line))
    return " ".join(dict.fromkeys(heads))[:80]


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def table_is_usable(rows):
    """Reject tables the detector got wrong (mostly empty / repeated cells).

    Merged cells in the financial statements come out as one label repeated
    on every row with the numbers scattered; such tables are rebuilt with
    rows_from_words() instead.
    """
    if len(rows) < 2:
        return False
    cells = [c for r in rows for c in r]
    empty = sum(1 for c in cells if c is None or not str(c).strip())
    if empty / max(len(cells), 1) > 0.4:
        return False
    first_col = [str(r[0]).strip() for r in rows if r and r[0]]
    if first_col and len(set(first_col)) / len(first_col) < 0.5:
        return False
    return True


def rows_from_words(page, rect, y_tol=3.0):
    """Fallback table reconstruction from word coordinates.

    Words whose vertical centres are within ``y_tol`` points belong to the same
    visual row; inside a row, words are ordered left to right and split into
    cells wherever the horizontal gap is large. This recovers rows such as
    'Research and development | 733 | 810 | 771 | 943 | 1,161'.
    """
    words = [w for w in page.get_text("words") if fitz.Rect(w[:4]).intersects(rect)]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, current, current_y = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        if current and abs(y - current_y) > y_tol:
            lines.append(current)
            current = []
        if not current:
            current_y = y
        current.append(w)
    if current:
        lines.append(current)

    rows = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        cells, cell, last_x1 = [], [], None
        for x0, _, x1, _, word, *_ in line:
            if cell and x0 - last_x1 > 8:  # big gap -> new cell
                cells.append(" ".join(cell))
                cell = []
            cell.append(word)
            last_x1 = x1
        cells.append(" ".join(cell))
        rows.append([c for c in cells if c != "$"])  # currency signs sit in their own column
    return rows


def rows_to_markdown(rows):
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    out = ["|" + "|".join(rows[0]) + "|", "|" + "---|" * width]
    out += ["|" + "|".join(r) + "|" for r in rows[1:]]
    return "\n".join(out)


def extract_tables(page):
    """Return [(bbox, markdown)] for the tables on the page.

    PyMuPDF's detector is used first; if its output fails the quality check
    (typical for statements with merged cells) the rows are rebuilt from
    word positions instead.
    """
    tables = []
    for t in page.find_tables().tables:
        rect = fitz.Rect(t.bbox)
        if table_is_usable(t.extract()):
            md = t.to_markdown().strip()
            # drop empty separator rows like '||||||'
            md = "\n".join(l for l in md.splitlines() if l.replace("|", "").strip() or "---" in l)
            md = md.replace("<br>", " ")  # multi-line cells
        else:
            md = rows_to_markdown(rows_from_words(page, rect))
        if md:
            tables.append((rect, md))
    return tables


def page_text_outside(page, rects):
    """Page text with blocks that sit inside a table removed (avoids duplicates)."""
    kept = []
    for x0, y0, x1, y1, text, _no, kind in page.get_text("blocks"):
        if kind != 0:  # 0 = text block, 1 = image block
            continue
        block = fitz.Rect(x0, y0, x1, y1)
        inside = any(block.intersect(r).get_area() > 0.5 * block.get_area() for r in rects if block.get_area())
        if not inside:
            kept.append(text)
    return "\n\n".join(kept)


# --------------------------------------------------------------------------
# Chunk builders
# --------------------------------------------------------------------------
def _tag_period(chunk):
    meta = config.DOCUMENTS.get(chunk.doc, {})
    chunk.fiscal_year = meta.get("fiscal_year", 0)
    chunk.quarter = meta.get("quarter", 0)
    return chunk


def parse_pdf(path, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Yield text and table chunks for one PDF."""
    doc_name = path.name
    label = config.DOC_LABELS.get(doc_name, doc_name)
    with fitz.open(path) as pdf:
        for chunk in _parse_pages(pdf, doc_name, label, chunk_size, overlap):
            yield _tag_period(chunk)


def _parse_pages(pdf, doc_name, label, chunk_size, overlap):
    for page_no, page in enumerate(pdf, start=1):
        section = page_section(page)
        tables = extract_tables(page)

        for t_idx, (_, md) in enumerate(tables):
            for part_idx, part in enumerate(split_table(md, config.TABLE_MAX_TOKENS, token_counter())):
                yield Chunk(
                    chunk_id=f"{doc_name}:p{page_no}:table{t_idx}.{part_idx}",
                    doc=doc_name, doc_label=label, page=page_no, type="table",
                    text=part, section=section,
                )

        text = clean_text(page_text_outside(page, [r for r, _ in tables]))
        for c_idx, piece in enumerate(recursive_split(text, chunk_size, overlap)):
            if len(piece) < 40:  # page numbers, lone headings
                continue
            yield Chunk(
                chunk_id=f"{doc_name}:p{page_no}:text{c_idx}",
                doc=doc_name, doc_label=label, page=page_no, type="text",
                text=piece, section=section,
            )


def chart_chunks(cache_path=config.CHART_CACHE, sections=None):
    """One chunk per chart description produced by the vision model.

    Cache entries with description = null record "page checked, no charts".
    """
    if not cache_path.exists():
        return []
    sections = sections or {}
    entries = json.loads(cache_path.read_text(encoding="utf-8"))
    chunks, counter = [], {}
    for e in entries:
        if not e.get("description"):
            continue
        key = (e["doc"], e["page"])
        counter[key] = counter.get(key, -1) + 1
        chunks.append(_tag_period(Chunk(
            chunk_id=f"{e['doc']}:p{e['page']}:chart{counter[key]}",
            doc=e["doc"], doc_label=config.DOC_LABELS.get(e["doc"], e["doc"]),
            page=e["page"], type="chart",
            text=f"Chart description: {e['description']}",
            section=sections.get(key, ""),
        )))
    return chunks


def build_chunks(pdf_dir=config.PDF_DIR, chart_cache=config.CHART_CACHE,
                 chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP, docs=None):
    """Chunks for every registered document (or the given subset of file names).

    Only documents listed in the registry are ingested, in chronological order,
    so an unrelated PDF in the folder can't end up in the index.
    """
    names = list(docs) if docs is not None else list(config.DOCUMENTS)
    missing = [n for n in names if not (pdf_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing PDFs {missing} - run scripts/build_index.py to download them")
    chunks = []
    for name in names:
        chunks.extend(parse_pdf(pdf_dir / name, chunk_size, overlap))
    sections = {(c.doc, c.page): c.section for c in chunks}
    chunks.extend(c for c in chart_chunks(chart_cache, sections) if c.doc in names)
    return chunks
