"""Recursive character chunking (implemented from scratch).

Idea: try to split on the "largest" natural boundary first (paragraphs), and
only fall back to smaller boundaries (lines, sentences, words, characters) for
pieces that are still longer than ``chunk_size``. Neighbouring chunks share up
to ``overlap`` characters so a fact cut at a boundary still appears whole in
one of the two chunks.
"""

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_split(text, chunk_size, overlap, separators=DEFAULT_SEPARATORS):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # Pick the first separator that actually occurs in the text.
    sep, rest = separators[-1], []
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            sep, rest = candidate, separators[i + 1:]
            break

    pieces = list(text) if sep == "" else _split_keep(text, sep)

    # Pieces that are still too long are split with the next separator.
    small = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            small.append(piece)
        else:
            small.extend(recursive_split(piece, chunk_size, overlap, rest))
    return _merge(small, chunk_size, overlap)


def _split_keep(text, sep):
    """Split on ``sep`` but keep the separator attached to the left piece."""
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _merge(pieces, chunk_size, overlap):
    """Greedily pack pieces into chunks of <= chunk_size with overlap."""
    chunks, window, length = [], [], 0
    for piece in pieces:
        if length + len(piece) > chunk_size and window:
            chunks.append("".join(window).strip())
            # Drop pieces from the front until only ~overlap chars remain;
            # what remains becomes the start of the next chunk.
            while window and (length > overlap or length + len(piece) > chunk_size):
                length -= len(window[0])
                window.pop(0)
        window.append(piece)
        length += len(piece)
    if window:
        chunks.append("".join(window).strip())
    return [c for c in chunks if c]


def split_table(markdown, max_size, size=len):
    """Split a markdown table by rows, repeating the header in every part.

    A table row without its header ("|1,129|") is meaningless to both the
    retriever and the LLM, so the header always travels with the rows. ``size``
    measures a piece of text (characters by default; ingestion passes the
    embedding model's token count so every part fits the model's input).
    """
    lines = [l for l in markdown.strip().splitlines() if l.strip()]
    if len(lines) <= 3 or size(markdown) <= max_size:
        return [markdown.strip()]
    header, rows = lines[:2], lines[2:]
    parts, current = [], []
    for row in rows:
        if current and size("\n".join(header + current + [row])) > max_size:
            parts.append("\n".join(header + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(header + current))
    return parts
