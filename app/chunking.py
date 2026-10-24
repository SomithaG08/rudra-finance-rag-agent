"""
chunking.py - Turn corpus pages into chunks (the pieces that get embedded).

Two strategies:

1. "fixed"    (Day 1)  - cut every 500 characters with 100 characters overlap.
                         Simple, but it cuts words and sentences in half.
2. "sentence" (Day 2)  - split the page into sentences, then group whole
                         sentences until ~300 tokens; the last sentences of a
                         chunk are repeated at the start of the next one
                         (overlap), so an idea is never lost at a border.

Each chunk keeps its metadata: source file, PDF page, printed page number and
section title (read from the OpenStax page footer). The section title is what
lets the UI show rich citations such as
"Principles of Finance - 7.4 Applications of TVM in Finance - p. 229".
"""

from app.config import (CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, FIXED_CHUNK_CHARS,
                        FIXED_OVERLAP_CHARS, MIN_PAGE_CHARS)
from app.text_utils import count_tokens, parse_page_footer, split_sentences


def chunk_fixed(text: str, size: int = FIXED_CHUNK_CHARS, overlap: int = FIXED_OVERLAP_CHARS) -> list[str]:
    """Day-1 strategy: fixed windows of characters."""
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + size])
        start += size - overlap
    return pieces


def chunk_sentences(text: str, max_tokens: int = CHUNK_TOKENS,
                    overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Day-2 strategy: group whole sentences up to max_tokens, with sentence overlap."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        n = count_tokens(sentence)
        # A single giant "sentence" (tables, lists) is cut with the fixed method.
        if n > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            chunks.extend(chunk_fixed(sentence, size=max_tokens * 4, overlap=overlap_tokens * 4))
            continue
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(current))
            # keep the last sentences (up to overlap_tokens) as the start of the next chunk
            carry, carry_tokens = [], 0
            for s in reversed(current):
                t = count_tokens(s)
                if carry_tokens + t > overlap_tokens:
                    break
                carry.insert(0, s)
                carry_tokens += t
            current, current_tokens = carry, carry_tokens
        current.append(sentence)
        current_tokens += n

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_corpus(corpus: list[dict], strategy: str = "sentence") -> list[dict]:
    """
    Convert corpus pages into chunk dictionaries:
        {chunk_id, source, page, printed_page, section, chunk_index, text, char_count}
    """
    chunks = []
    last_section: dict[str, str] = {}          # remember the section per source

    for entry in corpus:
        source = entry.get("source", "unknown")
        page = entry.get("page", 0)
        text, section, printed_page = parse_page_footer(entry.get("text", ""))

        if section is None:
            section = last_section.get(source)   # page without footer: same section as before
        else:
            last_section[source] = section

        if len(text.strip()) < MIN_PAGE_CHARS:
            continue                               # blank / cover pages carry no knowledge

        pieces = chunk_fixed(text) if strategy == "fixed" else chunk_sentences(text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            chunks.append({
                "chunk_id": len(chunks),
                "source": source,
                "page": page,
                "printed_page": printed_page,
                "section": section,
                "chunk_index": i,
                "text": piece,
                "char_count": len(piece),
            })
    return chunks
