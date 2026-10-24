"""
summaries.py - Section summaries as an extra layer of the knowledge base.

Problem: chunks are small (~300 tokens), so a broad question such as "Summarize chapter 7"
or "What are the main ideas of section 15.2?" retrieves a few random details, never the
big picture.

Solution (a light version of hierarchical retrieval, like RAPTOR):
  * scripts/build_summaries.py writes ONE short summary per section of the book with
    gpt-4o-mini (once, ~$0.08) and embeds it -> data/summaries.json
  * for broad questions, rag.py adds the best matching section summaries to the passages,
    as passages of their own ([n] citations that open the first page of the section).

Sections come from the OpenStax page footers: odd pages say "7.4 • Applications of TVM ...",
even pages only give the chapter, so they inherit the last sub-section seen.
"""

import json
import re

import numpy as np

from app.config import DATA_DIR, source_title
from app.text_utils import parse_page_footer, tokenize

SUMMARIES_FILE = DATA_DIR / "summaries.json"

_BROAD = re.compile(
    r"\b(summar\w*|overview|outline|main (ideas?|points?|topics?|concepts?)|key (ideas?|points?|takeaways?|concepts?)"
    r"|big picture|in a nutshell|what (does|do) (the )?(chapter|section|book)|(chapter|section) \d+(\.\d+)?"
    r"|what is (this|the) (chapter|section) about|topics? (covered|of))\b", re.I)
_CHAPTER = re.compile(r"\bchapter\s+(\d{1,2})\b", re.I)
_SECTION = re.compile(r"\bsection\s+(\d{1,2}\.\d{1,2})\b", re.I)

_cache: list[dict] | None = None


# ---------------------------------------------------------------------------
# Grouping pages into sections (used by scripts/build_summaries.py)
# ---------------------------------------------------------------------------
def group_pages(corpus: list[dict]) -> list[dict]:
    """[{"id", "chapter", "section", "source", "pages", "text"}] in book order."""
    groups: dict[tuple, dict] = {}
    current: dict[str, str | None] = {}
    for e in corpus:
        src = e["source"]
        text, label, _ = parse_page_footer(e.get("text", ""))
        if label and re.match(r"\d{1,2}\.\d{1,2} ", label):
            current[src] = label                                   # new sub-section
        elif label and label.startswith("Chapter "):
            ch = re.match(r"Chapter (\d+)", label).group(1)
            cur = current.get(src)
            if not (cur and cur.split(".")[0] == ch):
                current[src] = label                               # chapter pages outside sub-sections
        label = current.get(src)
        if not label or len(text) < 50:
            continue
        key = (src, label)
        g = groups.get(key)
        if g is None:
            m = re.match(r"(?:Chapter )?(\d{1,2})(?:\.(\d{1,2}))?", label)
            sid = (f"{m.group(1)}.{m.group(2)}" if m.group(2) else f"{m.group(1)}") if m else label
            g = groups[key] = {"id": sid, "chapter": int(m.group(1)) if m else None,
                               "section": label, "source": src, "pages": [], "text": ""}
        g["pages"].append(e["page"])
        g["text"] += "\n" + text
    return list(groups.values())


# ---------------------------------------------------------------------------
# Search (used by app/rag.py)
# ---------------------------------------------------------------------------
def load_summaries() -> list[dict]:
    global _cache
    if _cache is None:
        _cache = []
        if SUMMARIES_FILE.exists():
            try:
                _cache = json.loads(SUMMARIES_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                _cache = []
    return _cache


def available() -> bool:
    return bool(load_summaries())


def is_broad_question(question: str) -> bool:
    return bool(_BROAD.search(question or ""))


def _as_passage(s: dict, score: float | None) -> dict:
    return {
        "rank": 0, "chunk_id": None, "source": s["source"], "source_title": source_title(s["source"]),
        "page": s["pages"][0], "printed_page": None,
        "section": f"Summary of {s['section']} (pages {s['pages'][0]}-{s['pages'][-1]})",
        "text": s["summary"], "score": score, "bm25": None, "rrf": 0.0, "found_by": ["summary"],
    }


def search_summaries(question: str, query_vec: np.ndarray | None = None, k: int = 2) -> list[dict]:
    """Best section summaries for a broad question (explicit chapter/section numbers first)."""
    items = load_summaries()
    if not items:
        return []
    m = _SECTION.search(question)
    sec = m.group(1) if m else None
    if sec:
        hits = [s for s in items if s["id"] == sec]
        if hits:
            return [_as_passage(s, None) for s in hits[:1]]
    m = _CHAPTER.search(question)
    if m:
        ch = int(m.group(1))
        hits = [s for s in items if s.get("chapter") == ch]
        if hits:
            return [_as_passage(s, None) for s in hits[:max(k, 4)]]
    if query_vec is not None and all(s.get("vector") for s in items):
        mat = np.array([s["vector"] for s in items], dtype="float32")
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        q = query_vec[0] / np.linalg.norm(query_vec[0])
        sims = mat @ q
        order = np.argsort(-sims)[:k]
        return [_as_passage(items[i], float(sims[i])) for i in order]
    qt = set(tokenize(question))                                    # offline: word overlap
    scored = sorted(items, key=lambda s: -len(qt & set(tokenize(s["section"] + " " + s["summary"]))))
    return [_as_passage(s, None) for s in scored[:k]]
