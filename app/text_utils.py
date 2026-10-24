"""
text_utils.py - Small text helpers shared by indexing, retrieval, UI and evaluation.

Nothing in this file calls OpenAI, so it all works offline.
"""

import difflib
import html
import re

# ---------------------------------------------------------------------------
# Token counting (tiktoken if available, otherwise ~4 characters per token)
# ---------------------------------------------------------------------------
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """Load the tiktoken encoder once. Return None if it cannot be loaded."""
    global _ENCODER, _ENCODER_TRIED
    if not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None          # no internet / not installed -> approximation
    return _ENCODER


def count_tokens(text: str) -> int:
    """Number of tokens in *text* (exact with tiktoken, approximate otherwise)."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Words for keyword (BM25) search
# ---------------------------------------------------------------------------
STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been
being this that these those it its into about over under than so such can could should would
do does did done has have had having what which who whom whose when where why how i you he she
we they them their our your my me us not no nor also very just only there here any all each
both more most other some same own too will shall may might must up down out off again further
once between during before after above below while per via tell explain describe give
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case words without stop-words. Keeps things like '2/10', 'p/e', '7.4'."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“•])")


def split_sentences(text: str) -> list[str]:
    """Split a page into sentences (good enough for textbook prose)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# OpenStax page footers -> section metadata
# ---------------------------------------------------------------------------
# Odd pages end with   "7.4 • Applications of TVM in Finance 215"
# Even pages end with  "216 7 • Time Value of Money I: Single Payment Value Access for free at openstax.org"
_ODD_FOOTER = re.compile(r"(\d{1,2}\.\d{1,2}) • (.{3,90}?) (\d{1,4})\s*$")
_EVEN_FOOTER = re.compile(
    r"(\d{1,4}) (\d{1,2}) • (.{3,110}?)\s*Access for free at openstax\.org\s*$"
)


def parse_page_footer(text: str) -> tuple[str, str | None, int | None]:
    """
    Remove the running footer of an OpenStax page and turn it into metadata.

    Returns (text_without_footer, section_label, printed_page_number).
    section_label looks like "7.4 Applications of TVM in Finance" or
    "Chapter 7: Time Value of Money I: Single Payment Value".
    """
    m = _ODD_FOOTER.search(text)
    if m:
        section = f"{m.group(1)} {m.group(2).strip()}"
        return text[: m.start()].rstrip(), section, int(m.group(3))
    m = _EVEN_FOOTER.search(text)
    if m:
        section = f"Chapter {m.group(2)}: {m.group(3).strip()}"
        return text[: m.start()].rstrip(), section, int(m.group(1))
    text = text.replace("Access for free at openstax.org", " ").strip()
    return text, None, None


# ---------------------------------------------------------------------------
# Source names
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def resolve_source(name: str | None, known_sources: list[str]) -> str | None:
    """
    Map a file name written by hand (e.g. in questions.json) to the real file
    name used in chunks.json. 'Principles_of_Finance-WEB.pdf' and
    'principles-finance_-_WEB.pdf' are the same book, so they must match.
    """
    if not name:
        return None
    if name in known_sources:
        return name
    target = _norm(name)
    best, best_ratio = None, 0.0
    for src in known_sources:
        ratio = difflib.SequenceMatcher(None, target, _norm(src)).ratio()
        if ratio > best_ratio:
            best, best_ratio = src, ratio
    if best_ratio >= 0.75 or len(known_sources) == 1:
        return best
    return name


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def highlight(text: str, query: str, max_len: int = 600) -> str:
    """HTML-escape *text*, cut it to max_len and wrap query words in <mark>."""
    snippet = text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + " ..."
    safe = html.escape(snippet)
    words = sorted({w for w in tokenize(query) if len(w) > 2}, key=len, reverse=True)
    for w in words:
        safe = re.sub(rf"(?i)\b({re.escape(w)}\w*)", r"<mark>\1</mark>", safe)
    return safe
