"""
generator.py - Everything that asks gpt-4o-mini to write something.

  * build_messages()      : system prompt (versioned, in prompts/) + dynamic hint
                            + conversation memory + numbered passages + question
  * generate_answer()     : one full answer (used by eval scripts)
  * stream_answer()       : same, but yields the text while it is written (UI)
  * condense_question()   : turns a follow-up ("and at 6%?") into a standalone
                            question using the chat history (conversation memory)
  * expand_query()        : 2 paraphrases of the question (query expansion)
  * judge_answer()        : LLM-as-judge used by scripts/eval_runner.py
  * extractive_answer()   : offline fallback when no API key is configured
  * parse_citations()     : finds the [n] markers the model wrote
"""

import json
import re

from app.config import MAX_HISTORY_TURNS, PROMPT_VERSION, PROMPTS_DIR, REFUSAL_TEXT
from app.text_utils import split_sentences, tokenize

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(version: str = PROMPT_VERSION) -> str:
    """Read prompts/<version>.txt (cached)."""
    if version not in _PROMPT_CACHE:
        path = PROMPTS_DIR / f"{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt {path} not found")
        _PROMPT_CACHE[version] = path.read_text(encoding="utf-8").strip()
    return _PROMPT_CACHE[version]


# Dynamic prompts: an extra instruction depending on the kind of question.
TYPE_HINTS = {
    "yes_no": "The question is a yes/no question: start the answer with 'Yes' or 'No' "
              "(or 'It depends'), then explain with citations.",
    "calculation": "The question involves numbers: give the formula, plug in the values "
                   "step by step and state the final result clearly, with citations.",
    "comparison": "The question compares things: present each side and then the key "
                  "differences (a short bullet list per side is fine), with citations.",
    "definition": "The question asks for a definition: give the definition first, then "
                  "one sentence on why it matters or an example from the passages.",
    "list": "The question asks for several items: answer with a numbered list, one "
            "item per line, each with its citation.",
    "general": "",
}


def classify_question(question: str) -> str:
    """Very small rule-based classifier used to pick the dynamic hint."""
    q = question.lower().strip()
    q = re.sub(r"\b(chapter|section|page|figure|table|example)s?\s+\d+(\.\d+)*", " ", q)  # not a calculation
    if re.match(r"^(is|are|do|does|did|can|could|should|would|will|has|have|was|were)\b", q):
        return "yes_no"
    if re.search(r"\d|calculate|compute|how (long|much|many)|what is the (npv|irr|rate|value)", q):
        return "calculation"
    if re.search(r"differ|difference|compare|comparison|versus|\bvs\.?\b|better than", q):
        return "comparison"
    if re.search(r"\b(three|four|five|several|main|list|types of|forms of|steps|reasons)\b", q):
        return "list"
    if re.match(r"^(what is|what are|what does|define|meaning of|what's)\b", q):
        return "definition"
    return "general"


def format_passage_header(p: dict) -> str:
    parts = [p.get("source_title") or p["source"]]
    if p.get("section"):
        parts.append(p["section"])
    page = f"PDF page {p['page']}"
    if p.get("printed_page"):
        page += f" (printed page {p['printed_page']})"
    parts.append(page)
    return " | ".join(parts)


def build_context(passages: list[dict], delimit: bool = False) -> str:
    """Numbered passages. delimit=True wraps them in <context>...</context> (prompt v5+)."""
    body = "\n\n".join(
        f"[{i}] {format_passage_header(p)}\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    return f"<context>\n{body}\n</context>" if delimit else body


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text)


def build_messages(question: str, passages: list[dict], history: list[dict] | None = None,
                   prompt_version: str = PROMPT_VERSION, extra: str = "") -> list[dict]:
    """
    history = list of {"role": "user"|"assistant", "content": str} (Gradio 'messages' format).
    Only the last MAX_HISTORY_TURNS question/answer pairs are kept (memory window);
    citation numbers of old answers are removed because they refer to OLD passages.
    """
    system = load_prompt(prompt_version)
    hint = TYPE_HINTS[classify_question(question)]
    if hint and prompt_version != "v1":
        system += "\n\nADDITIONAL INSTRUCTION FOR THIS QUESTION\n" + hint
    if extra:
        system += "\n\n" + extra          # answer style, language, tools (app/features.py)

    messages = [{"role": "system", "content": system}]
    for turn in (history or [])[-2 * MAX_HISTORY_TURNS:]:
        content = turn.get("content")
        if isinstance(content, str) and turn.get("role") in ("user", "assistant"):
            content = _strip_citations(content.split("\n\n**Sources")[0])[:1200]
            messages.append({"role": turn["role"], "content": content})

    messages.append({
        "role": "user",
        "content": f"Passages:\n\n{build_context(passages, delimit='<context>' in system)}"
                   f"\n\nQuestion: {question}",
    })
    return messages


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------
_CITE_RE = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")


def parse_citations(answer: str, n_passages: int) -> list[int]:
    """Numbers n written as [n] (or [1, 3]) in the answer, in order of first use."""
    found = []
    for group in _CITE_RE.findall(answer):
        for num in re.split(r"[,;]", group):
            n = int(num.strip())
            if 1 <= n <= n_passages and n not in found:
                found.append(n)
    return found


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return ("don't have enough information" in a or "do not have enough information" in a
            or "don't have information about this" in a) and len(a) < 220


def generate_answer(question: str, passages: list[dict], history: list[dict] | None = None,
                    prompt_version: str = PROMPT_VERSION) -> dict:
    """Non-streaming answer. Returns {"answer", "cited", "refused", "question_type"}."""
    from app.llm import chat
    if not passages:
        return {"answer": REFUSAL_TEXT, "cited": [], "refused": True, "question_type": "none"}
    text = chat(build_messages(question, passages, history, prompt_version), operation="answer")
    return {
        "answer": text,
        "cited": parse_citations(text, len(passages)),
        "refused": is_refusal(text),
        "question_type": classify_question(question),
    }


def stream_answer(question: str, passages: list[dict], history: list[dict] | None = None,
                  prompt_version: str = PROMPT_VERSION, extra: str = ""):
    """Yield the answer text progressively (used by the Gradio chat)."""
    from app.llm import chat_stream
    yield from chat_stream(build_messages(question, passages, history, prompt_version, extra))


# ---------------------------------------------------------------------------
# Conversation memory: rewrite follow-up questions
# ---------------------------------------------------------------------------
_FOLLOW_UP = re.compile(
    r"\b(it|its|they|them|this|that|these|those|he|she|the second|the first|the last|"
    r"above|previous|same|also|what about|and if|how about|why)\b", re.I)


def needs_condensing(question: str, history: list[dict] | None) -> bool:
    if not history:
        return False
    return bool(_FOLLOW_UP.search(question)) or len(tokenize(question)) <= 4


def condense_question(question: str, history: list[dict] | None) -> str:
    """
    "And at 6%?" + history about the rule of 72  ->
    "Using the rule of 72, how long does money take to double at 6%?"
    The standalone question is what we SEARCH with, so retrieval works for follow-ups.
    """
    from app.config import USE_QUERY_REWRITE
    if not history:
        return question                      # first turn: nothing to resolve, no call
    if not USE_QUERY_REWRITE and not needs_condensing(question, history):
        return question
    last = [t for t in history if isinstance(t.get("content"), str)][-2 * MAX_HISTORY_TURNS:]
    # only the start of long assistant answers: the call stays cheap
    convo = "\n".join(f"{t['role'].upper()}: {_strip_citations(t['content'])[:300 if t['role'] == 'assistant' else 600]}"
                       for t in last if t["content"])
    from app.config import has_api_key
    if not has_api_key():
        prev_q = next((t["content"] for t in reversed(last) if t["role"] == "user"), "")
        return f"{prev_q} {question}".strip()
    from app.llm import chat
    rewritten = chat([
        {"role": "system", "content":
            "Rewrite the user's last question so it can be understood WITHOUT the conversation "
            "(replace pronouns and references with what they refer to). If it is already "
            "standalone, return it unchanged. Return only the question."},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nLast question: {question}"},
    ], operation="condense_question", max_tokens=120)
    return rewritten.strip().strip('"') or question


# ---------------------------------------------------------------------------
# HyDE: hypothetical document embeddings
# ---------------------------------------------------------------------------
def hyde_passage(question: str) -> str:
    """
    A short hypothetical textbook passage answering the question. It is only used for the
    VECTOR search (an answer looks more like a corpus passage than a question does); it is
    never shown to the user and never used as a fact. One extra small call per question.
    """
    from app.llm import chat
    return chat([
        {"role": "system", "content":
            "Write a short passage (3-4 sentences) as it could appear in a finance textbook, "
            "answering the question. Use textbook vocabulary. No introduction."},
        {"role": "user", "content": question},
    ], operation="hyde", temperature=0, max_tokens=160)


# ---------------------------------------------------------------------------
# Citation check (unverified sources)
# ---------------------------------------------------------------------------
def unverified_citations(answer: str, n_passages: int) -> list[int]:
    """[n] numbers written in the answer that do NOT match a passage we actually sent."""
    bad = []
    for m in _CITE_RE.finditer(answer):
        for x in re.split(r"[,;]\s*", m.group(1)):
            n = int(x)
            if (n < 1 or n > n_passages) and n not in bad:
                bad.append(n)
    return bad


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
def expand_query(question: str, n: int = 2) -> list[str]:
    """Ask the model for n paraphrases using textbook vocabulary."""
    from app.llm import chat
    raw = chat([
        {"role": "system", "content":
            f"Write {n} alternative phrasings of the user's question, using the vocabulary a "
            "finance textbook would use (expand acronyms, use formal terms). "
            'Return JSON: {"queries": ["...", "..."]}'},
        {"role": "user", "content": question},
    ], operation="expand_query", json_mode=True, temperature=0.3, max_tokens=200)
    try:
        queries = json.loads(raw).get("queries", [])
        return [q for q in queries if isinstance(q, str) and q.strip()][:n]
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Offline fallback (no API key): extractive answer
# ---------------------------------------------------------------------------
def extractive_answer(question: str, passages: list[dict], max_sentences: int = 4) -> str:
    """Pick the sentences that share the most words with the question, with citations."""
    q_words = set(tokenize(question))
    scored = []
    for n, p in enumerate(passages[:5], start=1):
        for s in split_sentences(p["text"]):
            overlap = len(q_words & set(tokenize(s)))
            if overlap and 40 < len(s) < 400:
                scored.append((overlap, -n, s, n))
    if not scored:
        return REFUSAL_TEXT
    best = sorted(scored, reverse=True)[:max_sentences]
    lines = [f"- {s} [{n}]" for _, _, s, n in best]
    return ("*Offline mode (no OpenAI key): the most relevant sentences from the "
            "knowledge base are shown below instead of a generated answer.*\n\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# LLM-as-judge (evaluation)
# ---------------------------------------------------------------------------
def judge_answer(question: str, expected: str | None, answer: str, passages: list[dict],
                 category: str) -> dict:
    """Score an answer 1-5 on correctness, faithfulness, relevancy, completeness + a pass/fail verdict."""
    from app.llm import chat
    rubric = (
        "You grade answers of a RAG assistant about a finance textbook.\n"
        "Return JSON with keys: correct (true/false), correctness (1-5), faithfulness (1-5: "
        "every claim supported by the passages), relevancy (1-5: the answer addresses the question "
        "that was asked), completeness (1-5), reason (one sentence).\n"
        "Rules: for category 'out-of-scope' the answer is correct only if it refuses; the "
        "assistant's standard refusal sentence (\"I don't have enough information to answer this "
        "question.\") IS a correct refusal, it does not need to explain why. For "
        "category 'ambiguous' the answer is correct if it gives an overview AND/OR asks a "
        "clarifying question instead of inventing one specific interpretation. Otherwise "
        "compare with the expected answer: same key facts and numbers = correct."
    )
    if expected is None and category == "ambiguous":
        expected = ("(no single answer) The question is vague: a good answer gives a short overview of "
                    "the main meanings found in the passages and/or asks ONE clarifying question.")
    elif expected is None and category == "out-of-scope":
        expected = "(refusal expected) The knowledge base does not cover this question."
    content = (f"Category: {category}\nQuestion: {question}\nExpected answer: {expected}\n\n"
               f"Passages:\n{build_context(passages)[:6000]}\n\nAnswer to grade:\n{answer}")
    raw = chat([{"role": "system", "content": rubric}, {"role": "user", "content": content}],
               operation="judge", json_mode=True, max_tokens=250)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"correct": None, "reason": "judge returned invalid JSON"}
