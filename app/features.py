"""
features.py - Extra features on top of the core RAG pipeline.

  1. Answer styles        : standard / simple (beginner, with an analogy) / detailed / summary
  2. French support       : a French question is translated to English for SEARCH
                            (the book is in English), the answer is written in French
  3. Calculator tools     : function calling with app/tools.py for calculation questions
  4. Confidence badge     : high / medium / low, computed from scores and citations (free)
  5. Citation check       : a second gpt-4o-mini call checks every cited sentence against
                            its passage and flags unsupported claims (optional)
  6. Follow-up questions  : 3 suggested next questions after each answer
  7. Answer cache         : exact AND semantic (same meaning) cache of answers -> $0
  8. Quiz generator       : multiple-choice questions from a chapter/section or topic,
                            each with the page it comes from
"""

import json
import re
import threading

import numpy as np

from app.config import DATA_DIR, has_api_key

# ---------------------------------------------------------------------------
# 1. Answer styles
# ---------------------------------------------------------------------------
STYLES = {
    "Standard": "",
    "Simple (beginner)": (
        "ANSWER STYLE: the reader is a beginner. Use short sentences and everyday words, "
        "explain every technical term, and add one concrete everyday analogy or small "
        "numeric example (the analogy does not need a citation, facts still do)."),
    "Detailed": (
        "ANSWER STYLE: give a thorough answer: definition, how it works, formula if any, "
        "a worked example from the passages, and limitations - each with citations."),
    "Summary (3 bullets)": (
        "ANSWER STYLE: answer in at most 3 short bullet points, each with its citation."),
}

# ---------------------------------------------------------------------------
# 2. Language
# ---------------------------------------------------------------------------
_FR_WORDS = set("le la les un une des du de est sont quel quelle quels quelles comment pourquoi "
                "qu'est-ce que qui dans avec pour sur taux intérêt entreprise combien ce cette "
                "au aux ou où et mon ma mes notre votre".split())


def detect_language(text: str) -> str:
    """'fr' when the question looks French (accents or common French words), else 'en'."""
    words = re.findall(r"[a-zà-ÿ'-]+", text.lower())
    if not words:
        return "en"
    fr = sum(w in _FR_WORDS for w in words) + 2 * len(re.findall(r"[éèêàùçôî]", text.lower()))
    return "fr" if fr / len(words) >= 0.25 else "en"


def language_instruction(choice: str, question: str) -> str:
    lang = detect_language(question) if choice == "Auto" else ("fr" if choice.startswith("Fr") else "en")
    if lang == "fr":
        return ("LANGUAGE: write the whole answer in French (the passages are in English: translate "
                "faithfully, keep the [n] citations). If you must refuse, write exactly: "
                "I don't have enough information to answer this question.")
    return ""


def translate_for_search(question: str) -> str:
    """French question -> English search query (the embeddings and BM25 work best in English)."""
    from app.llm import chat
    return chat([
        {"role": "system", "content": "Translate the user's question into English for searching an "
                                      "English finance textbook. Return only the translation."},
        {"role": "user", "content": question}], operation="translate_query", max_tokens=150)


# ---------------------------------------------------------------------------
# 3. Answer with calculator tools (function calling)
# ---------------------------------------------------------------------------
TOOLS_INSTRUCTION = (
    "TOOLS: for any arithmetic (doubling time, compound interest, present/future value, EAR, NPV, "
    "IRR, payback, loan payment, ratios) call the matching tool instead of computing mentally, "
    "then use the tool result in the answer. Cite the passages for the method/formula; the numeric "
    "result comes from the tool.")


def answer_with_tools(messages: list[dict], max_rounds: int = 3):
    """
    Generator. Lets gpt-4o-mini call tools, runs them, and yields:
        {"type": "tool", "name", "arguments", "result"}   for each calculation
        {"type": "token", "text"}                          the final answer
    """
    from app.llm import chat_stream, chat_with_tools
    from app.tools import TOOL_SCHEMAS, run_tool

    msgs = list(messages)
    for _ in range(max_rounds):
        reply = chat_with_tools(msgs, TOOL_SCHEMAS)
        if not reply["tool_calls"]:
            yield {"type": "token", "text": reply["content"] or ""}
            return
        msgs.append({"role": "assistant", "content": reply["content"],
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                                    for c in reply["tool_calls"]]})
        for c in reply["tool_calls"]:
            result = run_tool(c["name"], c["arguments"])
            yield {"type": "tool", "name": c["name"], "arguments": c["arguments"], "result": result}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    for piece in chat_stream(msgs, operation="answer_after_tools"):   # final answer, streamed
        yield {"type": "token", "text": piece}


# ---------------------------------------------------------------------------
# 4. Confidence badge (no API call)
# ---------------------------------------------------------------------------
def confidence(result: dict) -> dict:
    """
    Combines three signals into high / medium / low:
      * best similarity between question and passages (retrieval quality)
      * number of passages the answer cites
      * share of answer sentences that carry a citation
    """
    if result["refused"]:
        return {"label": "n/a", "score": None, "reasons": ["question refused"]}
    reasons, score = [], 0.0
    top = result.get("top_score")
    if top is not None:
        score += min(1.0, max(0.0, (top - 0.35) / 0.35)) * 0.45
        reasons.append(f"best similarity {top:.2f}")
    else:
        score += 0.2
        reasons.append("keyword search only")
    cited = len(result.get("cited_order", []))
    score += min(cited, 3) / 3 * 0.30
    reasons.append(f"{cited} passage(s) cited")
    body = result["answer"].split("\n\n**Sources")[0]
    sentences = [x for x in re.split(r"(?<=[.!?])\s+|\n+", body) if len(x.strip()) > 25]
    with_cite = sum(bool(re.search(r"\[\d", x)) for x in sentences)
    share = with_cite / len(sentences) if sentences else 0
    score += share * 0.25
    reasons.append(f"{share:.0%} of sentences cited")
    if result.get("verification", {}).get("unsupported"):
        score -= 0.25
        reasons.append("unsupported claim flagged")
    label = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    return {"label": label, "score": round(score, 2), "reasons": reasons}


# ---------------------------------------------------------------------------
# 5. Citation check (LLM verifier)
# ---------------------------------------------------------------------------
def verify_citations(answer: str, citations: list[dict]) -> dict:
    """Ask gpt-4o-mini whether each cited sentence is supported by the passage(s) it cites."""
    from app.llm import chat
    passages = "\n\n".join(f"[{c['n']}] {c['text']}" for c in citations if c["cited"])
    if not passages:
        return {"checked": False, "unsupported": []}
    raw = chat([
        {"role": "system", "content":
            "You check citations. For each sentence of the ANSWER that contains [n], decide if the "
            "cited passage(s) support it. Ignore sentences without citations and simple arithmetic "
            "from a calculator. Return JSON: {\"unsupported\": [{\"sentence\": \"...\", "
            "\"reason\": \"...\"}]} (empty list if everything is supported)."},
        {"role": "user", "content": f"PASSAGES:\n{passages[:9000]}\n\nANSWER:\n{answer}"}],
        operation="verify_citations", json_mode=True, max_tokens=400)
    try:
        data = json.loads(raw)
        return {"checked": True, "unsupported": data.get("unsupported", [])[:5]}
    except json.JSONDecodeError:
        return {"checked": False, "unsupported": []}


# ---------------------------------------------------------------------------
# 6. Suggested follow-up questions
# ---------------------------------------------------------------------------
def suggest_followups(question: str, answer: str, passages: list[dict], language: str = "en") -> list[str]:
    from app.llm import chat
    context = "\n".join(p["text"][:300] for p in passages[:5])
    lang = " Write them in French." if language == "fr" else ""
    raw = chat([
        {"role": "system", "content":
            "Suggest 3 short follow-up questions a student could ask next, answerable from the "
            f"textbook excerpts given (not generic).{lang} Return JSON: {{\"questions\": [\"...\"]}}"},
        {"role": "user", "content": f"Question: {question}\nAnswer: {answer[:800]}\nExcerpts:\n{context}"}],
        operation="followups", json_mode=True, temperature=0.4, max_tokens=150)
    try:
        return [q for q in json.loads(raw).get("questions", []) if isinstance(q, str)][:3]
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# 7. Answer cache (exact + semantic)
# ---------------------------------------------------------------------------
ANSWER_CACHE_FILE = DATA_DIR / "answer_cache.json"
SEMANTIC_THRESHOLD = 0.97       # cosine between two questions to count as "the same question"
_lock = threading.Lock()


def _norm(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s%/.]", " ", q.lower()).split())


def _settings_key(s: dict) -> str:
    keys = ["top_k", "mode", "threshold", "expand", "rerank", "source_filter", "prompt_version",
            "style", "language", "tools"]
    return json.dumps({k: s.get(k) for k in keys}, sort_keys=True)


def _load() -> list[dict]:
    if not ANSWER_CACHE_FILE.exists():
        return []
    try:
        return json.loads(ANSWER_CACHE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def cache_lookup(question: str, settings: dict) -> tuple[dict | None, str]:
    """Return (cached_result, 'exact'|'semantic') or (None, '')."""
    entries = [e for e in _load() if e["settings"] == _settings_key(settings)]
    q = _norm(question)
    for e in entries:
        if e["question"] == q:
            return e["result"], "exact"
    if has_api_key() and entries:
        from app.llm import embed_query
        v = embed_query(question)[0]            # cached per question: usually free
        v = v / np.linalg.norm(v)
        for e in entries:
            if e.get("vector"):
                w = np.array(e["vector"], dtype="float32")
                if float(np.dot(v, w / np.linalg.norm(w))) >= SEMANTIC_THRESHOLD:
                    return e["result"], "semantic"
    return None, ""


def cache_store(question: str, settings: dict, result: dict) -> None:
    if result.get("refused") or result.get("offline"):
        return
    vector = None
    if has_api_key():
        from app.llm import embed_query
        vector = [round(float(x), 5) for x in embed_query(question)[0]]
    with _lock:
        entries = _load()
        entries.append({"question": _norm(question), "settings": _settings_key(settings),
                        "vector": vector, "result": result})
        ANSWER_CACHE_FILE.write_text(json.dumps(entries[-300:], ensure_ascii=False), encoding="utf-8")


def cache_clear() -> None:
    if ANSWER_CACHE_FILE.exists():
        ANSWER_CACHE_FILE.write_text("[]", encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. Quiz generator
# ---------------------------------------------------------------------------
def list_sections(chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        s = c.get("section")
        if s and s not in seen and re.match(r"^\d+\.\d+ ", s):
            seen.add(s)
            out.append(s)
    return sorted(out, key=lambda s: [int(x) for x in s.split(" ")[0].split(".")])


def quiz_passages(retriever, section: str | None, topic: str | None, k: int = 8) -> list[dict]:
    if topic:
        return retriever.search(topic, k=k)["passages"]
    pool = [c for c in retriever.chunks if c.get("section") == section and len(c["text"]) > 300]
    if not pool:
        return []
    step = max(1, len(pool) // k)
    picked = pool[::step][:k]
    return [{**c, "n": i} for i, c in enumerate(picked, start=1)]


def generate_quiz(passages: list[dict], n_questions: int = 5, language: str = "en") -> list[dict]:
    """Multiple-choice questions, each tied to the passage (and page) it comes from."""
    from app.llm import chat
    text = "\n\n".join(f"[{i}] (page {p['page']}) {p['text']}" for i, p in enumerate(passages, start=1))
    lang = " Write the quiz in French." if language == "fr" else ""
    raw = chat([
        {"role": "system", "content":
            f"Create {n_questions} multiple-choice questions to test understanding of these textbook "
            "passages. Each question: 4 options, exactly one correct, plausible distractors, based ONLY "
            f"on the passages.{lang} Return JSON: {{\"quiz\": [{{\"question\": \"...\", \"options\": "
            "[\"...\", \"...\", \"...\", \"...\"], \"answer_index\": 0, \"explanation\": \"...\", "
            "\"passage\": 1}]}"},
        {"role": "user", "content": text[:12000]}],
        operation="quiz", json_mode=True, temperature=0.5, max_tokens=1800)
    try:
        quiz = json.loads(raw).get("quiz", [])
    except json.JSONDecodeError:
        return []
    out = []
    for q in quiz[:n_questions]:
        try:
            p = passages[int(q.get("passage", 1)) - 1]
            out.append({"question": q["question"], "options": q["options"][:4],
                        "answer_index": int(q["answer_index"]), "explanation": q.get("explanation", ""),
                        "source": p["source"], "page": p["page"], "section": p.get("section")})
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# 9. Personal profile mode
# ---------------------------------------------------------------------------
# The user can enter a few numbers about their own situation in the Chat tab. For questions
# about "my" situation, the answer applies the book's methods (with citations) to these numbers
# and every calculation goes through the calculator tools. Nothing is stored on disk.
PROFILE_LABELS = {
    "monthly_income": "Monthly net income",
    "monthly_expenses": "Monthly expenses",
    "savings": "Current savings",
    "monthly_saving": "Amount saved per month",
    "goal_amount": "Savings goal",
    "goal_years": "Years to reach the goal",
    "expected_return": "Expected annual return (%)",
    "risk": "Risk preference",
}
_PERSONAL = re.compile(r"\b(my|me|mine|myself|i|i'm|i've|i'd|i'll|mon|ma|mes|je|j'ai)\b", re.I)


def profile_active(profile: dict | None) -> bool:
    return bool(profile) and profile.get("enabled", True) and any(
        v not in (None, "", 0) for k, v in profile.items() if k in PROFILE_LABELS)


def is_personal(question: str) -> bool:
    """'Can I afford...', 'How long until my savings...' -> about the user's own situation."""
    return bool(_PERSONAL.search(question or ""))


def profile_instruction(profile: dict) -> str:
    currency = profile.get("currency") or "EUR"
    lines = []
    for key, label in PROFILE_LABELS.items():
        v = profile.get(key)
        if v in (None, "", 0):
            continue
        unit = "" if key in ("goal_years", "expected_return", "risk") else f" {currency}"
        lines.append(f"- {label}: {v}{unit}")
    return (
        "USER PROFILE (numbers given by the user in the app - they are NOT from the book):\n"
        + "\n".join(lines) + "\n"
        "When the question is about the user's own situation ('my', 'I', 'me'): explain the method "
        "or rule from the passages with [n] citations, then apply it to these numbers using the "
        "calculator tools for every computation, and say clearly which numbers come from the "
        "profile. When the question is general, ignore the profile. End a personal answer with the "
        "line: 'Educational example based on the textbook, not personal financial advice.'"
    )
