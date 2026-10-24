"""
rag.py - The complete RAG pipeline, used by the UI and by the evaluation scripts.

    question (+ chat history)
        -> 1. condense follow-up into a standalone question      (memory)
        -> 2. optional query expansion (2 paraphrases)
        -> 3. hybrid retrieval: BM25 + FAISS + RRF, diversity cap
        -> 4. confidence check: best cosine similarity < threshold -> refuse
        -> 5. gpt-4o-mini writes the answer with [n] citations      (streamed)
        -> 6. citations are parsed and attached to the passages

run() is a generator of events so the UI can show progress:
    {"type": "status", "text": "..."}        loading indicator
    {"type": "token",  "text": "..."}        a piece of the answer
    {"type": "done",   "result": {...}}      the final result
answer() runs the same pipeline and returns only the final result.
"""

import re
import time

from app.config import (CONFIDENCE_THRESHOLD, DEFAULT_SEARCH_MODE, DEFAULT_TOP_K, INJECTION_TEST,
                        PROMPT_VERSION, REFUSAL_TEXT, RERANK_THRESHOLD, USE_BM25, USE_HYDE,
                        USE_QUERY_REWRITE, USE_RERANK, has_api_key)
from app import tracing
from app import features, summaries
from app.evidence import find_evidence
from app.generator import (build_messages, classify_question, condense_question, expand_query,
                           extractive_answer, hyde_passage, is_refusal, parse_citations, stream_answer,
                           unverified_citations)
from app.retriever import get_retriever
from app.tracking import cost_summary

DEFAULT_SETTINGS = {
    "top_k": DEFAULT_TOP_K,
    "mode": DEFAULT_SEARCH_MODE if USE_BM25 else "dense",   # hybrid | dense | bm25
    "threshold": CONFIDENCE_THRESHOLD,  # cosine threshold (used when reranking is off)
    "rerank_threshold": RERANK_THRESHOLD,  # cross-encoder threshold (used when reranking is on)
    "expand": False,                    # query expansion
    "rerank": USE_RERANK,               # cross-encoder rerank (sentence-transformers)
    "rewrite": USE_QUERY_REWRITE,       # rewrite follow-ups before searching
    "hyde": USE_HYDE,                   # search with a hypothetical answer
    "injection": INJECTION_TEST,        # add the prompt-injection test chunk
    "source_filter": "All sources",
    "prompt_version": PROMPT_VERSION,
    # extra features (app/features.py)
    "style": "Standard",                # Standard | Simple (beginner) | Detailed | Summary (3 bullets)
    "language": "Auto",                 # Auto | English | Francais
    "tools": True,                      # calculator tools for calculation questions
    "verify": False,                    # citation check (+1 LLM call)
    "followups": True,                  # 3 suggested follow-up questions (+1 small LLM call)
    "use_cache": True,                  # exact + semantic answer cache
    "summaries": True,                  # section summaries for broad questions (data/summaries.json)
    "profile": None,                    # personal profile mode (income, savings, goal ...)
}


def run(question: str, history: list[dict] | None = None, settings: dict | None = None):
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    if s.get("injection"):
        s["use_cache"] = False          # test runs must never read or write the answer cache
    profile_on = features.profile_active(s.get("profile"))
    if profile_on:
        s["use_cache"] = False          # answers depend on the personal numbers: never cache them
    t0 = time.time()
    cost_before = cost_summary()["total_usd"]
    offline = not has_api_key()
    retriever = get_retriever(injection=bool(s.get("injection")))

    # 1. Conversation memory: rewrite the follow-up into a standalone question (for SEARCH);
    #    the answer is still written from the original question + the recent history.
    standalone = question
    if history and s.get("rewrite", True) and not offline:
        yield {"type": "status", "text": "Rewriting the follow-up into a standalone question..."}
        standalone = condense_question(question, history)
    elif history:
        standalone = condense_question(question, history) if offline else question

    # 1b. Answer cache (only for a first question: a follow-up depends on the conversation)
    if s["use_cache"] and not history:
        cached, how = features.cache_lookup(standalone, s)
        if cached:
            yield {"type": "status", "text": f"Answer found in the cache ({how} match) - $0"}
            yield {"type": "token", "text": cached["answer"]}
            yield {"type": "done", "result": {**cached, "question": question, "cached": how,
                                              "cost_usd": 0.0,
                                              "timings": {"retrieval": 0, "generation": 0,
                                                          "total": round(time.time() - t0, 2)}}}
            return

    # 1c. Language: search in English, answer in the user's language
    language = features.detect_language(question) if s["language"] == "Auto" else \
        ("fr" if s["language"].startswith("Fr") else "en")
    search_query = standalone
    if language == "fr" and not offline:
        yield {"type": "status", "text": "Translating the question to search the English book..."}
        search_query = features.translate_for_search(standalone)

    # 2. Query expansion
    expansions = []
    if s["expand"] and not offline:
        yield {"type": "status", "text": "Expanding the question (2 paraphrases)..."}
        expansions = expand_query(search_query)

    # 2b. HyDE: search with a hypothetical textbook passage instead of the question
    hyde_text = None
    if s.get("hyde") and not offline:
        yield {"type": "status", "text": "HyDE: writing a hypothetical passage to search with..."}
        hyde_text = hyde_passage(search_query)

    # 3. Retrieval (+ cross-encoder reranking)
    mode = s["mode"] if retriever.dense_available else "bm25"
    yield {"type": "status", "text": f"Searching ({mode}) and reranking the best candidates..."
           if s["rerank"] else f"Embedding the question and searching ({mode})..."}
    t1 = time.time()
    search = retriever.search(search_query, k=int(s["top_k"]), mode=mode,
                              source_filter=s["source_filter"], expansions=expansions,
                              rerank=s["rerank"], dense_text=hyde_text)
    passages, top_score = search["passages"], search["top_score"]
    top_rerank, reranked = search.get("top_rerank"), search.get("reranked", False)

    # 3b. Broad question ("summarize chapter 7") -> add section summaries as passages
    summary_count = 0
    if s.get("summaries", True) and summaries.available() and summaries.is_broad_question(search_query):
        qvec = retriever._query_vector(search_query) if retriever.dense_available else None
        strong = re.search(r"summar|overview|main|key|outline|big picture|nutshell", search_query, re.I)
        extra_passages = summaries.search_summaries(search_query, qvec, k=4 if strong else 2)
        if extra_passages:
            yield {"type": "status", "text": f"Broad question: adding {len(extra_passages)} section summaries..."}
            passages = extra_passages + passages[: max(2, int(s["top_k"]) - len(extra_passages))]
            for i, p in enumerate(passages, start=1):
                p["rank"] = i
            sims = [p["score"] for p in extra_passages if p["score"] is not None]
            if top_score is not None:
                # an explicit chapter/section match is a confident match
                top_score = max(top_score, max(sims) if sims else float(s["threshold"]))
            summary_count = len(extra_passages)
    t_retrieval = time.time() - t1

    # 4. Confidence threshold (saves an LLM call on clearly off-topic questions).
    #    With reranking, the cross-encoder score decides (it produces the final ranking;
    #    cosine and RRF scores are on other scales). Without it, the best cosine similarity.
    if reranked and top_rerank is not None and not summary_count:
        low_confidence = top_rerank < float(s["rerank_threshold"]) or not passages
    else:
        low_confidence = (top_score is not None and top_score < float(s["threshold"])) or not passages
    if top_score is None and retriever.bm25.coverage(search_query) < 0.75:
        low_confidence = True        # offline mode: too many question words unknown to the corpus
    personal = profile_on and features.is_personal(question)
    if personal and passages:
        low_confidence = False       # "my savings..." questions: the prompt still refuses if off-topic

    # 5. Generation
    t2 = time.time()
    question_type = classify_question(search_query)
    extra = "\n\n".join(x for x in [features.STYLES.get(s["style"], ""),
                                     features.language_instruction(s["language"], question),
                                     features.profile_instruction(s.get("profile")) if profile_on else ""]
                        if x)
    use_tools = bool(s["tools"]) and (question_type == "calculation" or personal)
    calculations = []
    if low_confidence:
        answer = REFUSAL_TEXT
        yield {"type": "token", "text": answer}
    elif offline:
        answer = extractive_answer(standalone, passages)
        yield {"type": "token", "text": answer}
    elif use_tools:
        yield {"type": "status", "text": "Calculation question: the model can use the calculator tools..."}
        answer = ""
        msgs = build_messages(standalone, passages, history, s["prompt_version"],
                              extra + "\n\n" + features.TOOLS_INSTRUCTION)
        for ev in features.answer_with_tools(msgs):
            if ev["type"] == "tool":
                calculations.append(ev)
                yield {"type": "status", "text": f"Calculated with {ev['name']}: {ev['result']}"}
            else:
                answer += ev["text"]
                yield {"type": "token", "text": ev["text"]}
    else:
        yield {"type": "status", "text": f"Writing the answer from {len(passages)} passages..."}
        answer = ""
        for piece in stream_answer(standalone, passages, history, s["prompt_version"], extra):
            answer += piece
            yield {"type": "token", "text": piece}
    t_generation = time.time() - t2

    # 6. Citations
    refused = low_confidence or is_refusal(answer)
    cited = [] if refused else parse_citations(answer, len(passages))
    citations = []
    for n, p in enumerate(passages, start=1):
        citations.append({**p, "n": n, "cited": n in cited})
    evidence = {} if refused else find_evidence(answer, citations)     # sentence-level proof
    unverified = [] if refused else unverified_citations(answer, len(passages))   # cited, never sent
    injection_hit = any(p.get("injection") for p in passages)

    # 7. Checks and suggestions (extra features)
    verification, followups = {}, []
    if not refused and not offline:
        if s["verify"]:
            yield {"type": "status", "text": "Checking every citation against its passage..."}
            verification = features.verify_citations(answer, citations)
        if s["followups"]:
            followups = features.suggest_followups(standalone, answer, passages, language)

    result = {
        "question": question,
        "standalone_question": standalone,
        "search_query": search_query,
        "language": language,
        "calculations": calculations,
        "evidence": evidence,
        "unverified_citations": unverified,
        "injection_chunk_retrieved": injection_hit,
        "top_rerank": top_rerank,
        "reranked": reranked,
        "hyde_text": hyde_text,
        "summaries_used": summary_count,
        "profile_used": bool(personal or profile_on),
        "personal_question": bool(personal),
        "verification": verification,
        "followups": followups,
        "cached": "",
        "expansions": expansions,
        "answer": answer,
        "refused": refused,
        "low_confidence": low_confidence,
        "cited_order": cited,
        "citations": citations,
        "top_score": top_score,
        "question_type": question_type,
        "offline": offline,
        "settings": {**s, "mode": search["mode"]},
        "timings": {"retrieval": round(t_retrieval, 2), "generation": round(t_generation, 2),
                    "total": round(time.time() - t0, 2)},
        "cost_usd": round(cost_summary()["total_usd"] - cost_before, 6),
    }
    result["confidence"] = features.confidence(result)
    if s["use_cache"] and not history:
        features.cache_store(standalone, s, result)
    yield {"type": "done", "result": result}


@tracing.observe(name="rag_answer")
def answer(question: str, history: list[dict] | None = None, settings: dict | None = None) -> dict:
    """Run the pipeline without streaming and return the final result dict."""
    result = None
    for event in run(question, history, settings):
        if event["type"] == "done":
            result = event["result"]
    return result
