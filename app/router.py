"""
router.py - Router extension (Agentic AI project, "going further").

Classify each question BEFORE spending model calls:
    simple lookup  -> the RAG pipeline (1 search + 1 generation call, like the Chat tab)
    anything else  -> the agent loop (tools, several model calls)
    RAG refused    -> escalated to the agent, which can rephrase and search again (router v3)

The router is rule-based on purpose: it costs $0 and 0 ms, and every decision has a readable
reason that is shown in the trace ("route -> RAG: single lookup, no number / action / section").
Measured with:  python scripts/agent_eval.py --router --tag router-v1 --compare-rag
"""

import re
import time

from app.agent import is_chitchat, wants_save, wants_send

RULES = [
    ("action requested (send / share)", lambda q: wants_send(q)),
    ("action requested (save a note)", lambda q: wants_save(q)),
    ("greeting / capabilities (answered directly)", lambda q: is_chitchat(q)),
    ("numbers to compute", lambda q: bool(re.search(r"\d+(\.\d+)?\s*%|\b\d{2,}\b", q))
     or bool(re.search(r"\b(calculate|compute|how many years|how long|how much)\b", q, re.I))),
    ("chapter / section / page", lambda q: bool(re.search(r"\b(chapter|section|page|contents)\b\s*\d*", q, re.I))),
    ("book glossary requested", lambda q: bool(re.search(r"\b(glossary|key terms?|book'?s definition|define)\b", q, re.I))),
    ("cash-flow analysis", lambda q: bool(re.search(r"\b(npv|irr|payback|profitability index)\b", q, re.I))
     and bool(re.search(r"\d", q))),
    ("several questions in one", lambda q: q.count("?") > 1
     or bool(re.search(r"\b(compare|difference between|versus|vs\.?)\b", q, re.I))),
]


def route(question: str) -> dict:
    """{"route": "agent" | "rag", "reason": str}"""
    q = question or ""
    for reason, test in RULES:
        if test(q):
            return {"route": "agent", "reason": reason}
    return {"route": "rag", "reason": "single lookup: no number, action, section or multi-part request"}


def rag_as_agent_result(question: str, res: dict, reason: str, latency: float) -> dict:
    """RAG result with the fields the agent UI / eval expect (one search step, no loop)."""
    trace = [f"route -> RAG pipeline ({reason})",
             f"1. search_corpus(query={question[:80]!r}) -> {len(res.get('citations') or [])} passage(s)"
             + (f", top rerank {res['top_rerank']:.2f}" if res.get("top_rerank") is not None else ""),
             "2. answer (RAG generator)"]
    return {**res,
            "trajectory": [], "trace": trace, "stop_reason": "rag_route", "route": "rag",
            "route_reason": reason, "approval_required": False, "pending_message": {},
            "send_proposals": [], "tools_used": ["search_corpus"], "steps": 1, "repeated_calls": 0,
            "hit_step_cap": False, "latency_s": round(latency, 2), "plan": None, "nudges": 0,
            "model_calls": None, "tokens": {}, "prompt_version": "rag"}


def run_routed(question: str, history: list[dict] | None = None, **agent_kwargs) -> dict:
    """Route, then run the RAG pipeline or the agent. Same result shape in both cases."""
    decision = route(question)
    if decision["route"] == "rag":
        from app import rag
        t0 = time.time()
        res = rag.answer(question, history=history, settings={"use_cache": False, "followups": False})
        if not res.get("refused"):
            return rag_as_agent_result(question, res, decision["reason"], time.time() - t0)
        decision = {"route": "agent", "reason": "RAG refused -> escalated to the agent (it can rephrase and retry)"}
        rag_seconds = time.time() - t0
    else:
        rag_seconds = 0.0
    from app.agent import run_agent
    r = run_agent(question, history, **agent_kwargs)
    r["route"], r["route_reason"] = "agent", decision["reason"]
    r["trace"] = [f"route -> agent ({decision['reason']})"] + r["trace"]
    r["latency_s"] = round(r["latency_s"] + rag_seconds, 2)
    return r
