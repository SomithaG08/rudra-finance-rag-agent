"""
tracking.py - Cost tracker, query log and feedback log.

* Every OpenAI call is written to data/cost_log.json with its token counts
  and its price in USD, so the team can watch the $5 budget.
* Every question asked in the UI is appended to data/query_log.jsonl
  (used by the Analytics tab).
* Every thumbs-up / thumbs-down is appended to data/feedback.jsonl.
"""

import json
import threading
from collections import Counter, defaultdict
from datetime import datetime

from app.config import COST_LOG_FILE, DATA_DIR, FEEDBACK_FILE, PRICES, QUERY_LOG_FILE

_LOCK = threading.Lock()   # Gradio can run several requests at once


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------------------
def price_of(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """USD price of one call."""
    p = PRICES.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def log_cost(operation: str, model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Append one API call to data/cost_log.json and return its cost."""
    cost = price_of(model, input_tokens, output_tokens)
    entry = {
        "timestamp": _now(),
        "operation": operation,          # e.g. "embed_query", "answer", "judge"
        "model": model,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": round(cost, 8),
    }
    with _LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        log = []
        if COST_LOG_FILE.exists():
            try:
                log = json.loads(COST_LOG_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log = []
        log.append(entry)
        COST_LOG_FILE.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return cost


def _cost(entry: dict) -> float:
    """Price of one log entry (the agent scripts wrote "estimated_cost_usd" instead of "cost_usd")."""
    return float(entry.get("cost_usd", entry.get("estimated_cost_usd", 0.0)) or 0.0)


def cost_summary() -> dict:
    """Total spend and spend per operation."""
    if not COST_LOG_FILE.exists():
        return {"total_usd": 0.0, "calls": 0, "by_operation": {}}
    try:
        log = json.loads(COST_LOG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"total_usd": 0.0, "calls": 0, "by_operation": {}}
    by_op = defaultdict(lambda: {"calls": 0, "cost_usd": 0.0, "tokens": 0})
    for e in log:
        op = by_op[e.get("operation", "unknown")]
        op["calls"] += 1
        op["cost_usd"] += _cost(e)
        op["tokens"] += e.get("input_tokens", 0) + e.get("output_tokens", 0)
    return {
        "total_usd": sum(_cost(e) for e in log),
        "calls": len(log),
        "by_operation": dict(by_op),
    }


# ---------------------------------------------------------------------------
# JSON-lines helpers
# ---------------------------------------------------------------------------
def _append_jsonl(path, record: dict) -> None:
    with _LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Query log + feedback
# ---------------------------------------------------------------------------
def log_query(result: dict) -> None:
    """Store one answered question (called by the UI after each answer)."""
    _append_jsonl(QUERY_LOG_FILE, {
        "timestamp": _now(),
        "question": result.get("question"),
        "standalone_question": result.get("standalone_question"),
        "refused": result.get("refused"),
        "top_score": result.get("top_score"),
        "search_mode": result.get("settings", {}).get("mode"),
        "cited_pages": [
            f"{c['source']}#p{c['page']}" for c in result.get("citations", []) if c.get("cited")
        ],
        "latency_s": result.get("timings", {}).get("total"),
        "cached": result.get("cached") or "",
        "confidence": (result.get("confidence") or {}).get("label"),
        "tools_used": [c["name"] for c in result.get("calculations") or []],
        "language": result.get("language"),
        "cost_usd": result.get("cost_usd"),
    })


def read_queries() -> list[dict]:
    return _read_jsonl(QUERY_LOG_FILE)


def log_feedback(question: str, answer: str, helpful: bool, comment: str = "") -> None:
    _append_jsonl(FEEDBACK_FILE, {
        "timestamp": _now(),
        "question": question,
        "answer": answer,
        "helpful": helpful,
        "comment": comment,
    })


def read_feedback() -> list[dict]:
    return _read_jsonl(FEEDBACK_FILE)


def most_cited_pages(n: int = 10) -> list[tuple[str, int]]:
    counter = Counter()
    for q in read_queries():
        counter.update(q.get("cited_pages") or [])
    return counter.most_common(n)
