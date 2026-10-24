"""
agent_tools.py - The tools of the agent (Agentic AI project, day 1 + day 2).

Shared by app/agent.py (the agent loop) and scripts/mcp_server.py (the same tools exposed
over MCP). Every tool returns a dict with "success" and, on failure, a short "message" that
the model can read and act on (recovery = a clear error, never a crash).

    Required tools
      search_corpus(query)                 reranked hybrid search; "No matching passages found"
                                           when the best cross-encoder score < RERANK_THRESHOLD
      calculator(expression)               exact arithmetic with simpleeval (no eval())
    Our own tools (3, one per real need of a finance-textbook assistant)
      lookup_term(term)                    the book's own glossary definition ("Key Terms" pages)
      get_section(section)                 a chapter / section / page of the book by number or name
      cash_flow_analysis(cash_flows, rate) NPV, IRR, payback, discounted payback, PI in one call
    Action tool
      send_message(...)                    see app/message_tool.py (human approval required)

Tools that return book text give every passage a number n, so the final answer can cite [n].
"""

import json
import math
import re
from difflib import get_close_matches
from functools import lru_cache

from app.config import (CONFIDENCE_THRESHOLD, CORPUS_FILE, RERANK_THRESHOLD, USE_RERANK,
                        source_title)
from app.text_utils import parse_page_footer

AGENT_SEARCH_K = 3                    # passages returned by one search (spec: top 3 after rerank)
NO_MATCH = "No matching passages found"


# ============================================================
# Book text (corpus.json = one entry per PDF page)
# ============================================================
@lru_cache(maxsize=1)
def _corpus() -> list[dict]:
    return json.loads(CORPUS_FILE.read_text(encoding="utf-8"))


def _passage(n: int, source: str, page, text: str, section: str | None = None, **extra) -> dict:
    """Same fields as a search passage, so the Sources panel and citations treat them alike."""
    base = {"n": n, "rank": n, "chunk_id": None, "source": source, "source_title": source_title(source),
            "page": page, "printed_page": None, "section": section, "text": text, "score": None,
            "bm25": None, "rrf": 0.0, "rerank": None, "found_by": ["book tool"], "injection": False}
    return {**base, **extra}


# ============================================================
# 1. search_corpus (required)
# ============================================================
def search_corpus(query: str, first_n: int = 1, k: int = AGENT_SEARCH_K, injection: bool | None = None,
                  injection_mode: str | None = None, rerank_threshold: float | None = None) -> dict:
    """Hybrid BM25 + FAISS -> cross-encoder rerank (top-20 -> top-k) -> threshold."""
    from app.retriever import get_retriever
    query = (query or "").strip()
    if not query:
        return {"success": False, "tool": "search_corpus", "message": "Empty query: give search terms."}
    thr = RERANK_THRESHOLD if rerank_threshold is None else rerank_threshold
    r = get_retriever(injection=injection, injection_mode=injection_mode)
    res = r.search(query, k=k, rerank=USE_RERANK)
    passages = res["passages"]
    top_rr, top_cos = res.get("top_rerank"), res.get("top_score")
    if res.get("reranked"):
        below, used = (top_rr is None or top_rr < thr), f"best reranker score {top_rr:.2f} < {thr}" if top_rr is not None else ""
    else:                                                  # reranker not installed: cosine fallback
        below = top_cos is not None and top_cos < CONFIDENCE_THRESHOLD
        used = f"best similarity {top_cos:.2f} < {CONFIDENCE_THRESHOLD}" if top_cos is not None else ""
    if not passages or below:
        return {"success": False, "tool": "search_corpus", "query": query, "message": NO_MATCH,
                "reason": used, "top_rerank": top_rr, "top_score": top_cos, "results": []}
    results = [{**p, "n": first_n + i} for i, p in enumerate(passages)]
    return {"success": True, "tool": "search_corpus", "query": query, "mode": res["mode"],
            "reranked": res.get("reranked", False), "top_rerank": top_rr, "top_score": top_cos,
            "results": results}


# ============================================================
# 2. calculator (required: simpleeval)
# ============================================================
def calculator(expression: str) -> dict:
    """Exact arithmetic. simpleeval parses the expression (no Python eval, no names, no imports).
    Several expressions separated by ";" are computed in ONE call (saves agent steps)."""
    parts = [e for e in str(expression or "").split(";") if e.strip()]
    if len(parts) > 1:
        outs = [_calculate_one(e) for e in parts[:6]]
        bad = [o for o in outs if not o["success"]]
        if bad:
            return bad[0]
        return {"success": True, "tool": "calculator", "expression": "; ".join(o["expression"] for o in outs),
                "result": [o["result"] for o in outs], "engine": outs[0]["engine"]}
    return _calculate_one(expression)


def _calculate_one(expression: str) -> dict:
    expr = str(expression or "").replace("^", "**").replace(",", "").replace("$", "").strip()
    if not expr:
        return {"success": False, "tool": "calculator", "message": "Empty expression."}
    try:
        try:
            from simpleeval import SimpleEval
            ev = SimpleEval(functions={"sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                                       "abs": abs, "round": round, "min": min, "max": max})
            value = ev.eval(expr)
            engine = "simpleeval"
        except ImportError:                                  # fallback: our AST calculator
            from app.tools import calculator as ast_calc
            value, engine = ast_calc(expr)["result"], "ast"
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("the result is not a number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("the result is not finite (division by zero?)")
        return {"success": True, "tool": "calculator", "expression": expr,
                "result": round(value, 6), "engine": engine}
    except Exception as error:                               # noqa: BLE001
        return {"success": False, "tool": "calculator", "expression": expr,
                "message": f"Could not calculate: {type(error).__name__}: {error}. "
                           "Use only numbers, + - * / ** % ( ) and sqrt/log/exp/round."}


# ============================================================
# 3. lookup_term (own tool #1) - the book's glossary
# ============================================================
_END_OF_TERMS = re.compile(r"\s(CFA Institute|Multiple Choice|Review Questions|Problems|Video Activity|"
                           r"Critical Thinking)\s")


@lru_cache(maxsize=1)
def glossary() -> list[dict]:
    """
    Parse the "Key Terms" list at the end of every chapter into
    [{"term_line": "capital structure the mix of financing ...", "source", "page", "chapter"}].
    The PDF text keeps one line per printed line (separated by two spaces); an entry ends on a
    short line (the printed line did not reach the right margin).
    """
    entries, inside, buf = [], False, []
    for e in _corpus():
        if e["page"] < 25:                                   # the table of contents also says "Key Terms"
            continue
        text, label, _ = parse_page_footer(e.get("text", ""))
        if not inside:
            i = text.find("Key Terms")
            if i < 0:
                continue
            text, inside, buf = text[i + len("Key Terms"):], True, []
        m = _END_OF_TERMS.search(text)
        buf.append((e["source"], e["page"], label, text[:m.start()] if m else text))
        if not m:
            continue
        inside, cur, where = False, "", None
        for source, page, lab, seg in buf:
            for line in seg.split("  "):
                line = line.strip()
                if not line:
                    continue
                if not cur:
                    where = (source, page, lab)
                cur = f"{cur} {line}".strip()
                if len(line) < 85:
                    entries.append({"entry": cur, "source": where[0], "page": where[1], "section": where[2]})
                    cur = ""
        if cur:
            entries.append({"entry": cur, "source": where[0], "page": where[1], "section": where[2]})
    return entries


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()&/ -]", " ", s.lower())).strip()


def lookup_term(term: str, first_n: int = 1) -> dict:
    """Definition of a finance term from the book's end-of-chapter Key Terms (all chapters)."""
    t = _norm(term or "")
    if not t:
        return {"success": False, "tool": "lookup_term", "message": "Give the term to look up."}
    hits = []
    for g in glossary():
        e = _norm(g["entry"])
        abbrev = re.match(r"^[^()]{2,60}\((\w[\w&/ -]{0,12})\)", e)
        if e.startswith(t + " ") or (abbrev and abbrev.group(1) == t):
            hits.append(g)
        if len(hits) == 3:
            break
    if not hits:                                            # fuzzy match on the first words of entries
        heads = {" ".join(_norm(g["entry"]).split()[:len(t.split())]): g for g in glossary()}
        close = get_close_matches(t, list(heads), n=3, cutoff=0.8)
        hits = [heads[c] for c in close]
    if not hits:
        return {"success": False, "tool": "lookup_term", "term": term,
                "message": f"'{term}' is not in the book's Key Terms. Try search_corpus instead."}
    def where(label):
        m = re.match(r"(?:Chapter )?(\d{1,2})", label or "")
        return f"Key Terms of chapter {m.group(1)}" if m else "Key Terms"
    results = [_passage(first_n + i, h["source"], h["page"], h["entry"], where(h["section"]),
                        found_by=["glossary"])
               for i, h in enumerate(hits)]
    return {"success": True, "tool": "lookup_term", "term": term, "results": results}


# ============================================================
# 4. get_section (own tool #2) - a chapter / section / page by number or name
# ============================================================
@lru_cache(maxsize=1)
def _sections() -> list[dict]:
    from app.summaries import group_pages
    return group_pages(_corpus())


def table_of_contents() -> list[str]:
    seen, out = set(), []
    for s in _sections():
        if s["id"] not in seen:
            seen.add(s["id"])
            out.append(f"{s['section']} (pages {s['pages'][0]}-{s['pages'][-1]})")
    return out


def get_section(section: str, first_n: int = 1, max_chars: int = 2500) -> dict:
    """
    section = "7.4", "chapter 7", "page 229", "contents" or words of a title
    ("Applications of TVM"). Returns the section text (start) + its summary when built.
    """
    q = str(section or "").strip()
    ql = q.lower()
    if not q:
        return {"success": False, "tool": "get_section", "message": "Give a section number, chapter, page or title."}
    if ql in ("toc", "contents", "table of contents", "chapters", "list"):
        toc = [t for t in table_of_contents() if t.startswith("Chapter ")] or table_of_contents()
        return {"success": True, "tool": "get_section", "section": "Table of contents", "toc": toc[:60]}

    m = re.search(r"\bpage\s*(\d+)", ql) or (re.fullmatch(r"p\.?\s*(\d+)", ql))
    if m:                                                   # one page, exact stored text
        page = int(m.group(1))
        for e in _corpus():
            if e["page"] == page:
                text, label, printed = parse_page_footer(e["text"])
                return {"success": True, "tool": "get_section", "section": label or f"page {page}",
                        "pages": [page], "results": [_passage(first_n, e["source"], page, text[:max_chars],
                                                               label, printed_page=printed,
                                                               found_by=["get_section"])]}
        return {"success": False, "tool": "get_section", "message": f"Page {page} does not exist."}

    secs = _sections()
    m = re.search(r"(\d{1,2})\.(\d{1,2})", ql)
    ch = re.search(r"chapter\s*(\d{1,2})", ql) or re.fullmatch(r"(\d{1,2})", ql)
    if m:
        hits = [s for s in secs if s["id"] == f"{int(m.group(1))}.{int(m.group(2))}"]
    elif ch:
        hits = [s for s in secs if s["chapter"] == int(ch.group(1))]
    else:                                                   # by title words
        words = set(re.findall(r"[a-z]{3,}", ql))
        scored = sorted(((len(words & set(re.findall(r"[a-z]{3,}", s["section"].lower()))), s)
                         for s in secs), key=lambda x: -x[0])
        hits = [scored[0][1]] if scored and scored[0][0] else []
    if not hits:
        return {"success": False, "tool": "get_section",
                "message": f"No section matches '{q}'. Use get_section('contents') for the list."}

    from app.summaries import load_summaries
    summaries = {s["id"]: s["summary"] for s in load_summaries()}
    if ch and not m:                                        # whole chapter: list its sub-sections
        subs = []
        for s in hits:
            if s["id"] not in [x["id"] for x in subs]:
                subs.append(s)
        first = subs[0]
        results = [_passage(first_n + i, s["source"], s["pages"][0],
                            summaries.get(s["id"]) or s["text"].strip()[:600], s["section"],
                            found_by=["get_section"])
                   for i, s in enumerate(subs[:8])]
        return {"success": True, "tool": "get_section", "section": f"Chapter {first['chapter']}",
                "pages": [subs[0]["pages"][0], subs[-1]["pages"][-1]],
                "subsections": [s["section"] for s in subs], "results": results}
    s = hits[0]
    body = summaries.get(s["id"])
    text = s["text"].strip()
    results = [_passage(first_n, s["source"], s["pages"][0], text[:max_chars], s["section"],
                        found_by=["get_section"])]
    if body:
        results.insert(0, _passage(first_n, s["source"], s["pages"][0], body, f"Summary of {s['section']}",
                                   found_by=["summary"]))
        results[1]["n"] = first_n + 1
    return {"success": True, "tool": "get_section", "section": s["section"],
            "pages": [s["pages"][0], s["pages"][-1]], "results": results}


# ============================================================
# 5. cash_flow_analysis (own tool #3) - capital budgeting in one call
# ============================================================
def _to_floats(values) -> list[float]:
    if isinstance(values, str):
        values = [v for v in re.split(r"[;,\s]+", values.strip("[] ")) if v]
    return [float(str(v).replace("$", "").replace(",", "")) for v in values]


def cash_flow_analysis(cash_flows, rate_percent=None) -> dict:
    """
    cash_flows[0] = time 0 (the investment, negative); then one cash flow per year.
    Returns NPV, IRR, payback, discounted payback, profitability index and the decision rules
    of chapter 16 (Capital Budgeting Decisions).
    """
    from app import tools as fin
    try:
        cfs = _to_floats(cash_flows)
    except (TypeError, ValueError):
        return {"success": False, "tool": "cash_flow_analysis",
                "message": "cash_flows must be a list of numbers, e.g. [-1000, 300, 400, 500]."}
    if len(cfs) < 2:
        return {"success": False, "tool": "cash_flow_analysis",
                "message": "Give at least 2 cash flows: the investment at time 0 and one later cash flow."}
    if cfs[0] > 0 and all(c <= 0 for c in cfs[1:]):
        cfs = [-c for c in cfs]
    if cfs[0] > 0:
        cfs[0] = -cfs[0]                                      # the investment is an outflow
    out = {"success": True, "tool": "cash_flow_analysis", "cash_flows": cfs,
           "irr_percent": fin.irr(cfs).get("irr_percent"),
           "payback_years": fin.payback_period(-cfs[0], cfs[1:]).get("payback_years")}
    if rate_percent not in (None, ""):
        try:
            r = float(str(rate_percent).replace("%", ""))
        except ValueError:
            return {"success": False, "tool": "cash_flow_analysis", "message": "rate_percent must be a number like 10."}
        pv = [c / (1 + r / 100) ** t for t, c in enumerate(cfs)]
        npv = sum(pv)
        cum, dpb = pv[0], None
        for t in range(1, len(pv)):
            if cum + pv[t] >= 0 and dpb is None:
                dpb = round(t - 1 + (-cum) / pv[t], 3)
            cum += pv[t]
        out.update(rate_percent=r, npv=round(npv, 2), discounted_payback_years=dpb,
                   profitability_index=round(sum(pv[1:]) / -cfs[0], 4) if cfs[0] else None,
                   decision=("accept: NPV > 0" if npv > 0 else "reject: NPV <= 0")
                   + (f"; IRR {out['irr_percent']}% {'>' if (out['irr_percent'] or -1e9) > r else '<='} "
                      f"required return {r}%" if out["irr_percent"] is not None else ""))
    else:
        out["note"] = "Give rate_percent to also get NPV, discounted payback and the profitability index."
    return out







def amortization_schedule(principal, annual_rate_pct, years):
    if principal <= 0 or years <= 0:
        return {"success": False, "tool": "amortization_schedule", "message": "principal and years must be positive."}
    r = annual_rate_pct / 100.0 / 12
    n = years * 12
    mp = (principal * r * (1 + r) ** n / ((1 + r) ** n - 1)) if r else principal / n
    total_paid = mp * n
    total_interest = total_paid - principal
    rows, bal = [], principal
    for month in range(1, min(4, n + 1)):
        ip = bal * r; pp = mp - ip; bal -= pp
        rows.append("Month {}: payment ${:.2f} (principal ${:.2f} / interest ${:.2f}) -> balance ${:.2f}".format(
            month, mp, pp, ip, max(0.0, bal)))
    result = "Loan: ${:,.2f} at {}% APR for {} years | Monthly: ${:,.2f} | Total: ${:,.2f} | Interest: ${:,.2f}".format(
        principal, annual_rate_pct, years, mp, total_paid, total_interest
    ) + "\nFirst months:\n" + "\n".join(rows)
    return {"success": True, "tool": "amortization_schedule", "monthly_payment": round(mp, 2),
            "total_paid": round(total_paid, 2), "total_interest": round(total_interest, 2), "result": result}


def goal_calculator(goal_amount, current_savings, monthly_contribution, annual_rate_pct):
    if goal_amount <= 0 or monthly_contribution <= 0:
        return {"success": False, "tool": "goal_calculator", "message": "goal_amount and monthly_contribution must be positive."}
    if current_savings >= goal_amount:
        return {"success": True, "tool": "goal_calculator", "months": 0,
                "result": "Goal already reached! Current savings: ${:,.2f}".format(current_savings)}
    r = annual_rate_pct / 100.0 / 12
    bal, months = current_savings, 0
    while bal < goal_amount and months < 1200:
        bal = bal * (1 + r) + monthly_contribution
        months += 1
    if months >= 1200:
        return {"success": False, "tool": "goal_calculator",
                "message": "Goal not reachable in 100 years."}
    y, mo = divmod(months, 12)
    result = "Goal: ${:,.2f} | Current: ${:,.2f} | ${:,.2f}/month at {}% | Time: {} yr {} mo ({} months) | Projected: ${:,.2f}".format(
        goal_amount, current_savings, monthly_contribution, annual_rate_pct, y, mo, months, bal)
    return {"success": True, "tool": "goal_calculator", "months": months, "years": y,
            "rem_months": mo, "projected_balance": round(bal, 2), "result": result}


def budget_analyzer(monthly_income, expenses):
    if monthly_income <= 0:
        return {"success": False, "tool": "budget_analyzer", "message": "monthly_income must be positive."}
    if not expenses:
        return {"success": False, "tool": "budget_analyzer", "message": "expenses dict is empty."}
    total = sum(expenses.values())
    surplus = monthly_income - total
    rate = surplus / monthly_income * 100
    lines = ["Monthly income: ${:,.2f}".format(monthly_income), "Expenses:"]
    for cat, amt in sorted(expenses.items(), key=lambda x: -x[1]):
        lines.append("  {}: ${:,.2f} ({:.1f}%)".format(cat, amt, amt / monthly_income * 100))
    lines += ["-" * 40,
              "  Total: ${:,.2f} ({:.1f}%)".format(total, total / monthly_income * 100),
              "  {}: ${:,.2f} ({:.1f}%)".format("Surplus" if surplus >= 0 else "Deficit", abs(surplus), abs(rate)),
              ""]
    lines.append("Budget deficit." if surplus < 0 else
                 "Savings < 10% - cut spending." if rate < 10 else
                 "Savings 10-20% - on track." if rate < 20 else
                 "Savings >= 20% - excellent.")
    return {"success": True, "tool": "budget_analyzer", "surplus": round(surplus, 2),
            "savings_rate_pct": round(rate, 2), "result": "\n".join(lines)}



# ============================================================
# Registry (used by the agent prompt, the UI and the MCP server)
# ============================================================
TOOLS = {
    "search_corpus": search_corpus,
    "calculator": calculator,
    "lookup_term": lookup_term,
    "get_section": get_section,
    "cash_flow_analysis": cash_flow_analysis,
    "amortization_schedule": amortization_schedule,
    "goal_calculator": goal_calculator,
    "budget_analyzer": budget_analyzer,
}
OWN_TOOLS = ["lookup_term", "get_section", "cash_flow_analysis",
             "amortization_schedule", "goal_calculator", "budget_analyzer"]
