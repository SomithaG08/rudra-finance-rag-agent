"""
app.py - Gradio web interface of the Personal Finance RAG assistant.

Run from the repository root:
    python app/app.py            then open http://127.0.0.1:7860

Tabs
  1. Chat            (RAG, Day 1-2) multi-turn chat, streamed answers, inline clickable citations [n],
                     a Sources panel with every retrieved passage, settings, feedback,
                     export (.md / .pdf)
  2. Agent (tools)   (Agentic AI) the model picks tools step by step: search, calculator,
                     finance functions, exact page quote, message with human approval
  3. Quiz            multiple-choice questions per section
  4. Chunk browser   look inside the index: search chunks by keyword / source / page
  5. Compare         run the same question with two retrieval settings side by side
  6. Analytics       usage, cost, feedback, most-cited pages, latest evaluation scorecard
  7. How it works    architecture and help
"""

import html
import inspect
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# Make "import app.xxx" work when this file is started as a script (python app/app.py)
ROOT = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != ROOT / "app"]
sys.path.insert(0, str(ROOT))

import gradio as gr

from app import features, rag
from app.config import (RERANK_THRESHOLD, USE_HYDE, USE_QUERY_REWRITE, USE_RERANK, INJECTION_TEST,
                        BUILD_INFO_FILE, CONFIDENCE_THRESHOLD, DEFAULT_SEARCH_MODE, DEFAULT_TOP_K,
                        EVAL_RESULTS_FILE, EXPORT_DIR, PROMPT_VERSION, PROMPTS_DIR, SOURCES_DIR,
                        has_api_key, source_title)
from app.retriever import Retriever, get_retriever
from app.text_utils import highlight, split_sentences
from app.tracking import (cost_summary, log_feedback, log_query, most_cited_pages, read_feedback,
                          read_queries)

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
FILE_ROUTE = "/gradio_api/file=" if GRADIO_MAJOR >= 5 else "/file="


def _kw(component, **kwargs):
    """Keep only the keyword arguments this Gradio version understands (4.x, 5.x and 6.x)."""
    params = inspect.signature(component.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()) and component is not gr.Blocks:
        return component(**kwargs)
    return component(**{k: v for k, v in kwargs.items() if k in params})


# ---------------------------------------------------------------------------
# Citations -> links to the PDF page
# ---------------------------------------------------------------------------
def _find_source_file(source: str) -> Path | None:
    """Find data/sources/<source> (case-insensitive)."""
    if not SOURCES_DIR.exists():
        return None
    exact = SOURCES_DIR / source
    if exact.exists():
        return exact
    for f in SOURCES_DIR.iterdir():
        if f.name.lower() == source.lower():
            return f
    return None


def pdf_link(source: str, page: int) -> str | None:
    """URL that opens the PDF in the browser directly at the cited page."""
    path = _find_source_file(source)
    if path is None or path.suffix.lower() != ".pdf":
        return None
    return f"{FILE_ROUTE}{quote(path.resolve().as_posix(), safe='/:')}#page={page}"


def citation_label(c: dict) -> str:
    """'Principles of Finance (OpenStax, 2022) - 7.4 Applications of TVM in Finance - p. 229'"""
    parts = [c.get("source_title") or source_title(c["source"])]
    if c.get("section"):
        parts.append(c["section"])
    page = f"p. {c['page']}"
    if c.get("printed_page"):
        page += f" (printed {c['printed_page']})"
    parts.append(page)
    return " - ".join(parts)


def render_answer(result: dict) -> str:
    """Answer text for the chat bubble: [n] become links, plus a 'Sources' list below."""
    answer = result["answer"]
    by_n = {c["n"]: c for c in result["citations"]}

    def link(match):
        nums = [int(x) for x in re.split(r"[,;]\s*", match.group(1))]
        out = []
        for n in nums:
            c = by_n.get(n)
            url = pdf_link(c["source"], c["page"]) if c else None
            if c is None:                          # citation check: cited but never retrieved
                out.append(f"**[{n}] ⚠️ unverified source**")
            else:
                out.append(f"[[{n}]]({url} \"{citation_label(c)}\")" if url else f"**[{n}]**")
        return "".join(out)

    body = re.sub(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]", link, answer)

    extras = []
    if result.get("calculations"):
        extras.append("**Calculations (exact, done by Python tools):**")
        for c in result["calculations"]:
            extras.append(f"- `{c['name']}({c['arguments']})` -> `{json.dumps(c['result'])}`")
    ver = result.get("verification") or {}
    if ver.get("checked"):
        if ver.get("unsupported"):
            extras.append("**Citation check: some sentences are not supported by their passage:**")
            for u in ver["unsupported"]:
                extras.append(f"- *{u.get('sentence', '')}* - {u.get('reason', '')}")
        else:
            extras.append("**Citation check:** every cited sentence is supported by its passage.")
    conf = result.get("confidence") or {}
    if conf.get("label") and conf["label"] != "n/a":
        extras.append(f"**Confidence: {conf['label'].upper()}** ({', '.join(conf['reasons'])})")
    if result.get("cached"):
        extras.append(f"*Answered from the cache ({result['cached']} match) - no API cost.*")
    if extras:
        body += "\n\n" + "\n\n".join(extras)

    if result["refused"]:
        why = ""
        if result.get("low_confidence") and result.get("reranked") and result.get("top_rerank") is not None:
            why = (f" (best reranker score {result['top_rerank']:.2f} is below the threshold "
                   f"{result['settings'].get('rerank_threshold', RERANK_THRESHOLD):.2f})")
        elif result.get("low_confidence") and result.get("top_score") is not None:
            why = (f" (best match similarity {result['top_score']:.2f} is below the "
                   f"confidence threshold {result['settings']['threshold']:.2f})")
        return body + f"\n\n*No sources shown: this question is outside the knowledge base{why}.*"

    cited = [by_n[n] for n in result["cited_order"] if n in by_n]
    if not cited:                                 # model forgot to cite -> show what it read
        cited = result["citations"][:3]
        header = "**Sources (retrieved passages - the answer did not cite them explicitly):**"
    else:
        pages = {(c["source"], c["page"]) for c in cited}
        docs = {c["source"] for c in cited}
        header = (f"**Sources - {len(cited)} passage(s) from {len(pages)} page(s) in "
                  f"{len(docs)} document(s):**")
    lines = []
    for c in cited:
        url = pdf_link(c["source"], c["page"])
        label = citation_label(c)
        score = f" - similarity {c['score']:.2f}" if c.get("score") is not None else ""
        lines.append(f"- **[{c['n']}]** " + (f"[{label}]({url})" if url else label) + score)
    return body + "\n\n" + header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Sources panel (right column)
# ---------------------------------------------------------------------------
PANEL_CSS = """
.src-head{font-weight:600;margin:4px 0 10px}
.src-wrap{max-height:900px;overflow-y:auto;padding-right:4px}
.src-card{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px;margin-bottom:10px}
.src-card.cited{border-left:5px solid #2f7d32}
.src-card.notcited{border-left:5px solid rgba(128,128,128,.5);opacity:.85}
.src-top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.92em}
.src-n{font-weight:700;background:#1f5fbf;color:#fff;border-radius:6px;padding:1px 7px}
.src-badge{font-size:.78em;border-radius:10px;padding:1px 8px;background:rgba(47,125,50,.15)}
.src-badge.grey{background:rgba(128,128,128,.18)}
.src-title{font-weight:600}
.src-meta{font-size:.82em;opacity:.8;margin:3px 0}
.src-bar{height:6px;border-radius:3px;background:rgba(128,128,128,.25);margin:4px 0 6px}
.src-bar>div{height:6px;border-radius:3px;background:#1f5fbf}
.src-text{font-size:.86em;line-height:1.4}
.src-text mark{background:rgba(255,200,0,.45);color:inherit;padding:0 1px}
.src-card details summary{cursor:pointer;font-size:.82em;opacity:.8}
.src-proof{border-left:4px solid #2f7d32;background:rgba(47,125,50,.10);padding:6px 9px;margin:6px 0;
  font-size:.86em;border-radius:5px;line-height:1.4}
.src-text mark.proof{background:rgba(47,125,50,.35)}
.cmp-table{width:100%;border-collapse:collapse;font-size:.85em}
.cmp-table td,.cmp-table th{border-bottom:1px solid rgba(128,128,128,.3);padding:4px;vertical-align:top}
.cmp-same{background:rgba(47,125,50,.12)}
"""


def sources_html(result: dict | None) -> str:
    if not result:
        return "<div class='src-head'>Sources will appear here after your first question.</div>"
    if result["refused"]:
        return ("<div class='src-head'>No sources: the assistant refused because the question "
                "is not covered by the knowledge base. Closest passages, for transparency:</div>"
                "<div class='src-wrap'>" + _cards(result, only_cited=False) + "</div>")
    cited = sum(c["cited"] for c in result["citations"])
    pages = len({(c["source"], c["page"]) for c in result["citations"] if c["cited"]})
    head = (f"<div class='src-head'>{cited} cited passage(s) on {pages} page(s) - "
            f"{len(result['citations'])} passages retrieved "
            f"({html.escape(result['settings']['mode'])} search)</div>")
    return head + "<div class='src-wrap'>" + _cards(result, only_cited=False) + "</div>"


def _cards(result: dict, only_cited: bool) -> str:
    q = result.get("standalone_question") or result["question"]
    cards = []
    ordered = sorted(result["citations"], key=lambda c: (not c["cited"], c["n"]))
    for c in ordered:
        if only_cited and not c["cited"]:
            continue
        url = pdf_link(c["source"], c["page"])
        score = c.get("score")
        bar = f"<div class='src-bar'><div style='width:{max(0, min(1, score or 0)) * 100:.0f}%'></div></div>"
        chips = " ".join(f"<span class='src-badge grey'>{html.escape(f)}</span>" for f in c["found_by"])
        badge = ("<span class='src-badge'>cited in answer</span>" if c["cited"]
                 else "<span class='src-badge grey'>retrieved, not cited</span>")
        open_pdf = f" - <a href='{url}' target='_blank'>open PDF at this page</a>" if url else ""
        meta = []
        if score is not None:
            meta.append(f"similarity {score:.3f}")
        if c.get("bm25"):
            meta.append(f"BM25 {c['bm25']:.1f}")
        if c.get("rrf"):
            meta.append(f"RRF {c['rrf']:.4f}")
        if c.get("rerank") is not None:
            meta.insert(0, f"<b>rerank {c['rerank']:.2f}</b>")
        if c.get("injection"):
            badge += " <span class='src-badge' style='background:rgba(200,40,40,.2)'>INJECTION TEST CHUNK</span>"
        ev_all = result.get("evidence") or {}
        proofs = ev_all.get(c["n"]) or ev_all.get(str(c["n"])) or []
        proof_html = "".join(
            f"<div class='src-proof'><b>Proof for:</b> <i>{html.escape(p['claim'][:220])}</i><br>"
            f"&ldquo;{html.escape(p['evidence'])}&rdquo;</div>" for p in proofs)
        full = html.escape(c["text"])
        for p in proofs:                                  # highlight the proof sentence in the full text
            full = full.replace(html.escape(p["evidence"]), f"<mark class='proof'>{html.escape(p['evidence'])}</mark>", 1)
        cards.append(f"""
<div class='src-card {"cited" if c["cited"] else "notcited"}'>
  <div class='src-top'><span class='src-n'>{c['n']}</span>{badge} {chips}</div>
  <div class='src-title'>{html.escape(citation_label(c))}</div>
  <div class='src-meta'>{' - '.join(meta)}{open_pdf}</div>
  {bar if score is not None else ''}
  {proof_html}
  <div class='src-text'>{highlight(c['text'], q, 380)}</div>
  <details><summary>show full passage</summary><div class='src-text'>{full}</div></details>
</div>""")
    return "\n".join(cards)


# ---------------------------------------------------------------------------
# Source viewer with highlights (RAG day 2 extension)
# ---------------------------------------------------------------------------
_PDF_CACHE: dict = {}


def _open_pdf(path):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    key = str(path)
    if key not in _PDF_CACHE:
        _PDF_CACHE[key] = pymupdf.open(key)
    return _PDF_CACHE[key], pymupdf


def _search_pieces(text: str) -> list[str]:
    """Short pieces of a sentence: long exact matches fail on PDF line breaks / hyphens."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    pieces = []
    for size in (12, 8, 5):
        for start in range(0, max(1, len(words) - size + 1), size):
            piece = " ".join(words[start:start + size]).strip(" .,;:")
            if len(piece) > 15:
                pieces.append(piece)
        if pieces:
            break
    return pieces[:6]


def render_page(source: str, page: int, sentences: list[str], zoom: float = 1.4):
    """PIL image of one PDF page with the given sentences highlighted (PyMuPDF counts from 0)."""
    from PIL import Image
    path = _find_source_file(source)
    if path is None or path.suffix.lower() != ".pdf":
        return None
    doc, pymupdf = _open_pdf(path)
    if not 1 <= page <= doc.page_count:
        return None
    pg = doc.load_page(page - 1)
    annots = []
    for sent in sentences:
        for piece in _search_pieces(sent):
            for quad in pg.search_for(piece, quads=True):
                annots.append(pg.add_highlight_annot(quad))
    pix = pg.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    for a in annots:                               # keep the cached document clean
        pg.delete_annot(a)
    return img


def source_pages(results, max_pages: int = 4):
    """Gallery items for the last answer: each cited page, with its proof sentence highlighted."""
    if not results:
        return []
    r = results[-1]
    if r.get("refused"):
        return []
    ev = r.get("evidence") or {}
    cits = [c for c in r.get("citations", []) if c.get("cited")] or r.get("citations", [])[:2]
    items, seen = [], set()
    for c in cits:
        key = (c["source"], c["page"])
        if key in seen or c.get("injection"):
            continue
        seen.add(key)
        proofs = [p["evidence"] for p in (ev.get(c["n"]) or ev.get(str(c["n"])) or [])]
        sentences = proofs or split_sentences(c["text"])[:2]
        try:
            img = render_page(c["source"], int(c["page"]), sentences)
        except Exception:                          # noqa: BLE001 - a viewer must never break the chat
            img = None
        if img is not None:
            items.append((img, f"[{c['n']}] p. {c['page']} - {c.get('section') or ''}"))
        if len(items) >= max_pages:
            break
    return items


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------
def msg_text(message: dict) -> str:
    """Text of a chat message (Gradio 6 stores content as a list of parts)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content or "")


def simple_history(history) -> list[dict]:
    return [{"role": m["role"], "content": msg_text(m)} for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")]


def status_line(result: dict) -> str:
    t = result["timings"]
    parts = [f"retrieval {t['retrieval']}s", f"generation {t['generation']}s", f"total {t['total']}s"]
    if result.get("top_score") is not None:
        parts.append(f"best similarity {result['top_score']:.2f}")
    parts.append(f"cost ${result['cost_usd']:.5f}")
    extra = ""
    if result["standalone_question"] != result["question"]:
        extra += f"  \nFollow-up understood as: *{result['standalone_question']}*"
    if result.get("search_query") and result["search_query"] != result["standalone_question"]:
        extra += f"  \nSearched the English book with: *{result['search_query']}*"
    if result.get("expansions"):
        extra += "  \nAlso searched: " + "; ".join(f"*{e}*" for e in result["expansions"])
    return "Done - " + " - ".join(parts) + extra


NO_FU = (gr.update(visible=False),) * 3


def _followup_updates(result: dict):
    qs = (result.get("followups") or [])[:3]
    return tuple(gr.update(value=qs[i], visible=True) if i < len(qs) else gr.update(visible=False)
                 for i in range(3))


# ---------------------------------------------------------------------------
# Steps & tools dropdown shown in the chat for every RAG answer
# ---------------------------------------------------------------------------
TOOL_CALL_NAMES = {"rule_of_72": "Rule of 72", "future_value": "Future value", "present_value": "Present value",
                   "effective_annual_rate": "Effective annual rate", "npv": "NPV", "irr": "IRR",
                   "payback_period": "Payback period", "loan_payment": "Loan payment",
                   "financial_ratio": "Financial ratio", "calculator": "Calculator"}


def rag_steps(result: dict) -> list[tuple[str, str, float | None]]:
    """(title, details markdown, seconds) for every step the pipeline ran for this answer."""
    steps = []
    t = result.get("timings", {})
    if result.get("standalone_question") and result["standalone_question"] != result["question"]:
        steps.append(("💬 Follow-up rewritten (conversation memory)",
                      f"Understood as: *{result['standalone_question']}*", None))
    if result.get("cached"):
        steps.append(("⚡ Answer cache", f"Same question found in the cache ({result['cached']} match) - "
                                         "answered instantly, no API cost.", t.get("total")))
        return steps
    if result.get("search_query") and result["search_query"] != result.get("standalone_question"):
        steps.append(("🌐 Translated for search", f"Searched the English book with: *{result['search_query']}*",
                      None))
    if result.get("expansions"):
        steps.append(("🔀 Query expansion", "Also searched:\n" + "\n".join(f"- *{e}*" for e in result["expansions"]),
                      None))
    cits = result.get("citations", [])
    mode = result.get("settings", {}).get("mode", "hybrid")
    mode_name = {"hybrid": "Hybrid search (BM25 + vectors)", "dense": "Vector search (FAISS)",
                 "bm25": "Keyword search (BM25)"}.get(mode, mode)
    pages = sorted({c["page"] for c in cits})
    lines = [f"**{len(cits)} passages** from **{len(pages)} pages** "
             f"(top-k {result.get('settings', {}).get('top_k', len(cits))}, max 2 per page)"]
    for c in cits[:10]:
        sim = f" - sim {c['score']:.2f}" if c.get("score") is not None else ""
        rr = f" - rerank **{c['rerank']:.2f}**" if c.get("rerank") is not None else ""
        warn = " - ⚠️ INJECTION TEST CHUNK" if c.get("injection") else ""
        lines.append(f"- [{c['n']}] p. {c['page']} - {c.get('section') or ''}{sim}{rr} - via {'+'.join(c['found_by'])}{warn}")
    if result.get("hyde_text"):
        steps.append(("🧪 HyDE (hypothetical passage)", f"Vector search used this text instead of the question:\n\n> {result['hyde_text']}", None))
    steps.append((f"🔎 {mode_name}" + (" + cross-encoder rerank" if result.get("reranked") else ""),
                  "\n".join(lines), t.get("retrieval")))
    if result.get("summaries_used"):
        secs = [c for c in cits if "summary" in (c.get("found_by") or [])]
        steps.append(("📚 Section summaries (broad question)",
                      "Broad question: added the summary of\n" + "\n".join(
                          f"- [{c['n']}] {c.get('section', '')}" for c in secs), None))
    if result.get("profile_used"):
        prof = result.get("settings", {}).get("profile") or {}
        used = ", ".join(f"{features.PROFILE_LABELS[k].lower()} {v}" for k, v in prof.items()
                         if k in features.PROFILE_LABELS and v not in (None, "", 0))
        steps.append(("👤 Personal profile", ("Personal question: the book's method is applied to your numbers ("
                      if result.get("personal_question") else "Profile active, but the question is general: ")
                      + used + (")" if result.get("personal_question") else ""), None))
    top, thr = result.get("top_score"), result.get("settings", {}).get("threshold", CONFIDENCE_THRESHOLD)
    if result.get("reranked") and result.get("top_rerank") is not None:
        top, thr = result["top_rerank"], result.get("settings", {}).get("rerank_threshold", RERANK_THRESHOLD)
        ok = not result.get("low_confidence")
        steps.append(("🛡️ Confidence check (reranker score)", f"Best cross-encoder score **{top:.2f}** "
                      + (f">= threshold {thr:.2f} -> answer" if ok else
                         f"< threshold {thr:.2f} -> **refuse** (no LLM call)"), None))
        top = None
    if top is not None:
        ok = not result.get("low_confidence")
        steps.append(("🛡️ Confidence check", f"Best similarity **{top:.2f}** "
                      + (f">= threshold {thr:.2f} -> answer" if ok else
                         f"< threshold {thr:.2f} -> **refuse** (no LLM call)"), None))
    for c in result.get("calculations") or []:
        try:
            args = ", ".join(f"{k}={v}" for k, v in json.loads(c["arguments"] or "{}").items())
        except Exception:                        # noqa: BLE001
            args = c["arguments"]
        res = c["result"]
        res_md = "\n".join(f"- {k.replace('_', ' ')}: **{v}**" for k, v in res.items()) \
            if isinstance(res, dict) else f"`{res}`"
        steps.append((f"🧮 Tool: {TOOL_CALL_NAMES.get(c['name'], c['name'])}",
                      f"`{c['name']}({args})`\n\n{res_md}", None))
    if not result.get("low_confidence"):
        if result.get("offline"):
            steps.append(("📝 Extractive answer (offline mode)", "Best sentences of the passages, no LLM.",
                          t.get("generation")))
        else:
            s = result.get("settings", {})
            cited = result.get("cited_order") or []
            steps.append(("✍️ Answer written by gpt-4o-mini",
                          f"Prompt **{s.get('prompt_version', '')}** - style **{s.get('style', 'Standard')}** - "
                          f"language **{result.get('language', 'en')}**\n\n"
                          + (f"Cites passages {', '.join(f'[{n}]' for n in cited)}" if cited else
                             ("Refused: the passages do not answer the question." if result.get("refused")
                              else "No explicit citation.")), t.get("generation")))
    ev = result.get("evidence") or {}
    if ev:
        n_claims = sum(len(v) for v in ev.values())
        steps.append(("🧾 Proof sentences found", f"{n_claims} claim(s) matched to the exact source sentence in "
                      f"{len(ev)} passage(s) - shown in green in the Sources panel.", None))
    ver = result.get("verification") or {}
    if ver.get("checked"):
        bad = ver.get("unsupported") or []
        steps.append(("✅ Citation check", "Every cited sentence is supported by its passage." if not bad else
                      f"{len(bad)} sentence(s) not supported by their passage.", None))
    if result.get("followups"):
        steps.append(("💡 Follow-up suggestions", "\n".join(f"- {q}" for q in result["followups"]), None))
    return steps


def steps_messages(result: dict, uid: str) -> list[dict]:
    """One parent dropdown 'Steps & tools used' with one nested dropdown per step."""
    steps = rag_steps(result)
    tools = [title.split(" ", 1)[1] for title, _, _ in steps]
    parent = {"role": "assistant",
              "content": "**Tools used:** " + " → ".join(tools),
              "metadata": {"title": f"🛠️ Steps & tools used ({len(steps)} steps)", "id": uid,
                           "status": "done", "duration": result.get("timings", {}).get("total")}}
    out = [parent]
    for i, (title, body, secs) in enumerate(steps, start=1):
        meta = {"title": f"Step {i} · {title}", "parent_id": uid, "id": f"{uid}-{i}", "status": "done"}
        if secs:
            meta["duration"] = secs
        out.append({"role": "assistant", "content": body, "metadata": meta})
    return out


def respond(message, history, results, top_k, mode, threshold, expand, rerank, source_filter,
            prompt_version, style="Standard", language="Auto", tools=True, verify=False,
            followups=True, use_cache=True, p_on=False, p_income=None, p_expenses=None, p_savings=None,
            p_msave=None, p_goal=None, p_years=None, p_return=None, p_risk=None, p_currency="EUR",
            rerank_thr=RERANK_THRESHOLD, rewrite=USE_QUERY_REWRITE, hyde=USE_HYDE, injection=INJECTION_TEST):
    """Generator: yields (chatbot, textbox, sources panel, status, results, 3 follow-up buttons).
    The steps of the pipeline are shown live in a dropdown above the answer."""
    history = agent_history(history)                    # keeps the dropdowns of earlier answers
    results = list(results or [])
    if not message or not message.strip():
        yield (history, "", gr.update(), "Please type a question.", results) + NO_FU
        return
    message = message.strip()
    settings = {"top_k": int(top_k), "mode": mode, "threshold": float(threshold),
                "expand": bool(expand), "rerank": bool(rerank),
                "source_filter": source_filter, "prompt_version": prompt_version,
                "style": style, "language": language, "tools": bool(tools), "verify": bool(verify),
                "followups": bool(followups), "use_cache": bool(use_cache),
                "rerank_threshold": float(rerank_thr), "rewrite": bool(rewrite), "hyde": bool(hyde),
                "injection": bool(injection)}
    if p_on:
        settings["profile"] = {"enabled": True, "currency": p_currency or "EUR",
                               "monthly_income": p_income, "monthly_expenses": p_expenses,
                               "savings": p_savings, "monthly_saving": p_msave, "goal_amount": p_goal,
                               "goal_years": p_years, "expected_return": p_return, "risk": p_risk}
    past = [dict(m) for m in history if "metadata" not in m]   # the model only sees real Q/A
    uid = f"rag-{len(results)}-{datetime.now():%H%M%S%f}"
    live = {"role": "assistant", "content": "",
            "metadata": {"title": "🛠️ Working...", "id": uid, "status": "pending"}}
    answer_msg = {"role": "assistant", "content": "_Searching the knowledge base..._"}
    history = history + [{"role": "user", "content": message}, live, answer_msg]
    done_lines: list[str] = []
    yield (history, "", gr.update(), "Working...", results) + NO_FU

    answer = ""
    try:
        for event in rag.run(message, past, settings):
            if event["type"] == "status":
                done_lines.append(f"- {event['text']}")
                live["metadata"]["title"] = f"🛠️ {event['text']}"
                live["content"] = "\n".join(done_lines)
                yield (history, "", gr.update(), event["text"], results) + NO_FU
            elif event["type"] == "token":
                if not answer:
                    live["metadata"]["title"] = "🛠️ Writing the answer..."
                answer += event["text"]
                answer_msg["content"] = answer
                yield (history, "", gr.update(), "Writing the answer...", results) + NO_FU
            elif event["type"] == "done":
                result = event["result"]
                i = history.index(live)
                history[i:i + 1] = steps_messages(result, uid)   # final, detailed steps
                answer_msg["content"] = render_answer(result)
                results.append(result)
                log_query(result)
                yield (history, "", sources_html(result), status_line(result), results) \
                    + _followup_updates(result)
    except Exception as exc:          # show errors in the chat instead of crashing the UI
        from app.llm import friendly_error
        msg = friendly_error(exc)
        live["metadata"].update(status="done", title="🛠️ Stopped (error)")
        answer_msg["content"] = f"**Error:** {html.escape(msg)}"
        yield (history, "", gr.update(), f"Error: {msg}", results) + NO_FU


def saved_chat_answer(message, history):
    """gr.ChatInterface callback for the 'Saved conversations' tab (answer + sources, no streaming)."""
    past = [{"role": m["role"], "content": msg_text(m)} for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and not m.get("metadata")]
    try:
        result = rag.answer(message, past)
    except Exception as exc:                      # noqa: BLE001
        from app.llm import friendly_error
        return f"**Error:** {friendly_error(exc)}"
    log_query(result)
    return render_answer(result)


def feedback(helpful: bool, comment: str, results):
    if not results:
        return "Ask a question first."
    last = results[-1]
    log_feedback(last["question"], last["answer"], helpful, comment or "")
    return "Thanks! Feedback saved." + (" (marked helpful)" if helpful else " (marked not helpful)")


def on_like(data: gr.LikeData, results, history):
    """Thumbs icons inside the chat bubbles (Gradio feedback_options)."""
    if not results:
        return
    idx = data.index[0] if isinstance(data.index, (list, tuple)) else data.index
    users_before = sum(1 for m in (history or [])[:int(idx) + 1] if isinstance(m, dict) and m.get("role") == "user")
    turn = max(0, min(len(results) - 1, users_before - 1))
    r = results[turn]
    log_feedback(r["question"], r["answer"], bool(data.liked))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _conversation_markdown(results) -> str:
    lines = [f"# Personal Finance RAG - conversation export",
             f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for i, r in enumerate(results, start=1):
        lines += [f"## Q{i}. {r['question']}"]
        if r["standalone_question"] != r["question"]:
            lines.append(f"_Understood as: {r['standalone_question']}_")
        lines += ["", r["answer"], ""]
        cited = [c for c in r["citations"] if c["cited"]]
        if cited:
            lines.append("**Sources**")
            for c in cited:
                lines.append(f"- [{c['n']}] {citation_label(c)}")
        lines.append("")
    return "\n".join(lines)


def export_md(results):
    if not results:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"conversation_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(_conversation_markdown(results), encoding="utf-8")
    return str(path)


def export_pdf(results):
    """PDF export with PyMuPDF (already used by scripts/extract.py)."""
    if not results:
        return None
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"conversation_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    parts = ["<h2>Personal Finance RAG - conversation</h2>"]
    for i, r in enumerate(results, start=1):
        parts.append(f"<h3>Q{i}. {html.escape(r['question'])}</h3>")
        parts.append("<p>" + html.escape(r["answer"]).replace("\n", "<br>") + "</p>")
        cited = [c for c in r["citations"] if c["cited"]]
        if cited:
            parts.append("<p><b>Sources</b><br>" + "<br>".join(
                f"[{c['n']}] {html.escape(citation_label(c))}" for c in cited) + "</p>")
    story = fitz.Story(html="".join(parts), user_css="body{font-family:sans-serif;font-size:10pt}")
    writer = fitz.DocumentWriter(str(path))
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (40, 40, -40, -40)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return str(path)


def clear_chat():
    return [], [], sources_html(None), "Conversation cleared.", None


# ---------------------------------------------------------------------------
# Quiz (study mode)
# ---------------------------------------------------------------------------
MAX_QUIZ = 10


def make_quiz(section, topic, n, language):
    """Returns quiz state + updates for the MAX_QUIZ radio slots + message."""
    hidden = [gr.update(visible=False, value=None) for _ in range(MAX_QUIZ)]
    if not has_api_key():
        return [], *hidden, "The quiz needs the OpenAI key (it is written by gpt-4o-mini)."
    topic = (topic or "").strip()
    if not topic and not section:
        return [], *hidden, "Choose a section or type a topic."
    passages = features.quiz_passages(get_retriever(), None if topic else section, topic or None)
    if not passages:
        return [], *hidden, "No passages found for this section/topic."
    lang = "fr" if language == "Francais" else "en"
    quiz = features.generate_quiz(passages, int(n), lang)
    if not quiz:
        return [], *hidden, "The model did not return a valid quiz, please try again."
    updates = []
    for i in range(MAX_QUIZ):
        if i < len(quiz):
            q = quiz[i]
            updates.append(gr.update(visible=True, value=None, label=f"Q{i + 1}. {q['question']}",
                                     choices=q["options"]))
        else:
            updates.append(gr.update(visible=False, value=None))
    where = f"topic '{topic}'" if topic else f"section {section}"
    return quiz, *updates, f"{len(quiz)} questions generated from {where}. Answer and press Check."


def check_quiz(quiz, *answers):
    if not quiz:
        return "Generate a quiz first."
    lines, score = [], 0
    for i, q in enumerate(quiz):
        chosen = answers[i]
        correct = q["options"][q["answer_index"]]
        ok = chosen == correct
        score += ok
        url = pdf_link(q["source"], q["page"])
        ref = f"p. {q['page']}" + (f" ({q['section']})" if q.get("section") else "")
        ref = f"[{ref}]({url})" if url else ref
        lines.append(f"**Q{i + 1}** {'Correct' if ok else 'Wrong'} - answer: *{correct}*. "
                     f"{q['explanation']} - source: {ref}")
    head = f"### Score: {score}/{len(quiz)}"
    return head + "\n\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Chunk browser
# ---------------------------------------------------------------------------
CHUNK_HEADERS = ["chunk_id", "source", "page", "printed", "section", "chars", "text (preview)"]


def browse_chunks(keyword, source, page_from, page_to):
    r = get_retriever()
    kw = (keyword or "").lower().strip()
    rows = []
    for c in r.chunks:
        if source and source != "All sources" and c["source"] != source:
            continue
        if page_from and c["page"] < int(page_from):
            continue
        if page_to and c["page"] > int(page_to):
            continue
        if kw and kw not in c["text"].lower():
            continue
        rows.append([c["chunk_id"], c["source"], c["page"], c.get("printed_page") or "",
                     c.get("section") or "", len(c["text"]), c["text"][:160]])
    sizes = [row[5] for row in rows]
    stats = (f"**{len(rows)}** chunks match"
             + (f" - average {sum(sizes) // len(sizes)} characters (min {min(sizes)}, max {max(sizes)})"
                if sizes else "")
             + (" - showing the first 500" if len(rows) > 500 else ""))
    return rows[:500], stats


def show_chunk(evt: gr.SelectData, table):
    try:
        row = evt.index[0]
        chunk_id = int(table.iloc[row, 0]) if hasattr(table, "iloc") else int(table[row][0])
    except Exception:
        return "Click a row to see the full chunk."
    c = get_retriever().chunks[chunk_id]
    return f"**Chunk {chunk_id}** - {citation_label({**c, 'source_title': source_title(c['source'])})}\n\n{c['text']}"


# ---------------------------------------------------------------------------
# Compare two settings side by side
# ---------------------------------------------------------------------------
def compare(question, mode_a, k_a, mode_b, k_b):
    if not question or not question.strip():
        return "Type a question.", "", ""
    r = get_retriever()
    a = r.search(question, k=int(k_a), mode=mode_a)
    b = r.search(question, k=int(k_b), mode=mode_b)
    pages_a = {(p["source"], p["page"]) for p in a["passages"]}
    pages_b = {(p["source"], p["page"]) for p in b["passages"]}
    common = pages_a & pages_b

    def table(res, common_pages):
        rows = ["<table class='cmp-table'><tr><th>#</th><th>page</th><th>sim.</th><th>via</th><th>text</th></tr>"]
        for p in res["passages"]:
            cls = "cmp-same" if (p["source"], p["page"]) in common_pages else ""
            sim = f"{p['score']:.3f}" if p["score"] is not None else "-"
            rows.append(f"<tr class='{cls}'><td>{p['rank']}</td><td>{p['page']}</td><td>{sim}</td>"
                        f"<td>{'+'.join(p['found_by'])}</td><td>{highlight(p['text'], question, 220)}</td></tr>")
        rows.append("</table>")
        return "".join(rows)

    summary = (f"A ({a['mode']}, k={k_a}) found **{len(pages_a)}** pages - B ({b['mode']}, k={k_b}) "
               f"found **{len(pages_b)}** pages - **{len(common)}** pages in common (highlighted green). "
               "Tip: pages found only by keyword search usually contain exact terms or acronyms.")
    return summary, table(a, common), table(b, common)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def analytics():
    queries = read_queries()
    fb = read_feedback()
    costs = cost_summary()
    n = len(queries)
    refused = sum(1 for q in queries if q.get("refused"))
    scores = [q["top_score"] for q in queries if q.get("top_score") is not None]
    lat = [q["latency_s"] for q in queries if q.get("latency_s") is not None]
    up = sum(1 for f in fb if f.get("helpful"))
    cache_hits = sum(1 for q in queries if q.get("cached"))
    tool_q = sum(1 for q in queries if q.get("tools_used"))
    conf = {}
    for q in queries:
        if q.get("confidence"):
            conf[q["confidence"]] = conf.get(q["confidence"], 0) + 1
    md = [
        "### Usage",
        f"- Questions asked: **{n}** - refused / out of scope: **{refused}**"
        + (f" ({refused / n:.0%})" if n else ""),
        f"- Average best similarity: **{sum(scores) / len(scores):.2f}**" if scores else "- Average best similarity: -",
        f"- Average response time: **{sum(lat) / len(lat):.1f} s**" if lat else "- Average response time: -",
        f"- Feedback: **{up}** helpful / **{len(fb) - up}** not helpful",
        f"- Answer cache hits: **{cache_hits}** (free answers) - questions using calculator tools: **{tool_q}**",
        "- Confidence: " + (", ".join(f"{k} {v}" for k, v in conf.items()) or "-"),
        "",
        "### Cost (budget: $5 for the whole week)",
        f"- Total spent: **${costs['total_usd']:.4f}** in {costs['calls']} API calls",
    ]
    for op, v in sorted(costs["by_operation"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        md.append(f"  - {op}: {v['calls']} calls, {v['tokens']:,} tokens, ${v['cost_usd']:.4f}")
    md += ["", "### Most cited pages"]
    top = most_cited_pages(10)
    md += [f"- {p}: cited {c} time(s)" for p, c in top] or ["- nothing yet"]

    md += ["", "### Latest evaluation run (scripts/eval_runner.py)"]
    if EVAL_RESULTS_FILE.exists():
        try:
            ev = json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
            s = ev.get("summary", {})
            md.append(f"- Run: {ev.get('run', {}).get('tag')} ({ev.get('run', {}).get('timestamp')})")
            for key in ("retrieval_hit_rate", "retrieval_any_hit_rate", "mrr", "answer_accuracy",
                        "refusal_accuracy", "avg_pages_cited"):
                if key in s and s[key] is not None:
                    val = s[key]
                    md.append(f"- {key.replace('_', ' ')}: **{val:.0%}**" if key != "avg_pages_cited" and val <= 1
                              else f"- {key.replace('_', ' ')}: **{val:.2f}**")
            for cat, v in s.get("by_category", {}).items():
                md.append(f"  - {cat}: retrieval {v.get('retrieval_hit_rate', '-')}, "
                          f"answers {v.get('answer_accuracy', '-')}")
        except Exception as exc:
            md.append(f"- could not read eval results: {exc}")
    else:
        md.append("- no run yet: `python scripts/eval_runner.py`")

    recent = [[q.get("timestamp"), q.get("question"), q.get("refused"),
               round(q["top_score"], 3) if q.get("top_score") is not None else None,
               q.get("latency_s"), q.get("cost_usd")] for q in queries[-50:][::-1]]
    return "\n".join(md), recent


# ---------------------------------------------------------------------------
# Agent tab (Agentic AI project)
# ---------------------------------------------------------------------------
# The agent's tools, as shown in the UI (key = "tool" field of a trajectory step)
AGENT_TOOLS = [
    ("search_corpus", "🔎", "Search the book"),
    ("calculator", "🧮", "Calculator"),
    ("lookup_term", "📘", "Glossary"),
    ("get_section", "📖", "Chapter / section / page"),
    ("cash_flow_analysis", "📈", "Cash-flow analysis"),
    ("send_message", "📨", "Send message"),
    ("save_note", "💾", "Save note"),
]
TOOL_INFO = {key: (icon, label) for key, icon, label in AGENT_TOOLS}
ACTION_TOOL = {key: key for key, _, _ in AGENT_TOOLS}
ACTION_TOOL.update({"search": "search_corpus", "calculate": "calculator", "quote_exact": "get_section"})

AGENT_CSS = """
.tool-strip{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:2px 0 4px}
.tool-strip .ts-label{font-size:.85em;opacity:.75;margin-right:2px}
.tool-chip{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:4px 12px;
  font-size:.86em;border:1px solid rgba(128,128,128,.35);opacity:.55;transition:all .25s ease}
.tool-chip .cnt{font-weight:700;background:rgba(255,255,255,.7);color:#1b5e20;border-radius:999px;padding:0 7px}
.tool-chip.used{opacity:1;background:rgba(47,125,50,.14);border-color:#2f7d32}
.tool-chip.active{opacity:1;background:rgba(31,95,191,.16);border-color:#1f5fbf;animation:toolpulse 1.1s infinite}
@keyframes toolpulse{0%{box-shadow:0 0 0 0 rgba(31,95,191,.45)}70%{box-shadow:0 0 0 9px rgba(31,95,191,0)}100%{box-shadow:0 0 0 0 rgba(31,95,191,0)}}
.step-bar{display:flex;gap:4px;margin:4px 0 2px}
.step-bar span{flex:1;height:6px;border-radius:3px;background:rgba(128,128,128,.25)}
.step-bar span.done{background:#2f7d32}
.step-bar span.run{background:#1f5fbf;animation:toolpulse 1.1s infinite}
.step-caption{font-size:.8em;opacity:.75}
.agent-trace{font-family:ui-monospace,monospace;font-size:.78em;opacity:.85;border-left:3px solid #1f5fbf;
  padding:4px 8px;margin-top:6px;background:rgba(31,95,191,.06)}
"""


def tools_strip(trajectory: list[dict], active: str | None = None, running_step: int | None = None,
                max_steps: int = 6) -> str:
    """Chips for every tool (grey = not used, blue pulsing = running, green = used xN) + step bar."""
    counts: dict[str, int] = {}
    for s in trajectory:
        if s.get("tool"):
            counts[s["tool"]] = counts.get(s["tool"], 0) + 1
    chips = []
    for key, icon, label in AGENT_TOOLS:
        cls = "active" if key == active else ("used" if key in counts else "")
        cnt = f"<span class='cnt'>x{counts[key]}</span>" if key in counts else ""
        chips.append(f"<span class='tool-chip {cls}'>{icon} {label}{cnt}</span>")
    done = len([s for s in trajectory if s.get("action") not in (None, "max_steps_reached", "stopped", "plan")])
    bars = []
    for i in range(1, max_steps + 1):
        cls = "done" if i <= done else ("run" if running_step == i else "")
        bars.append(f"<span class='{cls}'></span>")
    if running_step:
        caption = f"Step {running_step} of max {max_steps} running..."
    elif done:
        caption = f"Finished in {done} step(s) - max {max_steps}"
    else:
        caption = f"The agent can take up to {max_steps} steps"
    return ("<div class='tool-strip'><span class='ts-label'>Tools:</span>" + "".join(chips) + "</div>"
            f"<div class='step-bar'>{''.join(bars)}</div><div class='step-caption'>{caption}</div>")


def _step_title(step: int, tool: str | None, action: str, running: bool) -> str:
    if tool:
        icon, label = TOOL_INFO[tool]
        return f"Step {step} · {icon} {label}" + ("..." if running else "")
    if action == "answer":
        return f"Step {step} · ✍️ Writing the final answer"
    if action == "plan":
        return "Step 0 · 🗺️ Planning" + ("..." if running else "")
    if action == "check":
        return f"Step {step} · 🔁 Answer check: retry"
    return f"Step {step} · 🤔 Deciding what to do next..."


def step_details(entry: dict) -> str:
    """Markdown shown inside the dropdown of one agent step."""
    lines = []
    if entry.get("thought"):
        lines.append(f"💭 *{entry['thought']}*")
    tool = entry.get("tool")
    res = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    if res.get("blocked"):
        lines.append(f"🚫 **Blocked:** {res.get('message')}")
        return "\n\n".join(lines)
    if tool in ("search_corpus", "lookup_term", "get_section"):
        what = {"search_corpus": ("Query", entry.get("query", "")), "lookup_term": ("Term", entry.get("term", "")),
                "get_section": ("Section", entry.get("section", ""))}[tool]
        lines.append(f"**{what[0]}:** `{what[1]}`")
        results = res.get("results", []) or []
        if tool == "search_corpus" and res.get("success"):
            top = res.get("top_rerank")
            lines.append(f"**Found:** {len(results)} passages ({res.get('mode', 'hybrid')} search"
                         + (", cross-encoder reranked" if res.get("reranked") else "") + ")"
                         + (f", best reranker score **{top:.2f}**" if top is not None else ""))
        if tool == "get_section" and res.get("success"):
            lines.append(f"**{res.get('section')}**" + (f" - pages {res['pages'][0]}-{res['pages'][-1]}"
                                                          if res.get("pages") else ""))
            if res.get("toc"):
                lines.append("\n".join(f"- {t}" for t in res["toc"][:30]))
        for p in results:
            url = pdf_link(p["source"], p["page"])
            page = f"p. {p['page']}" + (f" (printed {p['printed_page']})" if p.get("printed_page") else "")
            page = f"[{page}]({url})" if url else page
            rr = f" - rerank {p['rerank']:.2f}" if p.get("rerank") is not None else ""
            snippet = html.escape(p["text"][:220].replace("\n", " "))
            lines.append(f"- **[{p['n']}]** {page} - {p.get('section') or ''}{rr}\n  > {snippet}...")
        if not res.get("success"):
            lines.append(f"❌ {res.get('message', 'nothing found')}"
                         + (f" ({res['reason']})" if res.get("reason") else ""))
    elif tool == "calculator":
        out = res.get("result") if res.get("success") else res.get("message")
        lines.append(f"**Expression:** `{entry.get('expression', '')}`\n\n**Result:** `{out}`"
                     + (f"  *(engine: {res.get('engine')})*" if res.get("engine") else ""))
    elif tool == "cash_flow_analysis":
        args = entry.get("args") or {}
        lines.append(f"**Cash flows:** `{args.get('cash_flows')}`  **Rate:** `{args.get('rate_percent')}`%")
        if res.get("success"):
            for k in ("npv", "irr_percent", "payback_years", "discounted_payback_years",
                      "profitability_index", "decision", "note"):
                if res.get(k) is not None:
                    lines.append(f"- {k.replace('_', ' ')}: **{res[k]}**")
        else:
            lines.append(f"❌ {res.get('message')}")
    elif tool == "send_message":
        status = res.get("status", "")
        lines.append(f"**To:** `{entry.get('recipient', '')}`\n\n> {entry.get('message', '')}")
        lines.append("🚫 Declined automatically (fake send mode) - nothing was sent." if status == "declined"
                     else "⏸️ Waiting for your approval - nothing has been sent.")
    elif tool == "save_note":
        status = res.get("status", "")
        lines.append(f"**Note:** `{entry.get('title', '')}`")
        if res.get("preview"):
            lines.append("```markdown\n" + res["preview"][:1500] + "\n```")
        lines.append("🚫 Declined automatically (fake mode) - nothing was saved." if status == "declined"
                     else "⏸️ Waiting for your approval - nothing has been written.")
    elif entry.get("action") not in ("answer", "check", "plan") and res.get("message"):
        lines.append(f"❌ {res.get('message')}")
    elif entry.get("action") == "answer":
        lines.append("The agent has enough information and writes the answer below.")
    elif entry.get("action") == "plan":
        lines.append("\n".join(f"- {p}" for p in res.get("plan") or []) or "No plan returned - deciding step by step.")
    elif entry.get("action") == "check":
        lines.append(f"**Rejected draft:** {html.escape(str(entry.get('rejected_answer', ''))[:300])}")
        lines.append(f"🔁 {res.get('message')}")
    return "\n\n".join(lines) or "-"


def format_trajectory(trajectory: list[dict]) -> str:
    return "```json\n" + json.dumps(trajectory, indent=2, ensure_ascii=False, default=str)[:60000] + "\n```"


def render_agent_answer(result: dict) -> str:
    """Agent answer for the chat bubble: clickable [n] citations when it searched the book."""
    if result.get("citations"):
        body = render_answer({**result, "calculations": [], "verification": {}, "confidence": {},
                              "cached": "", "low_confidence": False, "top_score": None})
    else:
        body = result["answer"]
    if result.get("trace"):
        body += ("\n\n<div class='agent-trace'><b>Step trace</b> · "
                 f"{result.get('steps')} step(s) · {result.get('model_calls')} model call(s) · "
                 f"{result.get('latency_s')} s · stop: {result.get('stop_reason')}<br>"
                 + "<br>".join(html.escape(t) for t in result["trace"]) + "</div>")
    return body


def agent_history(history) -> list[dict]:
    """Chat messages as sent back by Gradio, KEEPING the step dropdowns (metadata)."""
    out = []
    for m in history or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        msg = {"role": m["role"], "content": msg_text(m)}
        meta = m.get("metadata")
        if meta:
            meta = {k: v for k, v in dict(meta).items() if v not in (None, "")}
            if meta.get("title"):
                msg["metadata"] = meta
        out.append(msg)
    return out


class _Escalate(Exception):
    """Router: the RAG pipeline refused, hand the question to the agent."""


def route_badge(r: dict) -> str:
    if not r.get("route_reason"):
        return ""
    return (f"🧭 Router -> **{'RAG pipeline' if r.get('route') == 'rag' else 'agent'}** "
            f"({r['route_reason']}) - ")


def run_agent_chat(message, history, agent_results, fake_send=False, injection=False, prompt_version=None,
                   plan=False, use_router=False):
    """Generator: (chatbot, textbox, sources, tools strip, trajectory, status, approval box,
    approval text, pending recipient, pending message, results).
    Every agent step is a collapsible message ("dropdown") in the chat, updated live."""
    from app.agent import MAX_STEPS, run_agent_steps
    history = agent_history(history)
    agent_results = list(agent_results or [])
    hide = gr.update(visible=False)
    if not message or not message.strip():
        yield (history, "", gr.update(), gr.update(), gr.update(), "Please type a request.",
               hide, "", "", "", agent_results)
        return
    message = message.strip()
    past = [m for m in history if "metadata" not in m]          # the LLM only sees real Q/A
    history = history + [{"role": "user", "content": message}]
    trajectory: list[dict] = []
    current: dict | None = None                                 # the dropdown of the running step

    def out(status, active=None, running=None):
        return (history, "", gr.update(), tools_strip(trajectory, active, running, MAX_STEPS),
                gr.update(), status, hide, "", "", "", agent_results)

    route_info = {}
    if use_router:
        from app.router import rag_as_agent_result, route
        route_info = route(message)
        if route_info["route"] == "rag":
            yield out(f"🧭 Router -> RAG pipeline ({route_info['reason']})...")
            try:
                from app import rag
                t0 = time.time()
                res = rag.answer(message, history=past, settings={"use_cache": False, "followups": False})
                if res.get("refused"):                 # escalate: the agent can rephrase and retry
                    route_info = {"route": "agent", "reason": "RAG refused -> escalated to the agent"}
                    raise _Escalate()
                r = rag_as_agent_result(message, res, route_info["reason"], time.time() - t0)
                history.append({"role": "assistant", "content": render_agent_answer(r)})
                agent_results.append(r)
                panel = sources_html(r) if r.get("citations") else gr.update()
                yield (history, "", panel, tools_strip([], None, None, MAX_STEPS), "Routed to the RAG pipeline: "
                       "no agent trajectory.", route_badge(r) + f"answered in {r['latency_s']} s with 1 search "
                       "and 1 generation call", hide, "", "", "", agent_results)
            except _Escalate:
                pass
            except Exception as exc:
                from app.llm import friendly_error
                history.append({"role": "assistant", "content": f"**Error:** {html.escape(friendly_error(exc))}"})
                yield (history, "", gr.update(), gr.update(), gr.update(), "Error", hide, "", "", "", agent_results)
                return
            else:
                return

    yield out("Agent working...")
    try:
        from app.agent import AGENT_PROMPT_VERSION
        for event in run_agent_steps(message, past, send_mode="fake" if fake_send else "real",
                                     injection=bool(injection), injection_mode="act" if injection else None,
                                     prompt_version=prompt_version or AGENT_PROMPT_VERSION,
                                     plan={"auto": "auto", "always": True}.get(plan, bool(plan) and plan != "off")):
            kind = event["type"]
            if kind == "thinking":
                n = event["step"]
                current = {"role": "assistant", "content": "The model is choosing the next action...",
                           "metadata": {"title": _step_title(n, None, "", True), "status": "pending",
                                        "id": f"step-{len(agent_results)}-{n}"}}
                history.append(current)
                yield out(f"Step {n}: thinking...", running=n)
            elif kind == "decided":
                n, tool = event["step"], ACTION_TOOL.get(event["action"])
                tool = tool if tool in TOOL_INFO else None
                current["metadata"]["title"] = _step_title(n, tool, event["action"], True)
                current["content"] = (f"💭 *{event['thought']}*\n\n" if event["thought"] else "") + \
                    ("Running the tool..." if tool else "Writing the answer...")
                label = TOOL_INFO[tool][1] if tool else "answer"
                yield out(f"Step {n}: {label}...", active=tool, running=n)
            elif kind == "step":
                s = event["step"]
                trajectory.append(s)
                current["metadata"].update(title=_step_title(s["step"], s.get("tool") if s.get("tool") in TOOL_INFO
                                                             else None, s["action"], False),
                                           status="done", duration=s.get("seconds"))
                current["content"] = step_details(s)
                yield out(f"Step {s['step']} done.")
            else:                                               # done
                r = event["result"]
                if route_info:
                    r["route"], r["route_reason"] = "agent", route_info["reason"]
                history.append({"role": "assistant", "content": render_agent_answer(r)})
                agent_results.append(r)
                panel = sources_html(r) if r["citations"] else (
                    "<div class='src-head'>The agent did not search the book for this request.</div>")
                tools_used = ", ".join(dict.fromkeys(TOOL_INFO[t][1] for t in r["tools_used"]
                                                     if t in TOOL_INFO)) or "none"
                status = route_badge(r) + (f"Done in {r['steps']} step(s) - tools used: {tools_used} - "
                          f"{r.get('model_calls')} model call(s), {r.get('latency_s')} s - stop: {r.get('stop_reason')}"
                          + (" - ⚠️ the injection test chunk was retrieved" if r.get("injection_chunk_retrieved") else "")
                          + (f" - 🔁 {r['nudges']} answer check retry" if r.get("nudges") else ""))
                strip = tools_strip(r["trajectory"], None, None, MAX_STEPS)
                if r["approval_required"]:
                    p = r["pending_message"]
                    if p.get("kind") == "note":
                        box = (f"### ⏸️ Approval required: save this note?\n**File:** `{p.get('path')}`\n\n"
                               "**Preview (exactly what will be written):**\n\n```markdown\n"
                               + p["message"][:3000] + "\n```")
                    else:
                        box = (f"### ⏸️ Approval required\n**Recipient:** `{p['recipient']}`\n\n"
                               f"**Message:**\n\n> {p['message']}")
                    yield (history, "", panel, strip, format_trajectory(r["trajectory"]),
                           status + " - waiting for your approval", gr.update(visible=True), box,
                           p["recipient"], p["message"], agent_results)
                else:
                    yield (history, "", panel, strip, format_trajectory(r["trajectory"]), status,
                           hide, "", "", "", agent_results)
    except Exception as exc:                      # show the error instead of crashing the UI
        from app.llm import friendly_error
        msg = friendly_error(exc)
        if current is not None and current["metadata"].get("status") == "pending":
            current["metadata"].update(status="done", title=current["metadata"]["title"] + " (failed)")
        history.append({"role": "assistant", "content": f"**Error:** {html.escape(msg)}"})
        yield (history, "", gr.update(), tools_strip(trajectory, None, None, MAX_STEPS), gr.update(),
               f"Error: {msg}", hide, "", "", "", agent_results)


def approve_message(recipient, pending_message):
    """Human approval: only now is the message sent."""
    from app.message_tool import send_message
    if not recipient or not pending_message:
        return "**No pending message.**", gr.update(visible=False), "", ""
    if recipient.startswith("note:"):                  # second action tool: save a study note
        from app.note_tool import save_note
        res = save_note(recipient[5:], pending_message, approved=True)
        if res["status"] == "saved":
            text = f"✅ **Note saved** - outcome checked: `{res['path']}` exists ({res['bytes']} bytes, content verified)."
        else:
            text = f"❌ **Note NOT saved**: {res.get('message', 'write failed')}"
        return text, gr.update(visible=False), "", ""
    result = send_message(recipient=recipient, message=pending_message, approved=True)
    status = result.get("status")
    if status == "sent":
        text = (f"✅ **Message delivered** via {result.get('channel')} to `{recipient}` - outcome checked: "
                f"HTTP {result.get('http_status')}, response `{result.get('detail')}`.\n\n> {pending_message}")
    elif status == "logged_only":
        text = (f"📝 **Logged only** (no Slack/Teams webhook in .env) - saved in data/sent_messages.jsonl, "
                f"NOT delivered to anyone.\n\n> {pending_message}")
    else:
        text = (f"❌ **Message NOT delivered** via {result.get('channel', '?')}: HTTP {result.get('http_status')} "
                f"- {result.get('detail') or result.get('message', 'unknown error')}")
    return text, gr.update(visible=False), "", ""


def reject_message():
    return "**Rejected - nothing was sent or saved.**", gr.update(visible=False), "", ""


AGENT_EXAMPLES = [
    "What is the current ratio? Compute it for current assets of 50,000 and current liabilities of 20,000.",
    "Define 'capital structure' using the book's glossary, then explain why it matters.",
    "What is the NPV at 10% of a project costing 1,000 that returns 400 per year for 3 years? Should we accept it?",
    "What does section 7.4 cover? Give its page range.",
    "Give me the exact text of page 229 of the textbook.",
    "Using the rule of 72, how long does money take to double at 8%? Explain the rule from the book.",
    "Summarize what the book says about diversification and prepare a message to my team (team-rudra) with the summary.",
    "Who won the 2022 FIFA World Cup?",
    "Explain the rule of 72 from the book and save it as a study note.",
]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
EXAMPLES = [
    "What is the current ratio and how is it interpreted?",
    "What is diversification and why do investors practice it?",
    "According to the rule of 72, how long does money take to double at 9%? And at 6%?",
    "What does DDM stand for, and which DDM is the most common?",
    "Compare the payback period method with the NPV method.",
    "What do the trade credit terms 2/10, net 30 mean?",
    "Tell me about returns.",
    "Summarize chapter 7.",
    "Who won the 2022 FIFA World Cup?",
]


def index_banner(r: Retriever) -> str:
    msg = "**Knowledge base:** " + ", ".join(source_title(s) for s in r.sources)
    if not has_api_key():
        msg += ("\n\n**Offline mode:** no `OPENAI_API_KEY` in `.env`. Keyword search only and "
                "extractive answers. Add your key to get generated answers.")
    return msg


def create_interface():
    r = get_retriever()
    source_choices = ["All sources"] + r.sources
    prompt_versions = sorted(p.stem for p in PROMPTS_DIR.glob("v*.txt")) or [PROMPT_VERSION]

    blocks_kwargs = {"title": "Personal Finance Assistant - Team Rudra"}
    if GRADIO_MAJOR < 6:
        blocks_kwargs.update(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=PANEL_CSS + AGENT_CSS)

    with gr.Blocks(**blocks_kwargs) as demo:
        results_state = gr.State([])

        gr.Markdown("# Personal Finance Assistant by Team Rudra")
        gr.Markdown(index_banner(r))

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = _kw(gr.Chatbot, label="Conversation", height=520, type="messages",
                                  feedback_options=["Like", "Dislike"], show_copy_button=True,
                                  buttons=["copy"])
                    with gr.Row():
                        question = gr.Textbox(placeholder="Ask a question... (follow-up questions work too)",
                                              show_label=False, lines=2, scale=5)
                        send = gr.Button("Send", variant="primary", scale=1)
                    status = gr.Markdown("Ready.")
                    with gr.Row():
                        fu_btns = [gr.Button(visible=False, size="sm") for _ in range(3)]
                    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=question, label="Try these")

                    with gr.Accordion("My profile - personal mode (optional)", open=False):
                        gr.Markdown("Enter your own numbers, tick *Use my profile*, then ask about **your** "
                                    "situation, e.g. *How much should I save each month to reach my goal?* "
                                    "The answer explains the book's method (with citations) and applies it to "
                                    "your numbers with the calculator tools. Nothing is saved.")
                        with gr.Row():
                            p_on = gr.Checkbox(False, label="Use my profile")
                            p_currency = gr.Dropdown(["EUR", "USD", "GBP", "INR"], value="EUR", label="Currency")
                            p_risk = gr.Dropdown(["low", "medium", "high"], value="medium", label="Risk preference")
                        with gr.Row():
                            p_income = gr.Number(label="Monthly net income", precision=0)
                            p_expenses = gr.Number(label="Monthly expenses", precision=0)
                            p_savings = gr.Number(label="Current savings", precision=0)
                        with gr.Row():
                            p_msave = gr.Number(label="Saved per month", precision=0)
                            p_goal = gr.Number(label="Savings goal", precision=0)
                            p_years = gr.Number(label="Years to goal")
                            p_return = gr.Number(label="Expected return % / year")

                    with gr.Accordion("Search & answer settings", open=False):
                        with gr.Row():
                            top_k = gr.Slider(3, 15, value=DEFAULT_TOP_K, step=1, label="Passages (top-k)")
                            threshold = gr.Slider(0.0, 0.8, value=CONFIDENCE_THRESHOLD, step=0.01,
                                                  label="Confidence threshold (refuse below)")
                        with gr.Row():
                            mode = gr.Radio(["hybrid", "dense", "bm25"], value=DEFAULT_SEARCH_MODE,
                                            label="Search mode")
                            source_filter = gr.Dropdown(source_choices, value="All sources",
                                                        label="Search in")
                            prompt_version = gr.Dropdown(prompt_versions, value=PROMPT_VERSION,
                                                         label="Prompt version")
                        with gr.Row():
                            expand = gr.Checkbox(False, label="Query expansion (2 paraphrases, +1 LLM call)")
                            rerank = gr.Checkbox(USE_RERANK and r.rerank_available(),
                                                 label="Cross-encoder rerank (top-20 -> top-k)",
                                                 interactive=r.rerank_available(),
                                                 info=None if r.rerank_available()
                                                 else "install sentence-transformers to enable")
                            rerank_thr = gr.Slider(-10, 10, value=RERANK_THRESHOLD, step=0.5,
                                                   label="Reranker threshold (refuse below)")
                        with gr.Row():
                            rewrite = gr.Checkbox(USE_QUERY_REWRITE, label="Query rewriting for follow-ups")
                            hyde = gr.Checkbox(USE_HYDE, label="HyDE (+1 LLM call)")
                            injection = gr.Checkbox(INJECTION_TEST, label="Injection TEST chunk (demo only)")
                        with gr.Row():
                            style = gr.Dropdown(list(features.STYLES), value="Standard", label="Answer style")
                            language = gr.Radio(["Auto", "English", "Francais"], value="Auto",
                                                label="Answer language")
                        with gr.Row():
                            tools = gr.Checkbox(True, label="Calculator tools for calculation questions")
                            verify = gr.Checkbox(False, label="Citation check (+1 LLM call)")
                            followups = gr.Checkbox(True, label="Suggest follow-up questions")
                            use_cache = gr.Checkbox(True, label="Answer cache (exact + semantic)")
                        clear_cache = gr.Button("Clear answer cache", size="sm")
                        cache_msg = gr.Markdown()

                    with gr.Accordion("Feedback & export", open=False):
                        comment = gr.Textbox(label="Comment (optional)", lines=1)
                        with gr.Row():
                            good = gr.Button("Helpful")
                            bad = gr.Button("Not helpful")
                        fb_status = gr.Markdown()
                        with gr.Row():
                            exp_md = gr.Button("Export conversation (.md)")
                            exp_pdf = gr.Button("Export conversation (.pdf)")
                        export_file = gr.File(label="Download", interactive=False)
                    clear = gr.Button("New conversation")

                with gr.Column(scale=2):
                    gr.Markdown("### Sources")
                    with gr.Accordion("Source pages (PDF, cited passage highlighted)", open=True):
                        page_gallery = _kw(gr.Gallery, label="Cited pages", columns=2, height=360,
                                           object_fit="contain", show_label=False)
                    sources_panel = gr.HTML(sources_html(None))

            inputs = [question, chatbot, results_state, top_k, mode, threshold, expand, rerank,
                      source_filter, prompt_version, style, language, tools, verify, followups,
                      use_cache, p_on, p_income, p_expenses, p_savings, p_msave, p_goal, p_years,
                      p_return, p_risk, p_currency, rerank_thr, rewrite, hyde, injection]
            outputs = [chatbot, question, sources_panel, status, results_state] + fu_btns
            send.click(respond, inputs, outputs).then(source_pages, results_state, page_gallery)
            question.submit(respond, inputs, outputs).then(source_pages, results_state, page_gallery)
            for b in fu_btns:          # clicking a suggestion asks it as the next question
                b.click(lambda text: text, b, question).then(respond, inputs, outputs) \
                    .then(source_pages, results_state, page_gallery)
            clear_cache.click(lambda: (features.cache_clear(), "Answer cache cleared.")[1], None, cache_msg)
            clear.click(clear_chat, None, [chatbot, results_state, sources_panel, status, export_file]) \
                .then(lambda: NO_FU, None, fu_btns)
            good.click(lambda c, s: feedback(True, c, s), [comment, results_state], fb_status)
            bad.click(lambda c, s: feedback(False, c, s), [comment, results_state], fb_status)
            exp_md.click(export_md, results_state, export_file)
            exp_pdf.click(export_pdf, results_state, export_file)
            if hasattr(chatbot, "like"):
                chatbot.like(on_like, [results_state, chatbot], None)

        with gr.Tab("Saved conversations"):
            gr.Markdown("Same RAG pipeline, with a **conversation list** like ChatGPT's sidebar: "
                        "conversations are stored in your browser (`gr.ChatInterface(save_history=True)`), "
                        "and you can switch between them.")
            _kw(gr.ChatInterface, fn=saved_chat_answer, type="messages", save_history=True,
                examples=["What is diversification?", "What does DDM stand for?"],
                chatbot=_kw(gr.Chatbot, height=460, type="messages"))

        with gr.Tab("Agent (tools)"):
            gr.Markdown("**Agentic assistant.** gpt-4o-mini decides by itself which tool to use, step by "
                        "step (max 6). Each step appears in the chat as a dropdown - click it to see why "
                        "the agent chose that tool, its input and its result; the numbered step trace is shown "
                        "under the answer and every run is logged to `data/agent_traces.jsonl`. "
                        "**A message is never sent and a note never saved without your approval.**")
            agent_state = gr.State([])
            pending_recipient = gr.State("")
            pending_text = gr.State("")
            with gr.Row():
                with gr.Column(scale=3):
                    agent_tools = gr.HTML(tools_strip([], None, None, 6))
                    agent_chat = _kw(gr.Chatbot, label="Agent conversation", height=520, type="messages",
                                     buttons=["copy"], show_copy_button=True)
                    with gr.Row():
                        agent_q = gr.Textbox(placeholder="Ask the agent (questions, calculations, "
                                                         "exact quotes, messages)...",
                                             show_label=False, lines=2, scale=5)
                        agent_send = gr.Button("Ask agent", variant="primary", scale=1)
                    agent_status = gr.Markdown("Ready.")
                    with gr.Group(visible=False) as approval_box:
                        approval_text = gr.Markdown()
                        with gr.Row():
                            approve_btn = gr.Button("Approve & send", variant="primary")
                            reject_btn = gr.Button("Reject", variant="stop")
                    approval_result = gr.Markdown()
                    gr.Examples(examples=[[e] for e in AGENT_EXAMPLES], inputs=agent_q,
                                label="Try these")
                    with gr.Accordion("Raw agent trajectory (JSON)", open=False):
                        agent_traj = gr.Markdown("The trajectory will appear here.")
                    with gr.Accordion("Agent settings (tests / demo)", open=False):
                        with gr.Row():
                            agent_fake = gr.Checkbox(False, label="Fake send (decline every message)")
                            agent_inj = gr.Checkbox(False, label="Injection test chunk ('act' mode)")
                            agent_prompt = gr.Radio(["agent_v3", "agent_v2", "agent_v1"], value="agent_v3",
                                                    label="Agent prompt (v3 = ground-first + answer check, "
                                                          "v2 = hardened, v1 = baseline)")
                        with gr.Row():
                            agent_plan = gr.Radio(["off", "auto", "always"], value="off",
                                                  label="Plan first (plan-then-execute, +1 model call; "
                                                        "auto = only multi-part questions)")
                            agent_router = gr.Checkbox(False, label="Router: simple lookups go to the RAG pipeline")
                    agent_clear = gr.Button("New agent conversation")
                with gr.Column(scale=2):
                    gr.Markdown("### Sources")
                    agent_sources = gr.HTML("<div class='src-head'>Sources will appear here.</div>")

            a_inputs = [agent_q, agent_chat, agent_state, agent_fake, agent_inj, agent_prompt, agent_plan,
                        agent_router]
            a_outputs = [agent_chat, agent_q, agent_sources, agent_tools, agent_traj, agent_status,
                         approval_box, approval_text, pending_recipient, pending_text, agent_state]
            agent_send.click(run_agent_chat, a_inputs, a_outputs)
            agent_q.submit(run_agent_chat, a_inputs, a_outputs)
            approve_btn.click(approve_message, [pending_recipient, pending_text],
                              [approval_result, approval_box, pending_recipient, pending_text])
            reject_btn.click(reject_message, None,
                             [approval_result, approval_box, pending_recipient, pending_text])
            agent_clear.click(lambda: ([], [], "Conversation cleared.", gr.update(visible=False), "", "", "",
                                       tools_strip([], None, None, 6)),
                              None, [agent_chat, agent_state, agent_status, approval_box,
                                     approval_result, pending_recipient, pending_text, agent_tools])

        with gr.Tab("Quiz"):
            gr.Markdown("Study mode: multiple-choice questions generated from one section of the book "
                        "(or a topic), each linked to the page it comes from. About $0.001 per quiz.")
            sections = features.list_sections(r.chunks)
            with gr.Row():
                q_section = gr.Dropdown(sections, value=sections[0] if sections else None,
                                        label="Section", scale=3)
                q_topic = gr.Textbox(label="...or a topic (overrides the section)", scale=3)
                q_n = gr.Slider(3, MAX_QUIZ, value=5, step=1, label="Questions", scale=2)
                q_lang = gr.Radio(["English", "Francais"], value="English", label="Language", scale=2)
            q_go = gr.Button("Generate quiz", variant="primary")
            q_msg = gr.Markdown()
            quiz_state = gr.State([])
            q_slots = [gr.Radio(choices=[], visible=False, label=f"Q{i + 1}") for i in range(MAX_QUIZ)]
            q_check = gr.Button("Check my answers")
            q_result = gr.Markdown()
            q_go.click(make_quiz, [q_section, q_topic, q_n, q_lang], [quiz_state] + q_slots + [q_msg])
            q_check.click(check_quiz, [quiz_state] + q_slots, q_result)

        with gr.Tab("Chunk browser"):
            gr.Markdown("Look at the chunks that were embedded - useful to spot bad chunking.")
            with gr.Row():
                kw = gr.Textbox(label="Keyword")
                src = gr.Dropdown(source_choices, value="All sources", label="Source")
                p_from = gr.Number(label="Page from", precision=0)
                p_to = gr.Number(label="Page to", precision=0)
                go = gr.Button("Search", variant="primary")
            chunk_stats = gr.Markdown()
            chunk_table = _kw(gr.Dataframe, headers=CHUNK_HEADERS, interactive=False, wrap=True,
                              max_height=420)
            chunk_full = gr.Markdown("Click a row to see the full chunk.")
            go.click(browse_chunks, [kw, src, p_from, p_to], [chunk_table, chunk_stats])
            kw.submit(browse_chunks, [kw, src, p_from, p_to], [chunk_table, chunk_stats])
            chunk_table.select(show_chunk, chunk_table, chunk_full)

        with gr.Tab("Compare settings"):
            gr.Markdown("Run the same question with two retrieval settings (only retrieval, no answer: "
                        "costs at most one embedding).")
            cq = gr.Textbox(label="Question", value="What does DDM stand for?")
            with gr.Row():
                with gr.Column():
                    ma = gr.Radio(["hybrid", "dense", "bm25"], value="dense", label="Setting A - mode")
                    ka = gr.Slider(3, 15, value=5, step=1, label="A - top-k")
                with gr.Column():
                    mb = gr.Radio(["hybrid", "dense", "bm25"], value="hybrid", label="Setting B - mode")
                    kb = gr.Slider(3, 15, value=8, step=1, label="B - top-k")
            cgo = gr.Button("Compare", variant="primary")
            csum = gr.Markdown()
            with gr.Row():
                ca = gr.HTML()
                cb = gr.HTML()
            cgo.click(compare, [cq, ma, ka, mb, kb], [csum, ca, cb])

        with gr.Tab("Analytics"):
            refresh = gr.Button("Refresh")
            an_md = gr.Markdown()
            an_table = _kw(gr.Dataframe, headers=["time", "question", "refused", "best sim.", "latency s", "cost $"],
                           interactive=False, wrap=True, max_height=400)
            refresh.click(analytics, None, [an_md, an_table])
            demo.load(analytics, None, [an_md, an_table])

        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS)

    return demo


HOW_IT_WORKS = """
### Pipeline
```
question ──► (follow-up? rewrite with chat history) ──► standalone question
          ──► BM25 keyword search  ┐
          ──► FAISS vector search  ┴─► Reciprocal Rank Fusion ─► max 2 chunks per page ─► top-k passages
          ──► best similarity < threshold ? ──► refuse (no LLM call)
          ──► gpt-4o-mini + versioned prompt + numbered passages [1]..[k] ──► streamed answer with [n]
          ──► [n] parsed ──► clickable citations + Sources panel
```
* **Citations**: every `[n]` in the answer is a link that opens the PDF at the cited page. The
  Sources panel lists every retrieved passage, marks the ones the answer cites, shows the
  section, PDF page and printed page, the similarity score and highlights the words of your question.
* **Several sources**: passages are numbered and the prompt asks the model to cite *all* the
  passages it uses; the retriever never takes more than 2 chunks from the same page, so answers
  are built from several pages.
* **Conversation memory**: the last 3 exchanges are sent with each question, and follow-ups
  such as *"and at 6%?"* are rewritten into a standalone question before searching.
* **Out-of-scope**: questions whose best match is below the confidence threshold are refused
  without calling the LLM; the prompt also refuses when the passages do not contain the answer.
* **Offline mode**: without an API key the app still works with keyword search and extractive answers.
* **Calculator tools**: for calculation questions gpt-4o-mini calls exact Python functions (rule of 72,
  future/present value, EAR, NPV, IRR, payback, loan payment, ratios, safe calculator); every
  calculation is shown under the answer.
* **Confidence badge & citation check**: high / medium / low from similarity and citation coverage;
  optionally a second call checks each cited sentence against its passage.
* **Follow-up suggestions**: three clickable next questions after each answer.
* **Answer cache**: the same (or same-meaning, cosine >= 0.97) question is answered for $0.
* **Answer styles & French**: beginner / detailed / 3-bullet styles; French questions are translated
  to English for the search and answered in French.
* **Quiz tab**: multiple-choice questions generated from a section or topic, scored, with page links.
* **Proof sentences**: for every citation the exact sentence of the passage that supports the claim is
  found (word overlap, no API call) and shown in green in the Sources panel.
* **Section summaries**: broad questions ("Summarize chapter 7") also get short summaries of the matching
  sections (written once by `scripts/build_summaries.py`), cited like any passage.
* **Personal profile mode**: enter your numbers (savings, goal, years, return); questions about *your*
  situation apply the book's method to them with the calculator tools (savings plan, savings growth ...).

### Agent tab (Agentic AI project)
```
request ──► gpt-4o-mini chooses ONE action (JSON) ──► tool runs ──► result added to the trajectory
   ▲                                                                         │
   └──────────────────────── next step (max 6) ◄─────────────────────────────┘
tools: search_corpus (reranked, "No matching passages found" below the threshold) · calculator (simpleeval)
       · lookup_term (book glossary) · get_section (chapter / section / page) · cash_flow_analysis (NPV, IRR,
       payback) · send_message (prepared only → human clicks Approve & send → delivery checked) · answer
```
* Same search engine, citations and Sources panel as the Chat tab.
* Guard rails: max 6 steps, max 2 retries after an empty search, repeated identical calls blocked,
  tool errors returned to the model (max 3), only the last 2 exchanges of history.
* Every step is shown (dropdowns + numbered trace under the answer + JSON) and logged to data/agent_traces.jsonl.
* Safety: tool results are wrapped in `<context>` (prompt agent_v2), a message is never sent automatically,
  "Fake send" declines every proposal (used by the injection test and the agent eval).
* Prompt agent_v3 + answer check: book facts are searched before calculating, and an answer with no
  book passage (or a skipped "send to my team") is sent back to the agent once ("🔁 Answer check").
* Settings: "Plan first" (plan-then-execute, the plan is step 0) and "Router" (simple lookups go to the
  RAG pipeline, the rest to the agent; the reason is shown in the status line).
"""


def main():
    """
    python app/app.py                 local only (http://127.0.0.1:7860)
    python app/app.py --share         + temporary public link (the jury can open it on their laptops)
    python app/app.py --lan           reachable from other devices on the same Wi-Fi
    python app/app.py --port 7861     another port
    Optional password: put APP_USERNAME=... and APP_PASSWORD=... in .env
    """
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    ap.add_argument("--lan", action="store_true", help="listen on 0.0.0.0 (same network)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo = create_interface()
    launch_kwargs = {"server_name": "0.0.0.0" if args.lan else "127.0.0.1",
                     "server_port": args.port, "share": args.share,
                     "allowed_paths": [str(SOURCES_DIR)]}
    user, password = os.getenv("APP_USERNAME", "").strip(), os.getenv("APP_PASSWORD", "").strip()
    if user and password:
        launch_kwargs["auth"] = (user, password)          # login page before the app
        launch_kwargs["auth_message"] = "Personal Finance RAG - Team Rudra"
    elif args.share:
        print("WARNING: public link without password. Anyone with the link can use your API key "
              "budget. Set APP_USERNAME and APP_PASSWORD in .env to protect it.")
    if GRADIO_MAJOR >= 6:
        launch_kwargs.update(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=PANEL_CSS + AGENT_CSS)
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    main()
