"""
agent.py - Agentic Personal Finance Assistant (Agentic AI project, day 1 + day 2).

A plain Python loop (no framework): at each step gpt-4o-mini returns ONE JSON action,
we run the tool, append the result to the trajectory and ask again, until it answers.

    Tools (app/agent_tools.py)
      search_corpus       reranked hybrid search, "No matching passages found" below the threshold
      calculator          simpleeval
      lookup_term         OWN TOOL 1 - the book's glossary (Key Terms of every chapter)
      get_section         OWN TOOL 2 - a chapter / section / page / table of contents
      cash_flow_analysis  OWN TOOL 3 - NPV, IRR, payback, discounted payback, PI
      send_message        action tool (app/message_tool.py): prepared only, a human approves it
      save_note           action tool (app/note_tool.py): a Markdown study note with sources,
                          previewed, written only after human approval

    Guard rails
      MAX_STEPS = 6                      hard cap on steps (stop_reason "max_steps")
      MAX_SEARCH_RETRIES = 2             after 1 + 2 empty searches -> refusal ("retry_limit")
      MAX_TOOL_ERRORS = 3                tool errors are returned to the model so it can fix
                                         its arguments, but only 3 times ("tool_errors")
      repeated identical call            blocked (not executed); the 2nd time -> stop ("repeated_calls")
      history                            only the last MAX_HISTORY_TURNS exchanges
      prompt agent_v3                    v2 (tool results in <context>, data never instructions)
                                         + ground-first, one-best-tool and send trigger rules
      answer check (v3)                  ONE nudge if the agent answers a finance question with no
                                         book passage, or skips a send_message the user asked for

    Options (measured with scripts/agent_eval.py)
      plan=True                          plan-then-execute: one extra call writes a numbered plan
                                         (step 0 of the trajectory), then the loop follows it
      app/router.py                      router: simple lookups go straight to the RAG pipeline

    Observability
      every run -> one line in data/agent_traces.jsonl (question, numbered steps, answer,
      stop reason, latency, model calls, tokens); every model call is also in the cost log.

    python app/agent.py          command-line test
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import tracing  # noqa: E402
from app.agent_tools import NO_MATCH, TOOLS  # noqa: E402
from app.config import (DATA_DIR, MAX_HISTORY_TURNS, PROMPTS_DIR, REFUSAL_TEXT,  # noqa: E402
                        has_api_key, source_title)
from app.message_tool import SEND_MODE, fake_send, prepare_message  # noqa: E402
from app.note_tool import fake_save, prepare_note  # noqa: E402

MAX_STEPS = 6                  # hard cap on agent steps (spec: 5-6)
MAX_SEARCH_RETRIES = 2         # extra searches allowed after "No matching passages found"
MAX_TOOL_ERRORS = 3            # failed tool calls given back to the model before we stop
PASSAGE_CHARS = 1200           # text of each passage shown to the model
AGENT_PROMPT_VERSION = "agent_v3"
MAX_NUDGES = 2                 # answer-check nudges per run; all problems reported together, each kind once
TRACE_FILE = DATA_DIR / "agent_traces.jsonl"

# Old action names (Day-2 agent) still accepted
ALIASES = {"search": "search_corpus", "calculate": "calculator", "quote_exact": "get_section",
           "define": "lookup_term", "glossary": "lookup_term", "section": "get_section",
           "cash_flows": "cash_flow_analysis", "npv": "cash_flow_analysis", "final": "answer",
           "save": "save_note", "export": "save_note", "note": "save_note"}
ARG_NAMES = {"search_corpus": ["query"], "calculator": ["expression"], "lookup_term": ["term"],
             "get_section": ["section"], "cash_flow_analysis": ["cash_flows", "rate_percent"],
             "send_message": ["recipient", "message"], "save_note": ["title", "content"]}
ACTION_TOOLS = ("send_message", "save_note")      # act on the outside world -> human approval


# ============================================================
# ANSWER CHECK (v3 harness) - catches the two failure categories of the agent-v2 eval
# ============================================================
SEND_INTENT = re.compile(r"\b(send|share|post|forward|notify|message|tell)\b.{0,60}\b(team|channel|slack|teams|"
                         r"colleague|manager|group|everyone|them|him|her)\b", re.I | re.S)
CHITCHAT = re.compile(r"^\s*(hi|hello|hey|bonjour|salut|thanks|thank you|merci|good (morning|afternoon|evening))\b"
                      r"|what can you (do|help)|who are you|how can you help|what are you", re.I)


SAVE_INTENT = re.compile(r"\b(save|export|write down|keep|store|download)\b.{0,50}\b(note|notes|file|markdown|"
                         r"summary|cheat ?sheet|study)\b|\bstudy note\b", re.I | re.S)


def wants_save(question: str) -> bool:
    return bool(SAVE_INTENT.search(question or ""))


def wants_send(question: str) -> bool:
    return bool(SEND_INTENT.search(question or ""))


def is_chitchat(question: str) -> bool:
    return bool(CHITCHAT.search(question or ""))


NUMBER = re.compile(r"(?<![\[\w])\d[\d,]*(?:\.\d+)?")
INLINE_MATH = re.compile(r"\d[\d,.]*\s*[-+*/×÷^]\s*\(?\d[\d,.]*\)?.{0,25}?=\s*\$?\d")


LIST_MARKER = re.compile(r"\(\d{1,2}\)|^\s*\d{1,2}[.)]\s|\b(step|chapter|section|page|p\.)\s*\d+(\.\d+)?", re.I | re.M)


def _numbers(text: str) -> set[float]:
    """Numbers in a text, without list markers "(1)" / "1." and chapter / section / page references."""
    out = set()
    for m in NUMBER.findall(LIST_MARKER.sub(" ", text or "")):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def answer_check(question: str, trajectory: list[dict], answer: str, done: set | None = None) -> tuple | None:
    """Return (kind, correction) for the model, or None if the answer can be accepted.
    Kinds already used (done) are skipped, so each problem is nudged at most once."""
    from app.generator import is_refusal
    done = done or set()
    if is_refusal(answer) or is_chitchat(question):
        return None
    tools = [s.get("tool") for s in trajectory if s.get("tool")]
    passages = [p for s in trajectory if isinstance(s.get("result"), dict)
                for p in s["result"].get("results", []) or []]
    checks = []
    if wants_send(question) and "send_message" not in tools:
        checks.append(("send", "CHECK: the user asked you to send / share this. Call send_message now (recipient "
                               "= who the user named, message = a 2-4 sentence summary), THEN give the final answer."))
    if wants_save(question) and "save_note" not in tools:
        checks.append(("save", "CHECK: the user asked you to save / export a note. Call save_note now (title = "
                               "short topic, content = the answer as Markdown with [n]), THEN give the final answer."))
    if not passages and "cash_flow_analysis" not in tools and not (tools and set(tools) <= set(ACTION_TOOLS)):
        checks.append(("grounding", "CHECK: your answer is not grounded in the textbook (no passage retrieved). "
                                    "Finance facts, rules and formulas must come from the book: call search_corpus "
                                    "(or lookup_term / get_section) first, then answer with [n] citations. If the "
                                    "book does not cover it, refuse."))
    if "calculator" not in tools and "cash_flow_analysis" not in tools:
        known = _numbers(question) | _numbers(json.dumps([s.get("result") for s in trajectory], default=str))
        new = {n for n in _numbers(answer) if n not in known}
        if INLINE_MATH.search(answer) or new:
            checks.append(("math", "CHECK: your answer contains numbers you computed in your head "
                                   f"({', '.join(str(n) for n in sorted(new)[:4]) or 'inline arithmetic'}). "
                                   "Call calculator for every computed number, then answer with the tool results."))
    if passages and not re.search(r"\[\d+\]", answer):
        checks.append(("citation", "CHECK: your answer uses the book but cites no passage. Add [n] after each "
                                   "sentence that uses passage n."))
    todo = [(k, m) for k, m in checks if k not in done]
    if not todo:
        return None
    return "+".join(k for k, _ in todo), "\n".join(m for _, m in todo)


# ============================================================
# PLAN-THEN-EXECUTE (optional extension)
# ============================================================
PLAN_PROMPT = """You plan the work of a finance-textbook assistant before it acts.
Tools: search_corpus(query), calculator(expression), lookup_term(term), get_section(section),
cash_flow_analysis(cash_flows, rate_percent), send_message(recipient, message),
save_note(title, content), answer.
Write the SHORTEST numbered plan (1 to 5 steps) that answers the question. One tool per step,
say which tool and why. Greetings or out-of-scope questions: a 1-step plan "answer directly".
Return JSON: {"plan": ["1. ...", "2. ..."]}"""


MULTI_PART = re.compile(r"\b(compare|difference between|versus|vs\.?|and then|both|each of|step by step)\b", re.I)


def needs_plan(question: str) -> bool:
    """Adaptive planning: plan only for multi-part questions or an action on top of an answer."""
    q = question or ""
    return (q.count("?") > 1 or bool(MULTI_PART.search(q))
            or ((wants_send(q) or wants_save(q)) and len(q.split()) > 8))


def make_plan(question: str, history: list[dict] | None = None) -> list[str]:
    from app.llm import chat
    content = chat([{"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": f"QUESTION:\n{question}"}],
                   operation="agent_plan", temperature=0, json_mode=True)
    try:
        plan = json.loads(content).get("plan") or []
        return [str(p) for p in plan][:5]
    except (json.JSONDecodeError, AttributeError):
        return []


def load_agent_prompt(version: str = AGENT_PROMPT_VERSION) -> str:
    text = (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")
    return text.replace("{refusal}", REFUSAL_TEXT).replace("{{", "{").replace("}}", "}")


# ============================================================
# ASK THE LLM WHAT TO DO
# ============================================================
def _slim_result(result, delimit: bool):
    """Tool result as given back to the model (long texts cut, <context> around book text)."""
    if not isinstance(result, dict):
        return result
    r = {k: v for k, v in result.items() if k not in ("results",)}
    if "results" in result:
        items = []
        for p in result["results"]:
            text = p["text"][:PASSAGE_CHARS]
            items.append({"n": p["n"], "page": p["page"], "section": p.get("section"),
                          "rerank": round(p["rerank"], 2) if p.get("rerank") is not None else None,
                          "text": f"<context>\n{text}\n</context>" if delimit else text})
        r["results"] = items
    return r


def _trajectory_for_prompt(trajectory: list[dict], delimit: bool) -> list[dict]:
    return [{"step": s["step"], "action": s["action"], "args": s.get("args", {}),
             "result": _slim_result(s.get("result"), delimit)} for s in trajectory]


def generate_agent_response(question: str, trajectory: list[dict], history: list[dict] | None = None,
                            prompt_version: str = AGENT_PROMPT_VERSION) -> dict:
    """Ask gpt-4o-mini for the next action (JSON). Cost is logged by app/llm.py."""
    from app.llm import chat
    system = load_agent_prompt(prompt_version)
    delimit = "<context>" in system
    past = ""
    if history:
        turns = [m for m in history if m.get("role") in ("user", "assistant")][-2 * MAX_HISTORY_TURNS:]
        past = "\n".join(f"{m['role'].upper()}: {str(m['content'])[:500]}" for m in turns)
    user_prompt = ((f"PREVIOUS CONVERSATION (last {MAX_HISTORY_TURNS} exchanges):\n{past}\n\n" if past else "")
                   + f"USER QUESTION:\n{question}\n\nTRAJECTORY SO FAR:\n"
                   + json.dumps(_trajectory_for_prompt(trajectory, delimit), indent=1, ensure_ascii=False)
                   + "\n\nChoose the next action.")
    content = chat([{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
                   operation="agent_step", temperature=0, json_mode=True)
    try:
        decision = json.loads(content)
        if not isinstance(decision, dict):
            raise ValueError("not an object")
        return decision
    except (json.JSONDecodeError, ValueError):
        return {"action": "_invalid", "raw": content[:500]}


def _normalise(decision: dict) -> tuple[str, dict, str]:
    """(tool or 'answer', args, thought) from the model's JSON, tolerant to small format slips."""
    action = str(decision.get("action", "answer")).strip().lower()
    action = ALIASES.get(action, action)
    args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
    for name in ARG_NAMES.get(action, []):                  # fields at top level also accepted
        if name not in args and decision.get(name) not in (None, ""):
            args[name] = decision[name]
    if action == "search_corpus" and "query" not in args and decision.get("expression"):
        args["query"] = decision["expression"]
    if action == "get_section" and "section" not in args and decision.get("page"):
        args["section"] = f"page {decision['page']}"
    return action, args, str(decision.get("thought") or "").strip()


def call_signature(action: str, args: dict) -> str:
    return action + json.dumps({k: str(v).strip().lower() for k, v in sorted(args.items())},
                               ensure_ascii=False)


def trace_lines(trajectory: list[dict]) -> list[str]:
    """Numbered step trace shown next to the answer: 1. search_corpus(query="...") -> 3 passages."""
    lines = []
    for s in trajectory:
        if s["action"] == "answer":
            lines.append(f"{s['step']}. answer")
            continue
        if s["action"] == "plan":
            lines.append("0. plan: " + " | ".join((s.get("result") or {}).get("plan", [])))
            continue
        if s["action"] == "check":
            lines.append(f"{s['step']}. answer rejected by the {s.get('args', {}).get('kind', '')} check -> retry")
            continue
        if s["action"] in ("stopped", "_invalid"):
            lines.append(f"{s['step']}. {s['action']} - {s.get('result', {}).get('message', '')}")
            continue
        args = ", ".join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in (s.get("args") or {}).items())
        r = s.get("result") or {}
        if r.get("blocked"):
            out = "blocked (repeated call)"
        elif r.get("status") in ("approval_required", "declined"):
            out = r["status"].replace("_", " ")
        elif not r.get("success"):
            out = "error: " + str(r.get("message", ""))[:70]
        elif "results" in r:
            out = f"{len(r['results'])} passage(s)" + (f", top rerank {r['top_rerank']:.2f}"
                                                        if r.get("top_rerank") is not None else "")
        elif "result" in r:
            out = f"= {r['result']}"
        elif "npv" in r or "irr_percent" in r:
            out = f"NPV {r.get('npv')}, IRR {r.get('irr_percent')}%, payback {r.get('payback_years')} y"
        elif "toc" in r:
            out = f"{len(r['toc'])} chapters"
        else:
            out = "ok"
        lines.append(f"{s['step']}. {s['action']}({args}) -> {out}")
    return lines


# ============================================================
# MAIN AGENT LOOP
# ============================================================
def run_agent_steps(question: str, history: list[dict] | None = None, max_steps: int = MAX_STEPS,
                    decide=None, prompt_version: str = AGENT_PROMPT_VERSION, send_mode: str | None = None,
                    injection: bool | None = None, injection_mode: str | None = None, log: bool = True,
                    plan: bool = False, planner=None, checks: bool | None = None):
    """
    Generator used by the UI. Yields
        {"type": "thinking", "step": n}
        {"type": "decided",  "step": n, "action", "thought", "decision"}
        {"type": "step",     "step": {...}}               one trajectory entry
        {"type": "done",     "result": {...}}
    decide(question, trajectory, history) can replace the LLM (offline tests).
    send_mode: "real" (stop and wait for human approval) or "fake" (auto-decline, keep going).
    plan: plan-then-execute: True (always), "auto" (only when needs_plan(question)), False.
          planner(question, history) can replace the LLM planner in tests.
    checks: answer check on/off (default: on for prompts after agent_v2 when the LLM decides).
    """
    from app import llm
    send_mode = (send_mode or SEND_MODE).lower()
    llm_decides = decide is None
    decide = decide or (lambda q, t, h: generate_agent_response(q, t, h, prompt_version))
    t_start = time.time()
    usage0 = dict(llm.USAGE)
    trajectory: list[dict] = []
    if checks is None:
        checks = llm_decides and prompt_version not in ("agent_v1", "agent_v2")
    ctx = {"question": question, "prompt_version": prompt_version, "send_mode": send_mode,
           "injection": bool(injection), "t_start": t_start, "usage0": usage0, "log": log,
           "decisions": 0, "plan": plan if plan == "auto" else bool(plan), "checks": bool(checks), "nudges": 0, "nudge_kinds": set()}

    if llm_decides and not has_api_key():
        yield {"type": "done", "result": finish(ctx, "The agent needs the OpenAI key (it is driven by "
                                                     "gpt-4o-mini). Add OPENAI_API_KEY to .env, or use the "
                                                     "RAG Chat tab (works offline).", trajectory, "no_api_key")}
        return

    if plan == "auto":
        plan = needs_plan(question)
        ctx["plan"] = "auto:" + ("yes" if plan else "no")
    if plan:
        yield {"type": "thinking", "step": 0}
        t0 = time.time()
        steps_plan = (planner or make_plan)(question, history)
        ctx["decisions"] += 1
        entry = {"step": 0, "action": "plan", "tool": None, "args": {},
                 "thought": "Plan first, then follow it step by step.",
                 "result": {"success": bool(steps_plan), "plan": steps_plan},
                 "seconds": round(time.time() - t0, 2)}
        trajectory.append(entry)
        yield {"type": "decided", "step": 0, "action": "plan", "thought": entry["thought"], "decision": {}}
        yield {"type": "step", "step": entry}

    next_n, seen, repeats, failed_searches, tool_errors = 1, set(), 0, 0, 0
    for step in range(1, max_steps + 1):
        t0 = time.time()
        yield {"type": "thinking", "step": step}
        decision = decide(question, trajectory, history)
        ctx["decisions"] += 1
        action, args, thought = _normalise(decision)
        yield {"type": "decided", "step": step, "action": action, "thought": thought, "decision": decision}

        # ---------- final answer
        if action == "answer":
            answer = str(decision.get("answer") or decision.get("message") or "").strip() or REFUSAL_TEXT
            problem = answer_check(question, trajectory, answer, ctx["nudge_kinds"]) if ctx["checks"] else None
            if problem and ctx["nudges"] < MAX_NUDGES and step < max_steps:
                ctx["nudges"] += 1
                ctx["nudge_kinds"].update(problem[0].split("+"))
                entry = {"step": step, "action": "check", "tool": None, "args": {"kind": problem[0]},
                         "thought": thought, "rejected_answer": answer,
                         "result": {"success": False, "message": problem[1]},
                         "seconds": round(time.time() - t0, 2)}
                trajectory.append(entry)
                yield {"type": "step", "step": entry}
                continue
            entry = {"step": step, "action": "answer", "args": {}, "result": answer, "thought": thought,
                     "seconds": round(time.time() - t0, 2)}
            trajectory.append(entry)
            yield {"type": "step", "step": entry}
            yield {"type": "done", "result": finish(ctx, answer, trajectory, "answered")}
            return

        entry = {"step": step, "action": action, "tool": action if action in TOOLS or action in ACTION_TOOLS
                 else None, "args": args, "thought": thought}
        # UI-compatible fields
        for k in ("query", "expression", "recipient", "message", "term", "section", "title"):
            if k in args:
                entry[k] = args[k]

        # ---------- invalid output / unknown tool: tell the model, count as an error
        if action not in TOOLS and action not in ACTION_TOOLS:
            result = {"success": False, "message": f"Unknown action '{action}'. Use one of: "
                      + ", ".join(list(TOOLS) + list(ACTION_TOOLS) + ["answer"]) + "."}
            if action == "_invalid":
                result["message"] = "Your output was not valid JSON. Return ONLY the JSON object."
            tool_errors += 1

        # ---------- repeated identical call: blocked, not executed
        elif call_signature(action, args) in seen:
            repeats += 1
            result = {"success": False, "blocked": True,
                      "message": "Repeated call blocked: you already called this tool with the same "
                                 "arguments. Use the earlier result, try different arguments, or answer."}

        # ---------- action allowlist: only the USER'S words can unlock an action (v3 harness)
        elif ctx["checks"] and ((action == "send_message" and not wants_send(question))
                                or (action == "save_note" and not wants_save(question))):
            seen.add(call_signature(action, args))
            result = {"success": False, "blocked": True, "status": "not_requested",
                      "message": f"Blocked: the user did not ask for {action}. Only the user's question can "
                                 "request an action; answer the question instead."}

        # ---------- action tool (human in the loop)
        elif action == "send_message":
            seen.add(call_signature(action, args))
            recipient, message = str(args.get("recipient", "")), str(args.get("message", ""))
            if send_mode == "fake":
                result = fake_send(recipient, message, reason=f"agent eval / test: {question[:80]}")
            else:
                result = prepare_message(recipient, message)
                entry.update(result=result, seconds=round(time.time() - t0, 2))
                trajectory.append(entry)
                yield {"type": "step", "step": entry}
                answer = ("I prepared the following message for you:\n\n"
                          f"**Recipient:** {recipient} (channel: {result['channel']})\n\n**Message:** {message}\n\n"
                          "Nothing has been sent: a human must approve it first.")
                yield {"type": "done", "result": finish(ctx, answer, trajectory, "approval_required",
                                                        pending_message={"recipient": recipient,
                                                                         "message": message})}
                return

        elif action == "save_note":
            seen.add(call_signature(action, args))
            title, content = str(args.get("title", "") or "Study note"), str(args.get("content", ""))
            if send_mode == "fake":
                result = fake_save(title, content, reason=f"agent eval / test: {question[:80]}")
            else:
                passages = [p for s in trajectory if isinstance(s.get("result"), dict)
                            for p in s["result"].get("results", []) or []]
                result = prepare_note(title, content, passages)
                entry.update(result=result, seconds=round(time.time() - t0, 2))
                trajectory.append(entry)
                yield {"type": "step", "step": entry}
                answer = (f"I prepared a study note **{title}** (preview below). Nothing has been saved: "
                          "a human must approve it first.")
                yield {"type": "done", "result": finish(ctx, answer, trajectory, "approval_required",
                                                        pending_message={"kind": "note", "recipient": "note:" + title,
                                                                         "message": result["preview"],
                                                                         "path": result["path"]})}
                return

        # ---------- the corpus / finance tools
        else:
            seen.add(call_signature(action, args))
            fn = TOOLS[action]
            try:
                if action == "search_corpus":
                    result = fn(args.get("query") or question, first_n=next_n, injection=injection,
                                injection_mode=injection_mode)
                elif action == "lookup_term":
                    result = fn(args.get("term", ""), first_n=next_n)
                elif action == "get_section":
                    result = fn(args.get("section", ""), first_n=next_n)
                elif action == "cash_flow_analysis":
                    result = fn(args.get("cash_flows", []), args.get("rate_percent"))
                else:
                    result = fn(args.get("expression", ""))
            except TypeError as exc:                        # wrong / missing arguments
                result = {"success": False, "message": f"Bad arguments for {action}: {exc}. "
                                                      f"Expected: {', '.join(ARG_NAMES[action])}."}
            next_n += len(result.get("results", []) or [])
            if action == "search_corpus" and result.get("message") == NO_MATCH:
                failed_searches += 1
            elif not result.get("success"):
                tool_errors += 1

        entry.update(result=result, seconds=round(time.time() - t0, 2))
        trajectory.append(entry)
        yield {"type": "step", "step": entry}

        # ---------- recovery limits
        stop = None
        if failed_searches > MAX_SEARCH_RETRIES:
            stop = ("retry_limit", REFUSAL_TEXT)
        elif repeats >= 2:
            stop = ("repeated_calls", "I stopped because the same tool call was repeated. "
                                      + _best_effort(trajectory))
        elif tool_errors >= MAX_TOOL_ERRORS:
            stop = ("tool_errors", "I stopped after several tool errors. " + _best_effort(trajectory))
        if stop:
            trajectory.append({"step": step, "action": "stopped", "args": {},
                               "result": {"message": stop[0]}})
            yield {"type": "done", "result": finish(ctx, stop[1], trajectory, stop[0])}
            return

    trajectory.append({"step": max_steps, "action": "stopped", "args": {}, "result": {"message": "max_steps"}})
    yield {"type": "done", "result": finish(
        ctx, f"I could not complete the request within the maximum of {max_steps} agent steps. "
             + _best_effort(trajectory), trajectory, "max_steps")}


def _best_effort(trajectory: list[dict]) -> str:
    found = [p for s in trajectory if isinstance(s.get("result"), dict) for p in s["result"].get("results", [])]
    if found:
        return f"The most relevant passage I found is on page {found[0]['page']} [{found[0]['n']}]."
    return ""


def finish(ctx: dict, answer: str, trajectory: list[dict], stop_reason: str,
           pending_message: dict | None = None) -> dict:
    """Final result (citations in the same format as the RAG chat) + trace log line."""
    from app import llm
    from app.generator import is_refusal, parse_citations, unverified_citations
    passages = [p for s in trajectory if isinstance(s.get("result"), dict)
                for p in s["result"].get("results", []) or []]
    refused = is_refusal(answer)
    cited = [] if refused else parse_citations(answer, len(passages))
    citations = [{**p, "cited": p["n"] in cited} for p in passages]
    evidence = {}
    if not refused and citations:
        from app.evidence import find_evidence
        evidence = find_evidence(answer, citations)
    tools_used = [s["tool"] for s in trajectory if s.get("tool")]
    usage = {k: llm.USAGE[k] - ctx["usage0"].get(k, 0) for k in llm.USAGE}
    signatures = [call_signature(s["action"], s.get("args", {})) for s in trajectory if s.get("tool")]
    result = {
        "question": ctx["question"],
        "standalone_question": ctx["question"],
        "answer": answer,
        "trajectory": trajectory,
        "trace": trace_lines(trajectory),
        "stop_reason": stop_reason,
        "approval_required": stop_reason == "approval_required",
        "pending_message": pending_message or {},
        "send_proposals": [{"recipient": s["args"].get("recipient"), "message": s["args"].get("message"),
                            "status": (s.get("result") or {}).get("status")}
                           for s in trajectory if s.get("tool") == "send_message"],
        "save_proposals": [{"title": s["args"].get("title"), "status": (s.get("result") or {}).get("status")}
                           for s in trajectory if s.get("tool") == "save_note"],
        "refused": refused,
        "citations": citations,
        "evidence": evidence,
        "cited_order": cited,
        "unverified_citations": unverified_citations(answer, len(passages)) if not refused else [],
        "quotes": [],
        "tools_used": tools_used,
        "steps": len([s for s in trajectory if s["action"] not in ("stopped", "plan")]),
        "plan": next((s["result"].get("plan") for s in trajectory if s["action"] == "plan"), None),
        "nudges": ctx.get("nudges", 0),
        "repeated_calls": len(signatures) - len(set(signatures)),
        "hit_step_cap": stop_reason == "max_steps",
        "latency_s": round(time.time() - ctx["t_start"], 2),
        "model_calls": ctx["decisions"] if usage["calls"] == 0 else usage["calls"],
        "tokens": {"prompt": usage["prompt_tokens"], "completion": usage["completion_tokens"]},
        "prompt_version": ctx["prompt_version"],
        "injection_chunk_retrieved": any(p.get("injection") for p in passages),
        "route": "agent",
        "settings": {"mode": "hybrid", "send_mode": ctx["send_mode"], "injection": ctx["injection"],
                     "plan": ctx.get("plan", False), "checks": ctx.get("checks", False)},
    }
    if ctx.get("log"):
        log_trace(result)
    return result


def log_trace(result: dict) -> None:
    """One JSON line per agent run in data/agent_traces.jsonl (the 'trace logged to a file')."""
    def slim(s):
        r = s.get("result")
        if isinstance(r, dict) and "results" in r:
            r = {**{k: v for k, v in r.items() if k != "results"},
                 "results": [{"n": p["n"], "page": p["page"], "section": p.get("section"),
                              "rerank": p.get("rerank"), "text": p["text"][:200]} for p in r["results"]]}
        return {**{k: v for k, v in s.items() if k != "result"}, "result": r}
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              **{k: result[k] for k in ("question", "answer", "stop_reason", "steps", "tools_used",
                                        "repeated_calls", "latency_s", "model_calls", "tokens",
                                        "prompt_version", "trace", "send_proposals", "save_proposals",
                                        "nudges", "route")},
              "trajectory": [slim(s) for s in result["trajectory"]]}
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


@tracing.observe(name="agent_run")
def run_agent(question: str, history: list[dict] | None = None, **kwargs) -> dict:
    """Run the whole agent and return the final result."""
    result = None
    for event in run_agent_steps(question, history, **kwargs):
        if event["type"] == "done":
            result = event["result"]
    return result


# ============================================================
# COMMAND LINE TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60 + "\nPERSONAL FINANCE AGENT\n" + "=" * 60)
    q = input("Enter your question: ")
    r = run_agent(q)
    print("\n".join(r["trace"]))
    print("\n" + "=" * 60 + "\nANSWER\n" + "=" * 60 + "\n" + r["answer"])
    for c in r["citations"]:
        if c["cited"]:
            print(f"  [{c['n']}] {source_title(c['source'])} p.{c['page']}")
    print(f"\n{r['steps']} steps, {r['model_calls']} model calls, {r['latency_s']} s, stop: {r['stop_reason']}")
    tracing.flush()
