"""
agent_eval.py - Evaluate the AGENT (Agentic AI project, day 2, mandatory).

Questions: every item of data/questions.json with "category": "agent" (WRITTEN BY THE TEAM,
no GenAI - course rule; format in scripts/templates/agent_questions_TEMPLATE.json) plus the
priority questions of the RAG set (so the agent and the RAG are compared on the same items).

    {"question": "...", "category": "agent", "sub_type": "search_then_calculate",
     "expected_answer": "...", "expected_tools": ["search_corpus", "calculator"]}

Runs with fake send (every send_message proposal is declined and logged, nothing is sent)
and without the injection chunk. Measures, per question:

  outcome      correct (null = mark it by hand in the CSV, or --judge for an LLM verdict),
               refused, citations, unverified citations
  trajectory   tools used vs expected_tools (tool recall, extra tools, exact match),
               steps, repeated calls, step-cap hits, stop reason, clean out-of-scope stop,
               send_message proposals
  cost         latency, model calls, tokens

Outputs: data/agent_eval_results.json, data/agent_eval_marking.csv (column "correct" to fill in),
data/agent_eval_history/<tag>.json (regression check), README sections AGENT_EVAL and
AGENT_VS_RAG (with --compare-rag).

    python scripts/agent_eval.py --tag agent-v2                 # agent only (~6 model calls/question)
    python scripts/agent_eval.py --tag agent-v2 --judge         # + LLM judge
    python scripts/agent_eval.py --tag agent-v2 --compare-rag   # + the RAG pipeline on the same questions
    python scripts/agent_eval.py --marks data/agent_eval_marking.csv   # re-score with your manual marks ($0)
    python scripts/agent_eval.py --regression agent-v1 agent-v2        # what changed between two runs ($0)

Extensions (each one is a row of the Agent vs RAG table, compare with --regression):
    python scripts/agent_eval.py --tag agent-v3-plan --plan           # plan-then-execute
    python scripts/agent_eval.py --tag router-v1 --router             # router: simple lookups -> RAG
    python scripts/agent_eval.py --tag agent-v3-x3 --repeat 3         # reliability (flakiness), 3x the cost
    python scripts/agent_eval.py --extensions                         # README table of all saved runs ($0)
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import COST_LOG_FILE, DATA_DIR, QUESTIONS_FILE, has_api_key  # noqa: E402
from readme_tools import update_section  # noqa: E402

RESULTS = DATA_DIR / "agent_eval_results.json"
MARKING = DATA_DIR / "agent_eval_marking.csv"
HISTORY = DATA_DIR / "agent_eval_history"


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def load_questions(path: Path, include_priority: bool = True) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for q in items:
        if "WRITE YOURSELF" in q.get("question", ""):
            continue
        if q.get("category") == "agent":
            out.append(q)
        elif include_priority and q.get("priority"):
            out.append({**q, "sub_type": q.get("sub_type") or f"rag_priority_{q.get('category')}",
                        "expected_tools": q.get("expected_tools") or
                        ([] if q.get("category") == "out-of-scope" else ["search_corpus"])})
    return out


def _cost_log() -> list[dict]:
    try:
        return json.loads(COST_LOG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _calls_since(n0: int) -> tuple[int, float]:
    new = [e for e in _cost_log()[n0:] if e.get("operation") != "judge"]
    chat_calls = [e for e in new if "gpt" in e.get("model", "")]
    return len(chat_calls), round(sum(float(e.get("cost_usd", 0) or 0) for e in new), 6)


# ---------------------------------------------------------------------------
# One question
# ---------------------------------------------------------------------------
def is_out_of_scope(q: dict) -> bool:
    return q.get("category") == "out-of-scope" or q.get("sub_type") in ("out_of_scope", "no_tool_refusal")


def evaluate_agent(q: dict, args) -> dict:
    from app.agent import run_agent
    n0 = len(_cost_log())
    t0 = time.time()
    kwargs = dict(send_mode="fake", prompt_version=args.prompt, injection=False,
                  plan="auto" if getattr(args, "plan_auto", False) else getattr(args, "plan", False))
    if getattr(args, "max_steps", None):
        kwargs["max_steps"] = args.max_steps
    if getattr(args, "no_checks", False):
        kwargs["checks"] = False
    if getattr(args, "router", False):
        from app.router import run_routed
        r = run_routed(q["question"], q.get("history"), **kwargs)
    else:
        r = run_agent(q["question"], history=q.get("history"), **kwargs)
    latency = round(time.time() - t0, 2)
    calls, cost = _calls_since(n0)
    used = list(dict.fromkeys(r["tools_used"]))
    expected = list(q.get("expected_tools") or [])
    hit = [t for t in expected if t in used]
    oos = is_out_of_scope(q)
    row = {
        "question": q["question"], "category": q.get("category"), "sub_type": q.get("sub_type"),
        "expected_answer": q.get("expected_answer"), "answer": r["answer"],
        "expected_tools": expected, "tools_used": used, "tool_sequence": r["tools_used"],
        "tool_recall": (len(hit) / len(expected)) if expected else (1.0 if not used else 0.0),
        "extra_tools": [t for t in used if t not in expected],
        "tools_exact": set(used) == set(expected),
        "steps": r["steps"], "repeated_calls": r["repeated_calls"], "hit_step_cap": r["hit_step_cap"],
        "stop_reason": r["stop_reason"], "refused": r["refused"],
        "clean_oos_stop": (r["refused"] and r["steps"] <= 3 and not r["hit_step_cap"]) if oos else None,
        "send_proposals": len(r["send_proposals"]),
        "cited": [c["n"] for c in r["citations"] if c["cited"]],
        "unverified_citations": r["unverified_citations"],
        "latency_s": latency, "model_calls": calls or r["model_calls"], "cost_usd": cost,
        "tokens": r["tokens"], "trace": r["trace"], "correct": None, "judge_reason": None,
        "route": r.get("route", "agent"), "nudges": r.get("nudges", 0), "plan": r.get("plan"),
    }
    if args.judge:
        from app.generator import judge_answer
        passages = [c for c in r["citations"]]
        j = judge_answer(q["question"], q.get("expected_answer"), r["answer"], passages,
                         "out-of-scope" if oos else (q.get("category") or "agent"))
        row["correct"], row["judge_reason"] = j.get("correct"), j.get("reason")
    return row


def evaluate_rag(q: dict, judge: bool) -> dict:
    from app import rag
    n0 = len(_cost_log())
    t0 = time.time()
    res = rag.answer(q["question"], history=q.get("history"),
                     settings={"use_cache": False, "followups": False})
    latency = round(time.time() - t0, 2)
    calls, cost = _calls_since(n0)
    row = {"answer": res["answer"], "latency_s": latency, "model_calls": calls, "cost_usd": cost,
           "refused": res.get("refused"), "correct": None}
    if judge:
        from app.generator import judge_answer
        cat = "out-of-scope" if is_out_of_scope(q) else (q.get("category") or "agent")
        j = judge_answer(q["question"], q.get("expected_answer"), res["answer"], res.get("citations", []), cat)
        row["correct"] = j.get("correct")
    return row


# ---------------------------------------------------------------------------
# Summary + tables
# ---------------------------------------------------------------------------
def _avg(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _median(vals):
    vals = sorted(float(v) for v in vals if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def summarise(rows: list[dict]) -> dict:
    s = {"n": len(rows),
         "correct": _avg(r["correct"] for r in rows),
         "tool_recall": _avg(r["tool_recall"] for r in rows),
         "tools_exact": _avg(r["tools_exact"] for r in rows),
         "avg_steps": _avg(r["steps"] for r in rows),
         "repeated_call_rate": _avg(r["repeated_calls"] > 0 for r in rows),
         "step_cap_rate": _avg(r["hit_step_cap"] for r in rows),
         "clean_oos_stop": _avg(r["clean_oos_stop"] for r in rows),
         "avg_latency_s": _avg(r["latency_s"] for r in rows),
         "median_latency_s": _median(r["latency_s"] for r in rows),
         "avg_model_calls": _avg(r["model_calls"] for r in rows),
         "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 5)}
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("sub_type") or r.get("category") or "?"].append(r)
    s["by_sub_type"] = {k: {"n": len(v), "correct": _avg(x["correct"] for x in v),
                            "tool_recall": _avg(x["tool_recall"] for x in v),
                            "avg_steps": _avg(x["steps"] for x in v)} for k, v in sorted(groups.items())}
    return s


def pct(v):
    return "-" if v is None else f"{v:.0%}"


def num(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def eval_markdown(s: dict, tag: str) -> str:
    lines = [f"### Agent evaluation (`{tag}`, {s['n']} questions, fake send)", "",
             "| Correct | Tool recall | Exact tool set | Avg steps | Repeated calls | Step-cap hits | "
             "Clean out-of-scope stop | Latency avg / median (s) | Model calls | Cost |",
             "|---|---|---|---|---|---|---|---|---|---|",
             f"| {pct(s['correct'])} | {pct(s['tool_recall'])} | {pct(s['tools_exact'])} | {num(s['avg_steps'])} | "
             f"{pct(s['repeated_call_rate'])} | {pct(s['step_cap_rate'])} | {pct(s['clean_oos_stop'])} | "
             f"{num(s['avg_latency_s'])} / {num(s.get('median_latency_s'))} | {num(s['avg_model_calls'])} | ${s['total_cost_usd']:.4f} |", "",
             "| Sub-type | n | Correct | Tool recall | Avg steps |", "|---|---|---|---|---|"]
    for k, v in s["by_sub_type"].items():
        lines.append(f"| {k} | {v['n']} | {pct(v['correct'])} | {pct(v['tool_recall'])} | {num(v['avg_steps'])} |")
    lines += ["", "Correct = manual marks (`data/agent_eval_marking.csv`) or LLM judge (`--judge`); "
              "\"-\" = not marked yet. Generated by `python scripts/agent_eval.py`."]
    return "\n".join(lines)


def compare_markdown(rows: list[dict]) -> str:
    both = [r for r in rows if r.get("rag")]
    a = {"correct": _avg(r["correct"] for r in both), "calls": _avg(r["model_calls"] for r in both),
         "lat": _avg(r["latency_s"] for r in both), "cost": sum(r.get("cost_usd") or 0 for r in both)}
    g = {"correct": _avg(r["rag"]["correct"] for r in both), "calls": _avg(r["rag"]["model_calls"] for r in both),
         "lat": _avg(r["rag"]["latency_s"] for r in both), "cost": sum(r["rag"].get("cost_usd") or 0 for r in both)}
    lines = [f"### Agent vs RAG (same {len(both)} questions)", "",
             "| System | Correct | Model calls / question | Latency / question (s) | Total cost |",
             "|---|---|---|---|---|",
             f"| RAG pipeline (Chat tab) | {pct(g['correct'])} | {num(g['calls'])} | {num(g['lat'])} | ${g['cost']:.4f} |",
             f"| Agent (tools, max 6 steps) | {pct(a['correct'])} | {num(a['calls'])} | {num(a['lat'])} | ${a['cost']:.4f} |",
             "", "| Question | Sub-type | RAG correct | Agent correct | Agent tools |", "|---|---|---|---|---|"]
    for r in both:
        yn = {True: "yes", False: "**no**", None: "-"}
        lines.append(f"| {r['question'][:70]} | {r.get('sub_type')} | {yn[r['rag']['correct']]} | "
                     f"{yn[r['correct']]} | {', '.join(r['tools_used']) or 'none'} |")
    lines += ["", "When is the agent worth it? Look at the sub-types where it is more correct (calculations, "
              "glossary, sections) versus the extra model calls and latency it costs on simple lookups."]
    return "\n".join(lines)


def write_marking_csv(rows: list[dict]) -> None:
    with open(MARKING, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "sub_type", "expected_answer", "agent_answer", "tools_used", "correct"])
        for r in rows:
            c = "" if r["correct"] is None else ("1" if r["correct"] else "0")
            w.writerow([r["question"], r.get("sub_type"), r.get("expected_answer") or "", r["answer"],
                        " > ".join(r["tool_sequence"]), c])
    print(f"Marking sheet: {MARKING}  (fill the 'correct' column with 1/0, then run --marks)")


def apply_marks(path: str, tag: str | None = None) -> None:
    """Marks are matched by question text, so one marked CSV can be applied to several runs (--tag)."""
    data = json.loads(((HISTORY / f"{tag}.json") if tag else RESULTS).read_text(encoding="utf-8"))
    marks = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("correct", "").strip() in ("0", "1"):
                marks[row["question"]] = row["correct"].strip() == "1"
    for r in data["results"]:
        if r["question"] in marks:
            r["correct"] = marks[r["question"]]
    data["summary"] = summarise(data["results"])
    if not tag:
        RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    (HISTORY / f"{data['run']['tag']}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    print(f"{len(marks)} manual marks applied to run '{data['run']['tag']}'.")
    if not marks:
        print("No 1/0 found in the 'correct' column - open the CSV, fill it, save it, run again.")
    update_section("AGENT_EVAL", eval_markdown(data["summary"], data["run"]["tag"]))
    if any(r.get("rag") for r in data["results"]):
        update_section("AGENT_VS_RAG", compare_markdown(data["results"]))


def regression(tag_a: str, tag_b: str) -> None:
    a = json.loads((HISTORY / f"{tag_a}.json").read_text(encoding="utf-8"))
    b = json.loads((HISTORY / f"{tag_b}.json").read_text(encoding="utf-8"))
    pa = {r["question"]: r for r in a["results"]}
    print(f"{'metric':<22}{tag_a:>14}{tag_b:>14}")
    for k in ("correct", "tool_recall", "tools_exact", "avg_steps", "repeated_call_rate", "step_cap_rate",
              "clean_oos_stop", "avg_latency_s", "avg_model_calls"):
        va, vb = a["summary"].get(k), b["summary"].get(k)
        print(f"{k:<22}{num(va, 2):>14}{num(vb, 2):>14}")
    print()
    for r in b["results"]:
        old = pa.get(r["question"])
        if not old:
            continue
        changes = []
        if old.get("correct") is not None and r.get("correct") is not None and old["correct"] != r["correct"]:
            changes.append("FIXED" if r["correct"] else "REGRESSION")
        if old["tools_used"] != r["tools_used"]:
            changes.append(f"tools {old['tools_used']} -> {r['tools_used']}")
        if old["steps"] != r["steps"]:
            changes.append(f"steps {old['steps']} -> {r['steps']}")
        if changes:
            print(f"- {r['question'][:70]}: " + "; ".join(changes))


def passed(r: dict) -> bool:
    """A run passes if it is marked/judged correct; without a mark, if every expected tool was used."""
    return bool(r["correct"]) if r.get("correct") is not None else r["tool_recall"] >= 1.0


def reliability(runs: list[list[dict]]) -> dict:
    """runs[k][i] = run k of question i. Pass rate vs share of questions that passed EVERY run."""
    n_q, n_r = len(runs[0]), len(runs)
    per_q = [[runs[k][i] for k in range(n_r)] for i in range(n_q)]
    return {"runs": n_r, "questions": n_q,
            "pass_rate": _avg(passed(r) for rs in per_q for r in rs),
            "all_runs_pass": _avg(all(passed(r) for r in rs) for rs in per_q),
            "same_tools_every_run": _avg(len({tuple(r["tool_sequence"]) for r in rs}) == 1 for rs in per_q),
            "judged": any(r.get("correct") is not None for rs in per_q for r in rs),
            "flaky": [rs[0]["question"] for rs in per_q
                      if len({passed(r) for r in rs}) > 1 or len({tuple(r["tool_sequence"]) for r in rs}) > 1]}


def reliability_markdown(rel: dict, tag: str) -> str:
    basis = "correct (judge / marks)" if rel["judged"] else "all expected tools used (no correctness marks)"
    lines = [f"### Reliability (`{tag}`, {rel['questions']} questions x {rel['runs']} runs, temperature 0)", "",
             "| Pass rate (all runs) | Passed every run | Same tool sequence every run |", "|---|---|---|",
             f"| {pct(rel['pass_rate'])} | {pct(rel['all_runs_pass'])} | {pct(rel['same_tools_every_run'])} |", "",
             f"Pass = {basis}. The gap between the first two numbers measures flakiness."]
    if rel["flaky"]:
        lines += ["", "Flaky questions (result or path changed between runs):"]
        lines += [f"- {q[:90]}" for q in rel["flaky"]]
    return "\n".join(lines)


def extensions_markdown() -> str:
    """One row per saved run (baseline, prompt versions, plan, router) - the 'going further' table."""
    lines = ["### Agent variants (one row per saved eval run)", "",
             "| Run | Prompt | Plan | Router | Cap | Check | Correct | Tool recall | Exact tools | Avg steps | "
             "Model calls | Median latency (s) | Cost |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(HISTORY.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if f.stem.endswith("-priority"):                 # regression gates, not variants
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        run, s = d.get("run", {}), d.get("summary", {})
        lines.append(f"| {run.get('tag', f.stem)} | {run.get('prompt', '-')} | {('auto' if run.get('plan') == 'auto' else 'yes') if run.get('plan') else '-'} | "
                     f"{'yes' if run.get('router') else '-'} | {run.get('max_steps', 6)} | "
                     f"{'off' if run.get('checks') is False else ('-' if run.get('prompt') == 'agent_v2' else 'on')} | "
                     f"{pct(s.get('correct'))} | {pct(s.get('tool_recall'))} | "
                     f"{pct(s.get('tools_exact'))} | {num(s.get('avg_steps'))} | {num(s.get('avg_model_calls'))} | "
                     f"{num(s.get('median_latency_s') or _median(r['latency_s'] for r in d.get('results', [])))} | "
                     f"${s.get('total_cost_usd') or 0:.4f} |")
    lines += ["", "Generated by `python scripts/agent_eval.py --extensions` from `data/agent_eval_history/`."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_FILE))
    ap.add_argument("--tag", default="agent")
    ap.add_argument("--prompt", default="agent_v3")
    ap.add_argument("--plan", action="store_true", help="plan-then-execute extension")
    ap.add_argument("--max-steps", type=int, help="harness tuning: step cap (default 6)")
    ap.add_argument("--no-checks", action="store_true", help="harness tuning: answer check off")
    ap.add_argument("--plan-auto", action="store_true", help="adaptive planning: plan only multi-part questions")
    ap.add_argument("--router", action="store_true", help="router extension (simple lookups -> RAG)")
    ap.add_argument("--repeat", type=int, default=1, help="run every question N times (reliability)")
    ap.add_argument("--extensions", action="store_true", help="README table of all saved runs ($0)")
    ap.add_argument("--judge", action="store_true", help="LLM judge for 'correct' (else mark by hand)")
    ap.add_argument("--compare-rag", action="store_true", help="also run the RAG pipeline (Agent vs RAG table)")
    ap.add_argument("--no-priority", action="store_true", help="only the category 'agent' questions")
    ap.add_argument("--ids", help="comma-separated indexes of the questions to run (e.g. 0,3,5)")
    ap.add_argument("--marks", help="apply a filled marking CSV to the last run ($0)")
    ap.add_argument("--regression", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()
    HISTORY.mkdir(parents=True, exist_ok=True)

    if args.marks:
        return apply_marks(args.marks, args.tag if args.tag != "agent" else None)
    if args.regression:
        return regression(*args.regression)
    if args.extensions:
        md = extensions_markdown()
        print(md)
        return update_section("AGENT_VARIANTS", md)
    if not has_api_key():
        sys.exit("Needs OPENAI_API_KEY (the agent is driven by gpt-4o-mini).")

    questions = load_questions(Path(args.questions), include_priority=not args.no_priority)
    n_total_agent = sum(q.get("category") == "agent" for q in questions)
    if args.ids:
        keep = {int(i) for i in args.ids.split(",")}
        questions = [q for i, q in enumerate(questions) if i in keep]
    n_agent = sum(q.get("category") == "agent" for q in questions)
    if n_total_agent == 0:
        print("WARNING: no question with category 'agent' yet. Write them yourselves (no GenAI) in "
              "data/questions.json - see scripts/templates/agent_questions_TEMPLATE.json. "
              "Running the priority questions only.")
    mode = " + ".join(m for m, on in (("plan-then-execute", args.plan), ("adaptive planning", args.plan_auto),
                                      ("router", args.router)) if on)
    print(f"{len(questions)} questions ({n_agent} agent questions), prompt {args.prompt}"
          + (f", {mode}" if mode else "") + (f", {args.repeat} runs each" if args.repeat > 1 else "")
          + ". Fake send ON, injection OFF.\n")

    try:                                                 # warm-up: load the reranker before timing
        from app.agent_tools import search_corpus
        search_corpus("net present value")
    except Exception:                                    # noqa: BLE001
        pass

    all_runs = []
    for k in range(max(1, args.repeat)):
        if args.repeat > 1:
            print(f"--- run {k + 1}/{args.repeat}")
        rows = []
        for i, q in enumerate(questions):
            row = evaluate_agent(q, args)
            if args.compare_rag and k == 0:
                row["rag"] = evaluate_rag(q, args.judge)
            rows.append(row)
            mark = {True: "OK ", False: "BAD", None: "   "}[row["correct"]]
            route = "RAG " if row.get("route") == "rag" else ""
            print(f"[{i:>2}] {mark} {route}tools={','.join(row['tool_sequence']) or '-':<45} steps={row['steps']} "
                  f"{row['latency_s']:>5.1f}s  {q['question'][:60]}")
        all_runs.append(rows)
    rows = all_runs[0]

    summary = summarise(rows)
    run = {"tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "prompt": args.prompt, "judge": args.judge, "n": len(rows),
           "plan": "auto" if args.plan_auto else args.plan, "router": args.router, "repeat": args.repeat,
           "max_steps": args.max_steps or 6, "checks": not args.no_checks}
    data = {"run": run, "summary": summary, "results": rows}
    if args.repeat > 1:
        data["reliability"] = reliability(all_runs)
        data["other_runs"] = all_runs[1:]
        rel_md = reliability_markdown(data["reliability"], args.tag)
        print("\n" + rel_md)
        update_section("AGENT_RELIABILITY", rel_md)
    RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    (HISTORY / f"{args.tag}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    write_marking_csv(rows)
    md = eval_markdown(summary, args.tag)
    print("\n" + md)
    update_section("AGENT_EVAL", md)
    if args.compare_rag:
        cmp_md = compare_markdown(rows)
        print("\n" + cmp_md)
        update_section("AGENT_VS_RAG", cmp_md)
    update_section("AGENT_VARIANTS", extensions_markdown())
    from app import tracing
    tracing.flush()


if __name__ == "__main__":
    main()
