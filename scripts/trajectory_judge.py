"""
trajectory_judge.py - LLM-as-judge for the whole TRAJECTORY, validated against hand labels
(Agentic AI project, "after the week: production-ready").

The answer judge checks the final answer; this one asks "was this a sensible sequence of
actions?" for the question, looking at the numbered step trace. The judge may answer "unclear"
instead of inventing a verdict. Then WE label the same trajectories by hand and measure how
often the judge agrees with us - an unvalidated judge is not a metric.

    python scripts/trajectory_judge.py --tag agent-v3-final          # judge a saved eval run (~$0.002)
        -> data/trajectory_judge_<tag>.csv  (fill the column human_label: sensible / not_sensible)
    python scripts/trajectory_judge.py --agreement data/trajectory_judge_agent-v3-final.csv   # $0
        -> agreement %, confusion table, README section TRAJECTORY_JUDGE
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import DATA_DIR, has_api_key  # noqa: E402
from readme_tools import update_section  # noqa: E402

HISTORY = DATA_DIR / "agent_eval_history"
VERDICTS = ("sensible", "not_sensible", "unclear")

JUDGE_PROMPT = """You review the TRAJECTORY of a tool-using assistant for a personal-finance textbook.
Tools: search_corpus (search the book), calculator, lookup_term (book glossary), get_section
(chapter / section / page), cash_flow_analysis (NPV / IRR / payback), send_message and save_note
(actions that need human approval; fake mode declines them), answer.

Judge ONLY the sequence of actions, not the style of the answer:
- sensible: every step is useful for the question, in a reasonable order; numbers come from
  calculator / cash_flow_analysis; book facts come from a book tool; actions only when asked;
  out-of-scope or greeting questions stop without searching again and again.
- not_sensible: a needed tool is missing (e.g. math done in the head, no search for book facts,
  requested send/save never proposed), a wrong tool, useless or repeated steps, an action nobody asked for.
- unclear: you cannot tell from the trace. Use it instead of guessing.
A different path than the one you would take can still be sensible.

Return JSON: {"verdict": "sensible" | "not_sensible" | "unclear", "reason": "one sentence"}"""


def judge(question: str, trace: list[str], answer: str) -> dict:
    from app.llm import chat
    user = (f"QUESTION:\n{question}\n\nTRAJECTORY:\n" + "\n".join(trace)
            + f"\n\nFINAL ANSWER (for context only):\n{answer[:600]}")
    out = chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
               operation="judge", temperature=0, json_mode=True)
    try:
        d = json.loads(out)
        v = str(d.get("verdict", "unclear")).strip().lower()
        return {"verdict": v if v in VERDICTS else "unclear", "reason": str(d.get("reason", ""))[:300]}
    except json.JSONDecodeError:
        return {"verdict": "unclear", "reason": "invalid judge output"}


def run(tag: str) -> Path:
    data = json.loads((HISTORY / f"{tag}.json").read_text(encoding="utf-8"))
    rows = data["results"] + [r for runs in data.get("other_runs", []) for r in runs]
    out = DATA_DIR / f"trajectory_judge_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "sub_type", "expected_tools", "trace", "judge_verdict", "judge_reason", "human_label"])
        for i, r in enumerate(rows):
            j = judge(r["question"], r.get("trace") or [], r.get("answer") or "")
            w.writerow([r["question"], r.get("sub_type"), " ".join(r.get("expected_tools") or []),
                        " | ".join(r.get("trace") or []), j["verdict"], j["reason"], ""])
            print(f"[{i:>2}] {j['verdict']:<13} {r['question'][:70]}")
    counts = Counter()
    with open(out, encoding="utf-8") as f:
        counts.update(row["judge_verdict"] for row in csv.DictReader(f))
    print(f"\n{dict(counts)}\nWritten: {out}\nFill 'human_label' (sensible / not_sensible) yourselves, then run --agreement.")
    return out


def agreement(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    labeled = [r for r in rows if r["human_label"].strip().lower() in ("sensible", "not_sensible")]
    if not labeled:
        sys.exit("No human_label filled in yet (sensible / not_sensible).")
    decided = [r for r in labeled if r["judge_verdict"] != "unclear"]
    agree = sum(r["judge_verdict"] == r["human_label"].strip().lower() for r in decided)
    conf = Counter((r["human_label"].strip().lower(), r["judge_verdict"]) for r in labeled)
    md = [f"### Trajectory judge vs hand labels ({len(labeled)} trajectories)", "",
          "| Agreement (judge decided) | Judge said unclear | Trajectories judged sensible |", "|---|---|---|",
          f"| {agree}/{len(decided)} = {agree / len(decided):.0%} | {len(labeled) - len(decided)} | "
          f"{sum(r['judge_verdict'] == 'sensible' for r in labeled)}/{len(labeled)} |" if decided else
          f"| - | {len(labeled)} | - |", "",
          "| Human \\ Judge | sensible | not_sensible | unclear |", "|---|---|---|---|"]
    for h in ("sensible", "not_sensible"):
        md.append(f"| {h} | " + " | ".join(str(conf[(h, v)]) for v in VERDICTS) + " |")
    md += ["", "Disagreements:"] + [f"- {r['question'][:80]}: human {r['human_label']}, judge {r['judge_verdict']} "
                                    f"({r['judge_reason'][:100]})" for r in decided
                                    if r["judge_verdict"] != r["human_label"].strip().lower()]
    md += ["", "Generated by `python scripts/trajectory_judge.py --agreement`."]
    text = "\n".join(md)
    print(text)
    update_section("TRAJECTORY_JUDGE", text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="saved agent eval run to judge (data/agent_eval_history/<tag>.json)")
    ap.add_argument("--agreement", help="CSV with human_label filled in")
    a = ap.parse_args()
    if a.agreement:
        agreement(a.agreement)
    elif a.tag:
        if not has_api_key():
            sys.exit("Needs OPENAI_API_KEY.")
        run(a.tag)
    else:
        ap.print_help()
