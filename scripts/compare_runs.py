"""
compare_runs.py - Compare evaluation runs WITHOUT calling the API ($0).

Every run of scripts/eval_runner.py is saved in data/eval_history/. This script reads those
files and prints one line per run, so you can compare settings (prompt, k, rerank ...)
without re-running any question.

    python scripts/compare_runs.py                                    # all runs
    python scripts/compare_runs.py --filter synthetic                 # runs whose tag contains "synthetic"
    python scripts/compare_runs.py --questions data/questions_synthetic.json
          # re-score the saved runs on the questions that are STILL in that file
          # (e.g. after deleting bad questions) - no question is re-asked
    python scripts/compare_runs.py --diff v3-baseline v4              # question-by-question changes
"""

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "data" / "eval_history"

COLS = [("n", "n"), ("recall_at_1", "R@1"), ("recall_at_5", "R@5"), ("retrieval_any_hit", "hit"),
        ("reciprocal_rank", "MRR"),
        ("context_precision", "ctx-P"), ("context_recall", "ctx-R"), ("refusal_correct", "refusal"),
        ("faithfulness", "faithful"), ("relevancy", "relevant"), ("answer_correct", "judge"),
        ("pass", "pass")]


def _avg(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def score(rows):
    out = {"n": len(rows)}
    for key, _ in COLS[1:]:
        out[key] = _avg(r.get(key) for r in rows)
    return out


def load_runs(pattern):
    runs = []
    for f in sorted(glob.glob(str(HISTORY / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if pattern and pattern not in d["run"]["tag"]:
            continue
        runs.append(d)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", help="only runs whose tag contains this text")
    ap.add_argument("--questions", help="re-score only the questions still in this file")
    ap.add_argument("--diff", nargs=2, metavar=("TAG_A", "TAG_B"))
    args = ap.parse_args()

    runs = load_runs(args.filter)
    if not runs:
        sys.exit("No run found in data/eval_history/")

    if args.diff:
        by_tag = {r["run"]["tag"]: r for r in runs}
        a, b = (by_tag.get(t) for t in args.diff)
        if not a or not b:
            sys.exit(f"Tags not found. Available: {', '.join(by_tag)}")
        pa = {r["question"]: r for r in a["results"]}
        for r in b["results"]:
            before = pa.get(r["question"], {}).get("pass")
            if before is not None and before != r["pass"]:
                print(("FIXED      " if r["pass"] else "REGRESSION ") + r["question"])
        return

    keep = None
    if args.questions:
        keep = {q["question"] for q in json.loads(Path(args.questions).read_text(encoding="utf-8"))}

    print(f"{'tag':<22}" + "".join(f"{h:>9}" for _, h in COLS) + "   k  prompt  cost")
    for d in runs:
        rows = d["results"]
        if keep is not None:
            rows = [r for r in rows if r["question"] in keep]
            if not rows:
                continue
        s = score(rows)
        cells = []
        for key, _ in COLS:
            v = s[key]
            cells.append(f"{v:>9}" if key == "n" else ("        -" if v is None else f"{v:>9.0%}"))
        run = d["run"]
        print(f"{run['tag'][:22]:<22}" + "".join(cells)
              + f"  {run.get('k', ''):>2}  {run.get('prompt', ''):<6}  ${run.get('cost_usd', 0):.3f}")
    if keep is not None:
        print(f"\n(re-scored on the {len(keep)} questions of {args.questions}; no API call)")


if __name__ == "__main__":
    main()
