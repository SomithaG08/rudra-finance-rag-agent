"""
demo_dry_run.py - Demo Day rehearsal.

Asks 10 random questions from data/questions.json (plus a follow-up question),
and checks that:
  * the pipeline does not crash
  * in-scope answers contain citations (and how many different pages)
  * out-of-scope questions are refused
  * response time is acceptable

    python scripts/demo_dry_run.py
    python scripts/demo_dry_run.py --n 5 --seed 7
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import rag  # noqa: E402
from app.config import QUESTIONS_FILE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    random.seed(args.seed)
    sample = random.sample(questions, min(args.n, len(questions)))

    problems, times = [], []
    for item in sample:
        q = item["question"]
        try:
            r = rag.answer(q, settings={"use_cache": False})
        except Exception as exc:                    # noqa: BLE001
            problems.append(f"CRASH on '{q}': {exc}")
            print(f"CRASH  {q}")
            continue
        times.append(r["timings"]["total"])
        pages = {(c["source"], c["page"]) for c in r["citations"] if c["cited"]}
        oos = item["category"] == "out-of-scope"
        ok = r["refused"] if oos else (not r["refused"] and len(pages) >= 1)
        if not ok:
            problems.append(f"{'should refuse' if oos else 'no citation / wrongly refused'}: {q}")
        print(f"{'OK  ' if ok else 'WARN'}  {r['timings']['total']:5.1f}s  pages cited={len(pages)}  "
              f"refused={r['refused']}  [{item['category']}] {q[:60]}")

    # Conversation memory check
    first = rag.answer("According to the rule of 72, how long does it take money to double at 9%?")
    history = [{"role": "user", "content": first["question"]},
               {"role": "assistant", "content": first["answer"]}]
    follow = rag.answer("And at 6%?", history=history)
    print(f"\nFollow-up rewritten as: {follow['standalone_question']}")
    print(f"Follow-up answer: {follow['answer'][:200]}")

    print("\n===== DRY RUN SUMMARY =====")
    if times:
        print(f"Average response time: {sum(times) / len(times):.1f}s (max {max(times):.1f}s)")
    print("No problems found." if not problems else "\n".join(problems))


if __name__ == "__main__":
    main()
