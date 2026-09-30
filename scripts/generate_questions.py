"""
generate_questions.py - Synthetic evaluation questions written from the book itself.

Our hand-written set has 47 questions. This script picks random chunks of the book and asks
gpt-4o-mini to write ONE question that the chunk answers, with the expected answer. Because
we know which page the question came from, retrieval can be scored automatically
(was that page found?), exactly like the hand-written questions.

    python scripts/generate_questions.py --n 50 --dry-run   # free: cost estimate
    python scripts/generate_questions.py --n 50             # asks YES, ~$0.01
    python scripts/eval_runner.py --questions data/questions_synthetic.json --judge --tag synthetic

Output: data/questions_synthetic.json (same format as data/questions.json, sub_type "synthetic").
Tip: read a few generated questions before trusting the scores - drop the bad ones.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CHUNKS_FILE, DATA_DIR, PRICES, has_api_key  # noqa: E402

OUT_FILE = DATA_DIR / "questions_synthetic.json"
PROMPT = (
    "You write evaluation questions for a question-answering system about a finance textbook. "
    "Read the passage and write ONE question that a student could ask and that this passage "
    "answers on its own. Rules: the question must make sense without seeing the passage (never "
    "say 'the passage', 'the text', 'this section', 'above'); use your own words rather than "
    "copying a sentence; prefer definitions, reasons, formulas, rules and numbers from the "
    "passage. Return JSON: {\"question\": str, \"expected_answer\": str (1-2 sentences), "
    "\"difficulty\": \"easy\"|\"medium\"|\"hard\", \"usable\": true|false} - usable=false if the "
    "passage is only exercises, a table of contents, references or too fragmentary.")
# End-of-chapter exercises have no answer in the book: questions written from them are unfair tests.
_EXERCISE = re.compile(r"Review Questions|Multiple Choice|Problems\b|Critical Thinking|Video Activity|"
                       r"_{4,}|\n?\b\d{1,2} ?\. [A-Z][^?]{10,200}\?\s+a\. ", re.I)
_BAD = re.compile(r"\b(the|this) (passage|text|section|excerpt|chunk)\b|\babove\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    pool = [c for c in chunks if len(c["text"]) >= 350 and c["page"] > 20   # skip front matter
            and not _EXERCISE.search(c["text"])]                               # and exercises
    random.Random(args.seed).shuffle(pool)
    picked, seen_pages = [], set()
    for c in pool:                                       # one question per page at most
        if (c["source"], c["page"]) in seen_pages:
            continue
        seen_pages.add((c["source"], c["page"]))
        picked.append(c)
        if len(picked) >= int(args.n * 1.3) + 2:         # extra, some will be unusable
            break

    p = PRICES["gpt-4o-mini"]
    tokens_in = sum(len(c["text"]) // 4 + 220 for c in picked)
    cost = (tokens_in * p["input"] + 120 * len(picked) * p["output"]) / 1e6
    print(f"{len(picked)} chunks sampled for {args.n} questions  ->  about ${cost:.4f}")
    if args.dry_run:
        return
    if not has_api_key():
        sys.exit("OPENAI_API_KEY missing in .env")
    if not args.yes and input("Type YES to continue: ").strip() != "YES":
        sys.exit("Cancelled - nothing was spent.")

    from app.llm import chat
    out = []
    for c in picked:
        if len(out) >= args.n:
            break
        raw = chat([{"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"Passage (page {c['page']}):\n{c['text']}"}],
                   operation="synthetic_question", json_mode=True, temperature=0.3, max_tokens=220)
        try:
            q = json.loads(raw)
        except json.JSONDecodeError:
            continue
        text = (q.get("question") or "").strip()
        if not q.get("usable", True) or len(text) < 15 or _BAD.search(text):
            continue
        out.append({
            "question": text,
            "expected_answer": q.get("expected_answer"),
            "source": c["source"],
            "page": c["page"],
            "category": "factual",
            "sub_type": "synthetic",
            "difficulty": q.get("difficulty") if q.get("difficulty") in ("easy", "medium", "hard") else "medium",
            "priority": False,
            "expected_chunks": [{"source": c["source"], "page": c["page"]}],
        })
        print(f"[{len(out)}/{args.n}] p.{c['page']}: {text}")
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(out)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
