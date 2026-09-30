"""
threshold_sweep.py - Choose the confidence threshold from data instead of by eye.

For every question of the eval set it takes the best vector similarity of the retrieval
(question embeddings are cached in data/query_cache.json, so after the first eval run this
is free) and counts, for each threshold:
  * correct refusals : out-of-scope questions whose best similarity is BELOW the threshold
  * wrong refusals   : answerable questions (factual, cross-reference, stress-test, ambiguous)
                       that would be refused because they are below the threshold
The best threshold refuses all out-of-scope questions and no answerable one.

    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --questions data/questions_injection.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CONFIDENCE_THRESHOLD, QUESTIONS_FILE, has_api_key  # noqa: E402
from app.retriever import get_retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_FILE))
    args = ap.parse_args()
    if not has_api_key():
        sys.exit("Needs OPENAI_API_KEY (the similarity comes from the question embedding).")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    r = get_retriever()
    rows = []
    for q in questions:
        top = r.search(q["question"], k=5, mode="dense")["top_score"]
        should_refuse = q.get("category") == "out-of-scope"
        unanswerable = q.get("sub_type") == "on_topic_unanswerable"
        rows.append((top, should_refuse, unanswerable, q["question"]))

    print(f"\n{'sim':>6}  {'type':<14} question")
    for top, refuse, unans, text in sorted(rows):
        kind = "OUT-OF-SCOPE" if refuse else ("unanswerable" if unans else "answerable")
        print(f"{top:6.3f}  {kind:<14} {text[:80]}")

    n_out = sum(1 for _, refuse, _, _ in rows if refuse)
    n_in = sum(1 for _, refuse, unans, _ in rows if not refuse and not unans)
    print(f"\n{'threshold':>9} {'correct refusals':>17} {'wrong refusals':>15}")
    best = None
    for t in [x / 100 for x in range(25, 56, 1)]:
        good = sum(1 for top, refuse, _, _ in rows if refuse and top < t)
        bad = sum(1 for top, refuse, unans, _ in rows if not refuse and not unans and top < t)
        mark = " <- current" if abs(t - CONFIDENCE_THRESHOLD) < 1e-9 else ""
        print(f"{t:9.2f} {good:>11}/{n_out:<5} {bad:>9}/{n_in:<5}{mark}")
        score = good - 2 * bad                     # a wrong refusal is worse than a missed one
        if best is None or score > best[0]:
            best = (score, t)
    print(f"\nSuggested threshold: {best[1]:.2f} (set CONFIDENCE_THRESHOLD in app/config.py). "
          "Out-of-scope questions above the threshold are still refused by the prompt (rule 7).")


if __name__ == "__main__":
    main()
