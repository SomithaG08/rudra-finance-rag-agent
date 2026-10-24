"""
threshold_sweep.py - Choose the refusal threshold from data (RAG day 2, mandatory).

With reranking on, the score that decides is the CROSS-ENCODER score of the top result
(it produces the final ranking; cosine and RRF scores are on other scales). For every
question of the eval set this script runs search + rerank (no answer generated; question
embeddings are cached) and splits the questions in two groups:

  * should refuse : out-of-scope questions + on_topic_unanswerable ones
  * answerable    : everything else (factual, cross-reference, stress tests, ambiguous)

For each candidate threshold it counts correct refusals (should-refuse questions below the
threshold) and wrong refusals (answerable questions below it), suggests the best one, and
writes the table to README.md (section THRESHOLD) + data/threshold_results.json.

    python scripts/threshold_sweep.py                 # reranker score (default)
    python scripts/threshold_sweep.py --cosine        # old way: best vector similarity
    python scripts/threshold_sweep.py --from-json     # recompute the table from the last run ($0)

Agent eval questions (category "agent") are left out: the agent eval measures them.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import (CONFIDENCE_THRESHOLD, DATA_DIR, QUESTIONS_FILE, RERANK_THRESHOLD,  # noqa: E402
                        has_api_key)
from app.retriever import get_retriever  # noqa: E402
from readme_tools import update_section  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_FILE))
    ap.add_argument("--cosine", action="store_true", help="use the best cosine similarity instead")
    ap.add_argument("--from-json", action="store_true", help="recompute from data/threshold_results.json ($0)")
    args = ap.parse_args()
    questions = [q for q in json.loads(Path(args.questions).read_text(encoding="utf-8"))
                 if not q.get("history") and q.get("category") != "agent"]
    if args.from_json:
        old = json.loads((DATA_DIR / "threshold_results.json").read_text(encoding="utf-8"))
        scores = {x["question"]: x["score"] for x in old["questions"]}
        rows = [{"score": scores[q["question"]], "should_refuse": should_refuse(q), "category": q.get("category"),
                 "question": q["question"]} for q in questions if q["question"] in scores]
        return report(rows, old["score"], old["score"] == "reranker score")
    if not has_api_key():
        sys.exit("Needs OPENAI_API_KEY (the questions are embedded; cached after the first run).")

    r = get_retriever()
    use_rerank = not args.cosine
    if use_rerank and not r.rerank_available():
        sys.exit("Reranking needs sentence-transformers (or run with --cosine).")
    rows = []
    for q in questions:
        res = r.search(q["question"], k=5, mode="hybrid", rerank=use_rerank)
        score = res["top_rerank"] if use_rerank else res["top_score"]
        rows.append({"score": score, "should_refuse": should_refuse(q), "category": q.get("category"),
                     "question": q["question"]})
    report(rows, "reranker score" if use_rerank else "cosine similarity", use_rerank)


def should_refuse(q: dict) -> bool:
    return q.get("category") == "out-of-scope" or q.get("sub_type") in ("on_topic_unanswerable", "out_of_scope")


def report(rows: list[dict], name: str, use_rerank: bool) -> None:
    print(f"\n{name:>15}  group           question")
    for x in sorted(rows, key=lambda x: x["score"]):
        print(f"{x['score']:15.3f}  {'SHOULD REFUSE' if x['should_refuse'] else 'answerable':<14}  {x['question'][:80]}")

    n_ref = sum(x["should_refuse"] for x in rows)
    n_ans = len(rows) - n_ref
    scores = sorted({round(x["score"], 3) for x in rows})
    cands = sorted({round((a + b) / 2, 3) for a, b in zip(scores, scores[1:])})
    current = RERANK_THRESHOLD if use_rerank else CONFIDENCE_THRESHOLD
    table, best = [], None
    for t in cands + [current]:
        good = sum(1 for x in rows if x["should_refuse"] and x["score"] < t)
        bad = sum(1 for x in rows if not x["should_refuse"] and x["score"] < t)
        table.append({"threshold": t, "correct_refusals": good, "wrong_refusals": bad})
        key = (good - 2 * bad, -abs(t))                  # a wrong refusal costs more than a missed one
        if best is None or key > best[0]:
            best = (key, t)
    table = sorted({row["threshold"]: row for row in table}.values(), key=lambda row: row["threshold"])
    # keep the rows around the interesting zone for the README
    shown = [row for row in table if row["correct_refusals"] > 0 or row["wrong_refusals"] == 0][:14]

    lines = [f"| Threshold ({name}) | Correct refusals (of {n_ref}) | Wrong refusals (of {n_ans}) |",
             "|---|---|---|"]
    for row in shown:
        mark = " **(suggested)**" if row["threshold"] == best[1] else (" (current)" if row["threshold"] == current else "")
        lines.append(f"| {row['threshold']:.3f}{mark} | {row['correct_refusals']} | {row['wrong_refusals']} |")
    md = "\n".join(lines)
    print("\n" + md)
    print(f"\nSuggested threshold: {best[1]:.3f} -> set {'RERANK_THRESHOLD' if use_rerank else 'CONFIDENCE_THRESHOLD'} "
          "in app/config.py. Above the threshold the prompt still refuses (rule 7).")
    (DATA_DIR / "threshold_results.json").write_text(json.dumps(
        {"score": name, "suggested": best[1], "table": table, "questions": rows}, indent=1,
        ensure_ascii=False), encoding="utf-8")
    update_section("THRESHOLD", f"### Confidence threshold ({name})\n\n{md}\n\n"
                   f"Should refuse = out-of-scope + on_topic_unanswerable questions; answerable = all "
                   f"others. Generated by `python scripts/threshold_sweep.py`.")


if __name__ == "__main__":
    main()
