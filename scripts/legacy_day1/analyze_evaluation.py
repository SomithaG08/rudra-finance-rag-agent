"""Day-1 summary of data/evaluation_results.json (categories, priority questions, refusals,
lowest retrieval scores). No API call."""
import json
from collections import Counter
from pathlib import Path

import _setup  # noqa: F401  (sets the working directory to the repository root)

RESULTS_FILE = "data/evaluation_results.json"
REFUSAL = "I don't have enough information to answer this question."


def main():
    results = json.loads(Path(RESULTS_FILE).read_text(encoding="utf-8"))
    print(f"\n===== EVALUATION SUMMARY =====\nTotal questions: {len(results)}")
    print("\n----- Categories -----")
    for category, count in Counter(r.get("category", "unknown") for r in results).items():
        print(f"{category}: {count}")
    priority = [r for r in results if r.get("priority") is True]
    print(f"\n----- Priority Questions ({len(priority)}) -----")
    for r in priority:
        print(f"\nID: {r.get('id')}\nQuestion: {r['question']}\nAnswer: {r['answer']}")
    refusals = [r for r in results if REFUSAL.lower() in r["answer"].lower()]
    print(f"\n----- Refusals: {len(refusals)} -----")
    for r in refusals:
        print(f"\nQuestion: {r['question']}\nCategory: {r.get('category')}")
    print("\n----- Lowest Retrieval Scores -----")
    scored = sorted((r["retrieved_chunks"][0].get("score") or 0, r["question"])
                    for r in results if r.get("retrieved_chunks"))
    for score, question in scored[:10]:
        print(f"{score:.3f} - {question}")


if __name__ == "__main__":
    main()
