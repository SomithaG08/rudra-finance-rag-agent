"""Day-1: print answer + retrieved chunks of the questions that failed. No API call."""
import json
from pathlib import Path

import _setup  # noqa: F401

RESULTS_FILE = "data/evaluation_results.json"
TARGET_QUESTIONS = [
    "The textbook opens two different chapters with American Airlines. What point does each chapter make with this example?",
    "What does DDM stand for, and which DDM is the most common?",
    "What was Delta Air Lines' (DAL) stock return in 2023?",
]


def main():
    for r in json.loads(Path(RESULTS_FILE).read_text(encoding="utf-8")):
        if r["question"] not in TARGET_QUESTIONS:
            continue
        print("\n" + "=" * 80 + f"\nQUESTION:\n{r['question']}\n" + "=" * 80)
        print(f"\nANSWER:\n{r['answer']}\n\nRETRIEVED CHUNKS:")
        for i, c in enumerate(r["retrieved_chunks"], start=1):
            score = f"{c['score']:.3f}" if c.get("score") is not None else "-"
            print(f"\n--- Chunk {i} --- similarity {score} - {c['source']} p.{c['page']}\n{c['text']}")


if __name__ == "__main__":
    main()
