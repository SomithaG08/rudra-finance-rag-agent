"""Day-1 evaluation (Naveen): answers every question of data/questions.json and saves
data/evaluation_results.json. The full scorecard is scripts/eval_runner.py."""
import json
from pathlib import Path

from _setup import generate_answer, retrieve

QUESTIONS_FILE = "data/questions.json"
OUTPUT_FILE = "data/evaluation_results.json"


def main():
    questions = json.loads(Path(QUESTIONS_FILE).read_text(encoding="utf-8"))
    results = []
    print(f"Evaluating {len(questions)} questions...\n")
    for i, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{i}/{len(questions)}] {question}")
        retrieved = retrieve(question, k=10)
        answer = generate_answer(question, retrieved)
        results.append({
            "id": item.get("id"), "question": question, "category": item.get("category"),
            "priority": item.get("priority"), "answer": answer,
            "retrieved_chunks": [{"source": c["source"], "page": c["page"], "score": c["score"],
                                  "text": c["text"]} for c in retrieved],
        })
    Path(OUTPUT_FILE).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== EVALUATION COMPLETE =====\nQuestions evaluated: {len(results)}\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
