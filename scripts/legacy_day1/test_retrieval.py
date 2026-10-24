"""Day-1: show the top-10 chunks for two hard questions (one embedding each)."""
from _setup import retrieve

questions = [
    "The textbook opens two different chapters with American Airlines. What point does each chapter make with this example?",
    "What does DDM stand for, and which DDM is the most common?",
]
for question in questions:
    print("\n" + "=" * 80 + f"\nQUESTION:\n{question}\n" + "=" * 80)
    for i, r in enumerate(retrieve(question, k=10), start=1):
        score = f"{r['score']:.3f}" if r.get("score") is not None else "-"
        print(f"\n--- Result {i} --- similarity {score} - page {r['page']}\n{r['text'][:700]}")
