"""
warm_cache.py - Demo Day safety net.

Asks the demo questions ONCE before the demo, with the app's default settings, so their
answers are stored in the answer cache (data/answer_cache.json). During the live demo these
questions are then answered instantly, for $0, even if the network or OpenAI is slow.

    python scripts/warm_cache.py                 # the default demo questions (~$0.01)
    python scripts/warm_cache.py "My question?"  # add your own questions

Note: the cache is keyed on the settings. Keep the default settings in the UI during the demo
(or re-run this script after changing the defaults in app/rag.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.config import has_api_key  # noqa: E402

DEMO_QUESTIONS = [
    "According to the rule of 72, how long does money take to double at 9%? And at 6%?",
    "What is diversification and why do investors practice it?",
    "What does DDM stand for, and which DDM is the most common?",
    "How is beta defined in the regression chapter, and how is it interpreted in the risk and return chapter?",
    "What is the effective annual interest rate on a credit card that charges 1.5% per month?",
    "Compare the payback period method with the NPV method.",
    "What is the current ratio and how is it interpreted?",
    "Tell me about returns.",
    "Quelle est la règle de 72 ?",
]


def main():
    if not has_api_key():
        sys.exit("OPENAI_API_KEY missing in .env")
    questions = sys.argv[1:] or DEMO_QUESTIONS
    total = 0.0
    for q in questions:
        r = rag.answer(q)                                  # default settings, cache ON
        total += r["cost_usd"]
        how = f"already cached ({r['cached']})" if r.get("cached") else "answered + cached"
        pages = len({c["page"] for c in r["citations"] if c["cited"]})
        print(f"{how:<24} pages cited={pages}  refused={r['refused']}  {q[:70]}")
    print(f"\nDone. Cost of this run: ${total:.4f}. These questions are now instant during the demo.")


if __name__ == "__main__":
    main()
