"""Shared setup for the Day-1 scripts: run them from the repository root, e.g.
    python scripts/legacy_day1/evaluate.py
They now use the merged app (hybrid retrieval, prompt v3) instead of the Day-1 modules."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # the Day-1 scripts use paths such as "data/questions.json"

from app.generator import generate_answer as _generate  # noqa: E402
from app.retriever import retrieve  # noqa: E402,F401


def generate_answer(question, retrieved):
    """Day-1 signature: returns the answer text (the merged generator returns a dict)."""
    return _generate(question, retrieved)["answer"]
