"""
demo_save_note.py - Smoke test of the save_note action with a real model, in FAKE mode
(the proposal is declined and logged, nothing is written). Prints the step trace.

    python scripts/demo_save_note.py "Explain the rule of 72 from the book and save it as a study note."
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agent import run_agent  # noqa: E402

q = " ".join(sys.argv[1:]) or "Explain the rule of 72 from the book and save it as a study note."
r = run_agent(q, send_mode="fake")
print("QUESTION:", q)
print("\n".join(r["trace"]))
print("save proposals:", r["save_proposals"])
print("ANSWER:", r["answer"][:400])
print(f"{r['steps']} steps, {r['model_calls']} model calls, {r['latency_s']} s, stop: {r['stop_reason']}")
