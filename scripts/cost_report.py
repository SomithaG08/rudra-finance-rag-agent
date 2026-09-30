"""
cost_report.py - How much of the $5 budget has been spent, by operation.

    python scripts/cost_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tracking import cost_summary  # noqa: E402

BUDGET = 5.00

s = cost_summary()
print(f"Total: ${s['total_usd']:.4f} in {s['calls']} API calls "
      f"({s['total_usd'] / BUDGET:.1%} of the ${BUDGET:.0f} budget)")
for op, v in sorted(s["by_operation"].items(), key=lambda kv: -kv[1]["cost_usd"]):
    print(f"  {op:<18} {v['calls']:5d} calls  {v['tokens']:>10,} tokens  ${v['cost_usd']:.4f}")
