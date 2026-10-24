"""
readme_tools.py - Write a generated table into README.md between two markers, e.g.

    <!-- ABLATION:START -->
    ... replaced by scripts/ablation.py ...
    <!-- ABLATION:END -->

Used by ablation.py, threshold_sweep.py, injection_test.py, agent_eval.py and
agent_injection_test.py, so the README always shows the latest measured numbers.
"""

from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def update_section(name: str, markdown: str) -> None:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    block = f"{start}\n{markdown.strip()}\n{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        after = rest.split(end, 1)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    README.write_text(text, encoding="utf-8")
    print(f"README.md updated (section {name}).")
