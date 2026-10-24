"""
note_tool.py - Second action tool ("going further: more acting tools"): save a study note.

The agent PREPARES a Markdown note (title + content + the book passages it used as sources).
Nothing is written until a human clicks "Approve & save" after seeing a PREVIEW of the file.
After writing we CHECK THE OUTCOME (file exists, size, first line) instead of trusting the model.
Fake mode (agent eval / injection test): every proposal is declined and logged.

    data/notes/<date>_<title>.md       the saved notes
    data/fake_save_log.jsonl           declined proposals (fake mode)
"""

import json
import re
from datetime import datetime

from app.config import DATA_DIR, source_title

NOTES_DIR = DATA_DIR / "notes"
FAKE_SAVE_LOG = DATA_DIR / "fake_save_log.jsonl"
MAX_CHARS = 6000


def slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title or "note").strip("-").lower()
    return (s or "note")[:60]


def build_note(title: str, content: str, passages: list[dict] | None = None) -> str:
    """The exact Markdown that will be written (shown as the preview)."""
    lines = [f"# {title.strip() or 'Study note'}", "",
             f"*Saved by the Personal Finance Assistant (Team Rudra) on {datetime.now():%Y-%m-%d %H:%M}*", "",
             (content or "").strip()[:MAX_CHARS]]
    seen, srcs = set(), []
    for p in passages or []:
        key = (p.get("source"), p.get("page"))
        if key in seen or p.get("page") is None:
            continue
        seen.add(key)
        title_src = source_title(p["source"]) if p.get("source") else "Textbook"
        srcs.append(f"- [{p.get('n')}] {title_src}, page {p['page']}"
                    + (f" - {p['section']}" if p.get("section") else ""))
    if srcs:
        lines += ["", "## Sources", ""] + srcs
    return "\n".join(lines) + "\n"


def prepare_note(title: str, content: str, passages: list[dict] | None = None) -> dict:
    """Prepare the note for human approval. Writes NOTHING."""
    text = build_note(title, content, passages)
    return {"success": True, "status": "approval_required", "title": title, "preview": text,
            "path": str(NOTES_DIR / f"{datetime.now():%Y-%m-%d}_{slug(title)}.md")}


def fake_save(title: str, content: str, reason: str = "") -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAKE_SAVE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "title": title,
                            "content": content[:500], "status": "declined", "reason": reason},
                           ensure_ascii=False) + "\n")
    return {"success": False, "status": "declined", "title": title,
            "note": "The human declined saving this note (fake mode). Do not prepare it again; answer the user."}


def save_note(title: str, text: str, approved: bool = False) -> dict:
    """Write the previewed note ONLY when approved=True, then verify the file on disk."""
    if not approved:
        return {"success": True, "status": "approval_required", "title": title, "preview": text}
    if not (text or "").strip():
        return {"success": False, "status": "failed", "message": "Empty note: nothing saved."}
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTES_DIR / f"{datetime.now():%Y-%m-%d}_{slug(title)}.md"
    n = 2
    while path.exists():
        path = NOTES_DIR / f"{datetime.now():%Y-%m-%d}_{slug(title)}-{n}.md"
        n += 1
    path.write_text(text, encoding="utf-8")
    ok = path.exists() and path.read_text(encoding="utf-8") == text          # outcome check
    return {"success": ok, "status": "saved" if ok else "failed", "path": str(path),
            "bytes": path.stat().st_size if path.exists() else 0}
