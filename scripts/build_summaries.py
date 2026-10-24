"""
build_summaries.py - Write one short summary per book section (run ONCE, by one person).

    python scripts/build_summaries.py --dry-run     # free: number of sections + cost estimate
    python scripts/build_summaries.py               # asks you to type YES, then ~$0.08
    python scripts/build_summaries.py --limit 5     # try on 5 sections first (~$0.003)

Output: data/summaries.json (section id, title, pages, summary, embedding of the summary).
Already summarised sections (same text) are skipped, so you can stop and resume for free.
Share data/summaries.json with the team like my_index.faiss.
Used by app/rag.py for broad questions ("Summarize chapter 7", "main ideas of section 15.2").
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CORPUS_FILE, PRICES, has_api_key  # noqa: E402
from app.summaries import SUMMARIES_FILE, group_pages  # noqa: E402
from app.text_utils import count_tokens  # noqa: E402

MAX_INPUT_TOKENS = 7000           # longest sections are cut to this size
SUMMARY_PROMPT = (
    "You summarise one section of the OpenStax textbook 'Principles of Finance' for students. "
    "Using ONLY the text given, write 120-180 words: the main ideas, the key terms with a short "
    "definition, and any formula (written plainly, e.g. FV = PV x (1 + r)^n). No introduction "
    "sentence, no outside knowledge.")


def _cut(text: str) -> str:
    if count_tokens(text) <= MAX_INPUT_TOKENS:
        return text
    return text[: MAX_INPUT_TOKENS * 4]            # ~4 characters per token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    groups = group_pages(corpus)
    if args.limit:
        groups = groups[: args.limit]
    done = {}
    if SUMMARIES_FILE.exists():
        done = {s["hash"]: s for s in json.loads(SUMMARIES_FILE.read_text(encoding="utf-8"))}
    for g in groups:
        g["text"] = _cut(g["text"].strip())
        g["hash"] = hashlib.sha1(g["text"].encode("utf-8")).hexdigest()[:16]
    todo = [g for g in groups if g["hash"] not in done]

    tokens_in = sum(count_tokens(g["text"]) + 120 for g in todo)
    tokens_out = 300 * len(todo)
    p = PRICES["gpt-4o-mini"]
    cost = (tokens_in * p["input"] + tokens_out * p["output"]) / 1e6
    print(f"{len(groups)} sections, {len(groups) - len(todo)} already done, {len(todo)} to summarise")
    print(f"~{tokens_in:,} input + ~{tokens_out:,} output tokens  ->  about ${cost:.3f} (+ <$0.001 embeddings)")
    for g in todo[:8]:
        print(f"   {g['id']:>6}  {g['section'][:60]}  pages {g['pages'][0]}-{g['pages'][-1]}")
    if args.dry_run or not todo:
        return
    if not has_api_key():
        sys.exit("OPENAI_API_KEY missing in .env")
    if not args.yes and input("Type YES to continue: ").strip() != "YES":
        sys.exit("Cancelled - nothing was spent.")

    from app.llm import chat, embed_texts
    results = list(done.values())
    for i, g in enumerate(todo, start=1):
        summary = chat([{"role": "system", "content": SUMMARY_PROMPT},
                        {"role": "user", "content": f"Section {g['section']}\n\n{g['text']}"}],
                       operation="section_summary", temperature=0, max_tokens=350)
        item = {k: g[k] for k in ("id", "chapter", "section", "source", "pages", "hash")}
        item["summary"] = summary
        results.append(item)
        print(f"[{i}/{len(todo)}] {g['id']} {g['section'][:60]}")
        if i % 10 == 0 or i == len(todo):                        # save progress (resume-safe)
            SUMMARIES_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    missing = [s for s in results if not s.get("vector")]
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        vecs = embed_texts([f"{s['section']}\n{s['summary']}" for s in batch], operation="embed_summaries")
        for s, v in zip(batch, vecs):
            s["vector"] = [round(float(x), 6) for x in v]
    order = {g["hash"]: n for n, g in enumerate(groups)}
    results.sort(key=lambda s: order.get(s["hash"], 10**6))
    SUMMARIES_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(results)} summaries to {SUMMARIES_FILE}")


if __name__ == "__main__":
    main()
