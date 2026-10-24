"""
retrieval_checker.py - Assignment 2, tool 1 (required).

Shows exactly what the retriever finds for ONE question, and whether the pages
we expect (from data/questions.json) are in the top-k.

    python scripts/retrieval_checker.py "What does DDM stand for?"
    python scripts/retrieval_checker.py --id 24               # question #24 of questions.json
    python scripts/retrieval_checker.py --id 24 --mode dense --k 5
    python scripts/retrieval_checker.py --list                # list all questions with their id

If the right pages are NOT retrieved, the LLM never had a chance: fix retrieval
(chunking, hybrid search, k) before touching the prompt.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_SEARCH_MODE, DEFAULT_TOP_K, QUESTIONS_FILE  # noqa: E402
from app.retriever import get_retriever  # noqa: E402
from app.text_utils import resolve_source  # noqa: E402


def expected_pages(item: dict, known_sources: list[str]) -> set[tuple[str, int]]:
    """{(source, page)} expected for a question (None for out-of-scope / ambiguous)."""
    pages = set()
    for e in item.get("expected_chunks") or []:
        pages.add((resolve_source(e.get("source"), known_sources), int(e["page"])))
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--id", type=int, help="index of the question in data/questions.json (0-based)")
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default=DEFAULT_SEARCH_MODE)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-rerank", action="store_true", help="show the ranking before reranking")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")) if QUESTIONS_FILE.exists() else []
    if args.list:
        for i, q in enumerate(questions):
            print(f"{i:3d} [{q['category']:<15}] {q['question']}")
        return

    item = {}
    if args.id is not None:
        item = questions[args.id]
        question = item["question"]
    elif args.question:
        question = args.question
        item = next((q for q in questions if q["question"].strip().lower() == question.strip().lower()), {})
    else:
        ap.error("give a question or --id")

    r = get_retriever()
    res = r.search(question, k=args.k, mode=args.mode, rerank=not args.no_rerank)
    exp = expected_pages(item, r.sources)

    print("=" * 90)
    print(f"QUESTION : {question}")
    if item:
        print(f"CATEGORY : {item.get('category')}  |  difficulty: {item.get('difficulty')}")
        print(f"EXPECTED : {item.get('expected_answer')}")
        print(f"PAGES    : {sorted(p for _, p in exp) if exp else 'none (should be refused / clarified)'}")
    line = f"MODE     : {res['mode']}  |  k = {args.k}"
    if res["top_score"] is not None:
        line += f"  |  best FAISS similarity = {res['top_score']:.3f}"
    if res.get("reranked"):
        line += f"  |  best cross-encoder score = {res['top_rerank']:.2f}"
    else:
        line += "  |  not reranked"
    print(line)
    print("=" * 90)

    found = set()
    for p in res["passages"]:
        key = (p["source"], p["page"])
        mark = "✅" if key in exp else "  "
        if key in exp:
            found.add(key)
        sim = f"{p['score']:.3f}" if p["score"] is not None else "  -  "
        bm = f"{p['bm25']:.1f}" if p["bm25"] else "-"
        rr = f"{p['rerank']:.2f}" if p.get("rerank") is not None else "-"
        print(f"{mark} #{p['rank']:<2} faiss={sim} rerank={rr:<6} bm25={bm:<5} via={'+'.join(p['found_by']):<22} "
              f"{p['source']} p.{p['page']}  [{p.get('section') or ''}]")
        print(f"      {p['text'][:220].strip()}...")

    if exp:
        missed = exp - found
        print("-" * 90)
        print(f"Expected pages retrieved: {len(found)}/{len(exp)}")
        for s, pg in sorted(missed, key=lambda t: t[1]):
            print(f"❌ missed: {s} p.{pg}")
        if not missed:
            print("✅ all expected pages are in the top-k")


if __name__ == "__main__":
    main()
