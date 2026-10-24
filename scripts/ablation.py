"""
ablation.py - The ablation table (RAG day 2, mandatory).

Starts from the basic version (vector search only) and adds each improvement on top of the
row above, in RETRIEVAL-ONLY mode (no answer generated: recall@k and MRR don't need one, and
the question embeddings are cached, so the table costs almost nothing):

    1. Vector only
    2. + reranking (cross-encoder, top-20 -> top-k)
    3. + BM25 (hybrid search, RRF)
    4. + query rewriting            (measured on the follow-up questions: those with "history")
    5. + HyDE            (optional, --with-hyde: one small LLM call per question)
    6. + query expansion (optional, --with-expansion: one small LLM call per question)

    python scripts/ablation.py                       # rows 1-4, writes README + data/ablation.json
    python scripts/ablation.py --with-hyde --with-expansion

Metrics per row: recall@1, @3, @5, MRR, full recall (all expected pages), latency per question.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app import tracing  # noqa: E402
from app.config import (CONFIDENCE_THRESHOLD, DATA_DIR, DEFAULT_TOP_K, PROMPT_VERSION,  # noqa: E402
                        QUESTIONS_FILE, RERANK_THRESHOLD, has_api_key)
from app.retriever import get_retriever  # noqa: E402
from eval_runner import evaluate_one, mean, rate  # noqa: E402
from readme_tools import update_section  # noqa: E402

OUT = DATA_DIR / "ablation.json"


def run_row(questions, known, **flags) -> dict:
    args = SimpleNamespace(k=flags.get("k", DEFAULT_TOP_K), mode=flags["mode"], prompt=PROMPT_VERSION,
                           threshold=CONFIDENCE_THRESHOLD, rerank_threshold=RERANK_THRESHOLD,
                           rerank=flags["rerank"], rewrite=flags.get("rewrite", False),
                           hyde=flags.get("hyde", False), expand=flags.get("expand", False),
                           injection=False, retrieval_only=True, judge=False)
    t0 = time.time()
    rows = [evaluate_one(q, args, known) for q in questions]
    scored = [r for r in rows if r["recall_at_1"] is not None]
    return {
        "n": len(scored),
        "recall_at_1": rate(r["recall_at_1"] for r in scored),
        "recall_at_3": rate(r["recall_at_3"] for r in scored),
        "recall_at_5": rate(r["recall_at_5"] for r in scored),
        "mrr": mean(r["reciprocal_rank"] for r in scored),
        "full_recall": rate(r["retrieval_hit"] for r in scored),
        "latency_s": round((time.time() - t0) / max(1, len(rows)), 3),
        "failed": [r["question"] for r in scored if not r["recall_at_5"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(QUESTIONS_FILE))
    ap.add_argument("--with-hyde", action="store_true")
    ap.add_argument("--with-expansion", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()
    if not has_api_key():
        sys.exit("Needs OPENAI_API_KEY: vector search embeds the questions (cached after the first run).")
    r = get_retriever()
    if not r.rerank_available():
        sys.exit("Reranking needs sentence-transformers: uv pip install sentence-transformers")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    base = [q for q in questions if not q.get("history")]
    follow = [q for q in questions if q.get("history")]
    known = r.sources

    rows = []
    plan = [("Vector only", base, dict(mode="dense", rerank=False)),
            ("+ reranking", base, dict(mode="dense", rerank=True)),
            ("+ BM25 (hybrid, RRF)", base, dict(mode="hybrid", rerank=True))]
    extra = dict(mode="hybrid", rerank=True)
    if args.with_hyde:
        extra = {**extra, "hyde": True}
        plan.append(("+ HyDE", base, dict(extra)))
    if args.with_expansion:
        extra = {**extra, "expand": True}
        plan.append(("+ query expansion", base, dict(extra)))
    if follow:
        plan.append(("Follow-ups, no rewriting", follow, dict(mode="hybrid", rerank=True, rewrite=False)))
        plan.append(("Follow-ups + query rewriting", follow, dict(mode="hybrid", rerank=True, rewrite=True)))

    for name, qs, flags in plan:
        print(f"-> {name} ({len(qs)} questions)")
        res = run_row(qs, known, k=args.k, **flags)
        res["version"] = name
        rows.append(res)
        print(f"   R@1 {res['recall_at_1']}  R@5 {res['recall_at_5']}  MRR {res['mrr']}  {res['latency_s']} s/q")

    def pct(v):
        return "-" if v is None else f"{v:.0%}"
    lines = ["| Version | Questions | Recall@1 | Recall@3 | Recall@5 | MRR | Full recall | Latency (s/q) |",
             "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['version']} | {x['n']} | {pct(x['recall_at_1'])} | {pct(x['recall_at_3'])} | "
                     f"{pct(x['recall_at_5'])} | {x['mrr'] if x['mrr'] is not None else '-'} | "
                     f"{pct(x['full_recall'])} | {x['latency_s']} |")
    table = "\n".join(lines)
    note = (f"\n\nRetrieval-only runs on `{Path(args.questions).name}`, top-k = {args.k}, each row on top "
            "of the one above; generated by `python scripts/ablation.py`. Kept: _write here which rows "
            "you keep and why_.")
    print("\n" + table)
    OUT.write_text(json.dumps({"k": args.k, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    update_section("ABLATION", "### Ablation table\n\n" + table + note)
    tracing.flush()


if __name__ == "__main__":
    main()
