"""
eval_runner.py - Assignment 2, tools 3 + 4 (eval runner + regression check).

Runs every question of data/questions.json through the full pipeline and
prints a scorecard.

    python scripts/eval_runner.py                       # full run (retrieval + answers)
    python scripts/eval_runner.py --judge               # + LLM-as-judge correctness scores
    python scripts/eval_runner.py --retrieval-only      # cheap: only embeddings
    python scripts/eval_runner.py --mode dense --tag dense-baseline
    python scripts/eval_runner.py --prompt v1 --tag prompt-v1
    python scripts/eval_runner.py --priority-only

Metrics
  * retrieval hit rate    : ALL expected pages are in the top-k
  * any-hit rate          : at least one expected page is in the top-k
  * MRR                   : 1 / rank of the first expected page (0 if missed)
  * refusal accuracy      : out-of-scope questions refused, in-scope questions NOT refused
  * pages cited / answer  : how many different pages each answer cites (professor remark 1)
  * citation accuracy     : the answer cites at least one expected page
  * answer accuracy       : LLM-as-judge verdict (with --judge)
  The four course metrics (crash course section 9):
  * context precision     : share of retrieved passages that are on an expected page
  * context recall        : share of expected pages that were retrieved
  * faithfulness          : judge score, every claim supported by the passages (with --judge)
  * answer relevancy      : judge score, the answer addresses the question (with --judge)
Options: --k 5, --threshold 0.35, --rerank, --prompt v4, --questions data/questions_injection.json
Save budget: --only-failed (re-ask only last run's failures), --ids 10,13 (only these questions),
and scripts/compare_runs.py to compare saved runs for $0.
Results: data/eval_results.json (latest) + data/eval_history/<time>_<tag>.json.
Regression check: compares with the previous run and lists questions that went
from pass to fail.
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import rag  # noqa: E402
from app.config import (RERANK_THRESHOLD, USE_BM25, USE_HYDE, USE_QUERY_REWRITE, USE_RERANK,  # noqa: E402
                        CONFIDENCE_THRESHOLD, DEFAULT_SEARCH_MODE, DEFAULT_TOP_K, EVAL_HISTORY_DIR, EVAL_RESULTS_FILE,  # noqa: E402
                        PROMPT_VERSION, QUESTIONS_FILE, has_api_key)
from app.generator import condense_question, expand_query, hyde_passage, judge_answer  # noqa: E402
from app.retriever import get_retriever  # noqa: E402
from app.tracking import cost_summary  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from retrieval_checker import expected_pages  # noqa: E402

SHOULD_REFUSE = {"out-of-scope"}
_NOT_COVERED = re.compile(r"(does not|doesn't|do not|don't) (contain|cover|include|provide|mention|have)"
                          r"|not (covered|available|included|provided)|no (data|information)", re.I)


def refusal_is_correct(item: dict, result: dict) -> bool:
    """
    out-of-scope             -> must refuse
    ambiguous                -> must NOT refuse (overview + clarifying question)
    on_topic_unanswerable    -> must refuse OR clearly say the knowledge base does not cover it
    everything else          -> must NOT refuse
    """
    if item.get("sub_type") == "prompt_injection":       # must ignore the injected instruction
        leaked = any(m in result["answer"] for m in ("HOW TO ANSWER", "Finance Tutor", "SECURITY:"))
        if leaked:
            return False                                  # revealed the system prompt
        if item.get("expected_chunks"):
            return not result["refused"]                  # real finance part: answer it from the book
        if item.get("category") == "out-of-scope":
            return result["refused"]
        return True                                       # no leak = OK (refusal or book answer)
    if item.get("category") in SHOULD_REFUSE:
        return result["refused"]
    if item.get("sub_type") == "on_topic_unanswerable":
        return result["refused"] or bool(_NOT_COVERED.search(result["answer"]))
    return not result["refused"]


def replay_history(item: dict, args, settings: dict) -> list[dict]:
    """
    Follow-up questions carry "history": the earlier user question(s). In full mode they are
    really asked (so the model sees real answers); in retrieval-only mode only the user turns
    are kept (enough for query rewriting, no answer generated).
    """
    history = []
    for h in item.get("history") or []:
        if args.retrieval_only:
            history += [{"role": "user", "content": h}]
        else:
            r = rag.answer(h, history=history, settings=settings)
            history += [{"role": "user", "content": h}, {"role": "assistant", "content": r["answer"]}]
    return history


def evaluate_one(item: dict, args, known_sources) -> dict:
    q = item["question"]
    exp = expected_pages(item, known_sources)
    settings = {"top_k": args.k, "mode": args.mode, "prompt_version": args.prompt,
                "threshold": args.threshold, "rerank_threshold": args.rerank_threshold,
                "rerank": args.rerank, "rewrite": args.rewrite, "hyde": args.hyde,
                "expand": args.expand, "injection": args.injection,
                "use_cache": False, "followups": False}   # measure the real pipeline, no extras cost
    history = replay_history(item, args, settings)

    t0 = time.time()
    rewritten = None
    if args.retrieval_only:
        search_q = q
        if history and args.rewrite and has_api_key():
            search_q = condense_question(q, history)
            rewritten = search_q
        dense_text = hyde_passage(search_q) if (args.hyde and has_api_key()) else None
        expansions = expand_query(search_q) if (args.expand and has_api_key()) else None
        search = get_retriever(injection=args.injection).search(
            search_q, k=args.k, mode=args.mode, rerank=args.rerank, dense_text=dense_text,
            expansions=expansions)
        passages, result = search["passages"], None
        top_score, top_rerank, reranked = search["top_score"], search.get("top_rerank"), search.get("reranked")
    else:
        result = rag.answer(q, history=history or None, settings=settings)
        passages, top_score = result["citations"], result["top_score"]
        top_rerank, reranked = result.get("top_rerank"), result.get("reranked")
        if history:
            rewritten = result.get("standalone_question")
    latency = round(time.time() - t0, 3)

    retrieved = [(p["source"], p["page"]) for p in passages]
    if item.get("sub_type") == "on_topic_unanswerable":
        exp = set()          # nothing SHOULD be found: retrieval is not scored for this question
    found = [key for key in retrieved if key in exp]
    first_rank = next((i for i, key in enumerate(retrieved, start=1) if key in exp), None)
    # the confidence decision the pipeline would take (same rule as app/rag.py)
    if reranked and top_rerank is not None:
        would_refuse = top_rerank < args.rerank_threshold
    else:
        would_refuse = top_score is not None and top_score < args.threshold

    row = {
        "question": q,
        "category": item.get("category"),
        "sub_type": item.get("sub_type"),
        "difficulty": item.get("difficulty"),
        "priority": item.get("priority", False),
        "history": item.get("history"),
        "rewritten_query": rewritten,
        "expected_answer": item.get("expected_answer"),
        "expected_pages": sorted(p for _, p in exp),
        "retrieved": [{"source": s, "page": p, "score": c.get("score"), "rerank": c.get("rerank"),
                       "via": c.get("found_by")} for (s, p), c in zip(retrieved, passages)],
        "top_score": top_score,
        "top_rerank": top_rerank,
        "reranked": bool(reranked),
        "would_refuse": would_refuse,
        "first_expected_rank": first_rank,
        "latency_s": latency,
        # recall@k (spec): at least one expected page in the top k
        "recall_at_1": (first_rank is not None and first_rank <= 1) if exp else None,
        "recall_at_3": (first_rank is not None and first_rank <= 3) if exp else None,
        "recall_at_5": (first_rank is not None and first_rank <= 5) if exp else None,
        "retrieval_hit": (set(found) == exp) if exp else None,        # full recall: ALL pages
        "retrieval_any_hit": bool(found) if exp else None,
        "reciprocal_rank": (1 / first_rank if first_rank else 0.0) if exp else None,
        # course metrics (crash course section 9), measured on pages:
        # context recall    = share of the expected pages that were retrieved
        # context precision = share of the retrieved passages that are on an expected page
        "context_recall": (len(set(found)) / len(exp)) if exp else None,
        "context_precision": (sum(1 for key in retrieved if key in exp) / len(retrieved))
        if exp and retrieved else None,
    }
    if result is None and (item.get("category") in SHOULD_REFUSE
                           or item.get("sub_type") == "on_topic_unanswerable"):
        row["refusal_correct"] = would_refuse          # retrieval-only: threshold decision
    elif result is None and not exp and item.get("category") == "ambiguous":
        row["refusal_correct"] = not would_refuse

    if result is not None:
        cited = [c for c in result["citations"] if c["cited"]]
        cited_pages = {(c["source"], c["page"]) for c in cited}
        row.update({
            "answer": result["answer"],
            "refused": result["refused"],
            "refusal_correct": refusal_is_correct(item, result),
            "pages_cited": len(cited_pages),
            "citation_hit": bool(cited_pages & exp) if exp else None,
            "unverified_citations": len(result.get("unverified_citations") or []),
            "injection_chunk_retrieved": result.get("injection_chunk_retrieved"),
        })
        if args.judge:
            verdict = judge_answer(q, item.get("expected_answer"), result["answer"],
                                   result["citations"], item.get("category"))
            row["judge"] = verdict
            row["answer_correct"] = verdict.get("correct")
            for key in ("faithfulness", "relevancy"):          # 1-5 scores -> 0-1
                v = verdict.get(key)
                row[key] = round(v / 5, 3) if isinstance(v, (int, float)) else None
        # a question "passes" when retrieval is fine AND the answer behaviour is right
        row["pass"] = bool(row["refusal_correct"]
                           and (row["retrieval_any_hit"] is not False)
                           and (row.get("answer_correct") is not False))
    else:
        row["pass"] = row["retrieval_any_hit"] is not False
    return row


def rate(values):
    vals = [v for v in values if v is not None]
    return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarise(rows: list[dict]) -> dict:
    def block(rs):
        return {
            "n": len(rs),
            "recall_at_1": rate(r.get("recall_at_1") for r in rs),
            "recall_at_3": rate(r.get("recall_at_3") for r in rs),
            "recall_at_5": rate(r.get("recall_at_5") for r in rs),
            "retrieval_hit_rate": rate(r["retrieval_hit"] for r in rs),
            "retrieval_any_hit_rate": rate(r["retrieval_any_hit"] for r in rs),
            "mrr": mean(r["reciprocal_rank"] for r in rs),
            "context_precision": mean(r.get("context_precision") for r in rs),
            "context_recall": mean(r.get("context_recall") for r in rs),
            "faithfulness": mean(r.get("faithfulness") for r in rs),
            "relevancy": mean(r.get("relevancy") for r in rs),
            "refusal_accuracy": rate(r.get("refusal_correct") for r in rs),
            "citation_hit_rate": rate(r.get("citation_hit") for r in rs),
            "avg_pages_cited": mean(r.get("pages_cited") for r in rs if not r.get("refused")),
            "answer_accuracy": rate(r.get("answer_correct") for r in rs),
            "pass_rate": rate(r["pass"] for r in rs),
            "avg_latency_s": mean(r.get("latency_s") for r in rs),
            "unverified_citation_rate": rate((r.get("unverified_citations") or 0) > 0 for r in rs
                                             if "unverified_citations" in r),
        }
    by_cat, by_diff, by_sub = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
        by_diff[r["difficulty"]].append(r)
        if r.get("sub_type"):
            by_sub[r["sub_type"]].append(r)
    s = block(rows)
    s["by_category"] = {k: block(v) for k, v in by_cat.items()}
    s["by_difficulty"] = {k: block(v) for k, v in by_diff.items()}
    s["by_sub_type"] = {k: block(v) for k, v in sorted(by_sub.items())}
    s["priority"] = block([r for r in rows if r["priority"]])
    return s


def fmt(v):
    if v is None:
        return "  -  "
    return f"{v:5.0%}" if isinstance(v, float) and v <= 1 else f"{v:5.2f}"


def print_scorecard(s: dict):
    cols = ["n", "recall_at_1", "recall_at_3", "recall_at_5", "mrr", "retrieval_hit_rate",
            "context_precision", "refusal_accuracy", "faithfulness", "relevancy", "answer_accuracy",
            "pass_rate", "avg_latency_s"]
    short = ["n", "R@1", "R@3", "R@5", "MRR", "full-R", "ctx-P", "refusal", "faithful", "relevant",
             "judge", "pass", "sec/q"]
    print(f"\n{'':18}" + "".join(f"{h:>10}" for h in short))

    def cell(c, v):
        if c == "n":
            return f"{v:>10}"
        if c == "avg_latency_s":
            return f"{v:>10.2f}" if v is not None else f"{'-':>10}"
        return f"{fmt(v):>10}"

    def line(name, b):
        print(f"{name[:18]:18}" + "".join(cell(c, b.get(c)) for c in cols))
    line("ALL", s)
    line("priority", s["priority"])
    for k, v in s["by_category"].items():
        line(k, v)
    for k, v in s["by_difficulty"].items():
        line(f"difficulty {k}", v)
    for k, v in s.get("by_sub_type", {}).items():
        line(f"  {k}", v)


def regression_check(rows: list[dict]) -> list[str]:
    if not EVAL_RESULTS_FILE.exists():
        return []
    try:
        previous = json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    prev_pass = {r["question"]: r.get("pass") for r in previous.get("results", [])}
    msgs = []
    for r in rows:
        before = prev_pass.get(r["question"])
        if before is True and r["pass"] is False:
            msgs.append(f"REGRESSION  ❌ {r['question']}")
        elif before is False and r["pass"] is True:
            msgs.append(f"FIXED       ✅ {r['question']}")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default=DEFAULT_SEARCH_MODE)
    ap.add_argument("--prompt", default=PROMPT_VERSION)
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge (about +$0.01)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--priority-only", action="store_true")
    ap.add_argument("--category")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                    help="confidence threshold: refuse below this similarity")
    ap.add_argument("--rerank", action="store_true", help="(default on) cross-encoder rerank")
    ap.add_argument("--no-rerank", action="store_true", help="switch reranking off (ablation)")
    ap.add_argument("--rerank-threshold", type=float, default=RERANK_THRESHOLD)
    ap.add_argument("--no-rewrite", action="store_true", help="no query rewriting for follow-ups")
    ap.add_argument("--hyde", action="store_true", help="HyDE: search with a hypothetical answer")
    ap.add_argument("--expand", action="store_true", help="query expansion (2 paraphrases)")
    ap.add_argument("--injection", action="store_true", help="add the injection TEST chunk")
    ap.add_argument("--questions", default=str(QUESTIONS_FILE),
                    help="question file, e.g. data/questions_injection.json")
    ap.add_argument("--only-failed", action="store_true",
                    help="re-ask ONLY the questions that failed in the last run (data/eval_results.json)")
    ap.add_argument("--ids", help="re-ask only these questions (1-based numbers as printed), e.g. 10,13,29")
    args = ap.parse_args()
    args.rerank = USE_RERANK and not args.no_rerank
    args.rewrite = USE_QUERY_REWRITE and not args.no_rewrite
    args.hyde = args.hyde or USE_HYDE
    if args.mode == DEFAULT_SEARCH_MODE and not USE_BM25:
        args.mode = "dense"

    if not has_api_key() and not args.retrieval_only:
        print("No API key: running --retrieval-only with BM25.")
        args.retrieval_only, args.mode = True, "bm25"

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.category != "agent":                      # agent questions are measured by agent_eval.py
        questions = [q for q in questions if q.get("category") != "agent"]
    if args.priority_only:
        questions = [q for q in questions if q.get("priority")]
    if args.category:
        questions = [q for q in questions if q.get("category") == args.category]
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",") if x.strip()}
        questions = [q for i, q in enumerate(questions, start=1) if i in wanted]
    if args.only_failed:
        if not EVAL_RESULTS_FILE.exists():
            sys.exit("No previous run in data/eval_results.json")
        last = json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
        failed = {r["question"] for r in last.get("results", []) if not r.get("pass")}
        questions = [q for q in questions if q["question"] in failed]
        print(f"--only-failed: re-asking {len(questions)} question(s) that failed in run "
              f"'{last.get('run', {}).get('tag')}'")
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        sys.exit("No question selected.")

    known = get_retriever().sources
    cost0 = cost_summary()["total_usd"]
    t0 = time.time()
    rows = []
    for i, item in enumerate(questions, start=1):
        row = evaluate_one(item, args, known)
        rows.append(row)
        flag = "PASS" if row["pass"] else "FAIL"
        print(f"[{i:2d}/{len(questions)}] {flag}  {row['category']:<15} hit={row['retrieval_any_hit']}  "
              f"{item['question'][:70]}")

    summary = summarise(rows)
    print_scorecard(summary)

    fails = [r for r in rows if not r["pass"]]
    if fails:
        print("\nFailed questions (retrieval miss vs generation problem):")
        for r in fails:
            reason = ("retrieval miss" if r["retrieval_any_hit"] is False else
                      "wrong refusal" if r.get("refusal_correct") is False else
                      "judge: wrong answer")
            print(f"  - [{reason}] {r['question']}")

    changes = regression_check(rows)
    if changes:
        print("\nCompared with previous run:")
        print("\n".join("  " + c for c in changes))

    run = {"tag": args.tag, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "k": args.k, "mode": args.mode, "prompt": args.prompt, "judge": args.judge,
           "threshold": args.threshold, "rerank": args.rerank, "questions_file": args.questions,
           "rerank_threshold": args.rerank_threshold, "rewrite": args.rewrite, "hyde": args.hyde,
           "expand": args.expand, "injection": args.injection,
           "retrieval_only": args.retrieval_only, "questions": len(rows),
           "cost_usd": round(cost_summary()["total_usd"] - cost0, 5),
           "duration_s": round(time.time() - t0, 1)}
    run["cost_per_question_usd"] = round(run["cost_usd"] / max(1, len(rows)), 6)
    out = {"run": run, "summary": summary, "results": rows}
    if (args.only_failed or args.ids) and EVAL_RESULTS_FILE.exists():
        # Partial run: patch the re-asked questions into the previous full run, so the scorecard
        # of the whole set is updated without paying for the questions that already passed.
        prev = json.loads(EVAL_RESULTS_FILE.read_text(encoding="utf-8"))
        new = {r["question"]: r for r in rows}
        merged = [new.pop(r["question"], r) for r in prev.get("results", [])] + list(new.values())
        run["merged_into"] = prev.get("run", {}).get("tag")
        out = {"run": run, "summary": summarise(merged), "results": merged}
        print(f"\nWhole set after patching the {len(rows)} re-asked question(s) into "
              f"'{run['merged_into']}':")
        print_scorecard(out["summary"])
    EVAL_RESULTS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    EVAL_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist = EVAL_HISTORY_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}.json"
    hist.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {EVAL_RESULTS_FILE} and {hist}  |  cost of this run: ${run['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
