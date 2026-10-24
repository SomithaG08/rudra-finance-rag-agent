# RAG Evaluation

## Evaluation set
`data/questions.json` - 47 questions: factual 12, cross-reference 10, stress-test 15,
out-of-scope 5, ambiguous 5 (5 priority questions).

## Tools (Assignment 2)
| Tool | File | What it does |
|---|---|---|
| Retrieval checker | `scripts/retrieval_checker.py` | top-k passages for one question, ✅/❌ for expected pages |
| Failure log | `scripts/FAILURE_LOG.md` | every failure, its cause, the fix, fixed or not |
| Eval runner | `scripts/eval_runner.py` | scorecard by category / difficulty, saves `data/eval_results.json` |
| Regression check | inside `eval_runner.py` | compares with the previous run: `REGRESSION` / `FIXED` |
| LLM-as-judge | `eval_runner.py --judge` | correctness, faithfulness, relevancy, completeness (1-5) |
| Compare runs | `scripts/compare_runs.py` | compares every saved run (and re-scores subsets) for $0 |
| Partial re-run | `eval_runner.py --only-failed` / `--ids 10,13` | re-asks only some questions and patches them into the last full run |
| Synthetic set | `scripts/generate_questions.py` | extra questions written from random pages (exercises skipped) |
| Dry run | `scripts/demo_dry_run.py` | 10 random questions + a follow-up, crash / citation / timing check |
| Cost report | `scripts/cost_report.py` | spend per operation vs the $5 budget |

## Metrics
- **hit(all)**: every expected page is in the top-k. **hit(any)**: at least one.
- **MRR**: 1 / rank of the first expected page.
- **refusal**: out-of-scope refused, in-scope and ambiguous not refused, on-topic-unanswerable
  refused or explicitly "not covered".
- **cite-hit**: the answer cites at least one expected page. **pages**: distinct pages cited per answer.
- **judge**: LLM-as-judge verdict.

## Day 1 baseline (dense only, 500-char chunks, k=10, prompt v1)
- All expected pages retrieved: 72% (26/36); at least one: 97%.
- Cross-reference, all pages: 3/10. Out-of-scope refused: 5/5.
- Answers cite one page in almost every case.

## Day 2 measurements
- BM25 alone on the same chunks, k=8 (offline, measured): all pages 92%, any 100%, MRR 0.84.
- Full runs with the judge: see Day 3 measurements below.

## Day 3 measurements (30 Sep, hybrid search, Day-1 chunks, gpt-4o-mini)

### Hand-written set - 47 questions (`data/questions.json`)
| Run | k | Prompt | hit(any) | MRR | ctx-P | ctx-R | refusal | faithful | relevant | judge | pass | cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3-baseline | 8 | v3 | 100% | 0.94 | 32% | 97% | 100% | 91% | 100% | 81% | 81% | $0.030 |
| v4 | 8 | v4 | 100% | 0.94 | 32% | 97% | 100% | 91% | 100% | 79% | 79% | $0.030 |
| k5 | 5 | v3 | 100% | 0.94 | 44% | 91% | 100% | 91% | 100% | 79% | 79% | $0.023 |

Day 1 -> Day 3 on the same questions: all expected pages retrieved **72% -> 92%**, at least one
**97% -> 100%**, out-of-scope refused **5/5 -> 5/5**, faithfulness 91%, relevancy 100%.
k = 5 raises precision (32% -> 44%) but loses pages on cross-reference questions (hit(all) 92% -> 81%),
so **k stays 8**. v3 and v4 differ by one question (judge noise on a list answer); **v4 is kept**
for its prompt-injection rule (4/4 injection questions handled).

### Synthetic set - 47 questions written from random pages (`data/questions_synthetic.json`)
3 generated questions were removed (written from end-of-chapter exercises or with several valid
answers); scores below are re-scored on the 47 kept questions with `compare_runs.py` ($0).
| Run | k | Prompt | hit | MRR | ctx-P | ctx-R | faithful | relevant | judge | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic | 8 | v3 | 94% | 0.78 | 20% | 94% | 98% | 100% | 94% | 87% |
| synthetic-v4-k5 | 5 | v4 | 91% | 0.77 | 30% | 91% | 99% | 100% | 100% | 91% |

Context precision is low by construction: one expected page, max 2 chunks per page, so the best
possible value is 2/k (25% at k=8, 40% at k=5).

### Reranking
The `rerank` run gave exactly the same results as the baseline: `sentence-transformers` is not
installed, so the option was silently skipped. Install it to measure it for real.

## Out-of-scope threshold
Best cosine similarity on Day 1: out-of-scope 0.14-0.36, in-scope/ambiguous 0.46-0.85.
Threshold set to 0.40 (`CONFIDENCE_THRESHOLD` in `app/config.py`, slider in the UI).

See `scripts/FAILURE_LOG.md` for each failure and fix.
