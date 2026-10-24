# Failure Log

How to use: run `python scripts/eval_runner.py`, pick a failed question, run
`python scripts/retrieval_checker.py --id <n>` to see whether it is a **retrieval** problem
(right page not in top-k) or a **generation** problem (page found, answer wrong), fix, re-run
the eval runner (it prints `REGRESSION` / `FIXED` against the previous run), then log it here.

## Baseline (Day 1)

Source: `data/evaluation_results.json` produced by the Day-1 `evaluate.py`
(500-character chunks, dense FAISS only, k = 10, prompt v1). Retrieval figures below were
recomputed from that file by comparing retrieved pages with `expected_chunks` in `questions.json`
(36 questions that have expected pages, the "DAL 2023" trap question excluded).

| Metric | Day 1 |
|---|---|
| All expected pages in top-10 | 26 / 36 (72%) |
| At least one expected page in top-10 | 35 / 36 (97%) |
| Cross-reference questions, all pages found | 3 / 10 |
| Out-of-scope correctly refused | 5 / 5 |
| Pages cited per answer | almost always **one** ("Source: ..., Page 229") |

Same questions, keyword search (BM25) only on the same Day-1 chunks, k = 8, measured on Day 2:
all expected pages 92%, at least one 100%, MRR 0.84 - keyword search alone already finds the
pages vector search missed, which is why the default became **hybrid**.

## Failures and fixes

| # | Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|----------|----------|:---:|---------|-------------|:---:|
| 1 | "What does DDM stand for, and which DDM is the most common?" | stress-test (acronym) | ❌ | The acronym "DDM" has almost no meaning for the embedding model; no chunk with "Gordon growth model" in top-10, so the model refused. | Hybrid search: BM25 keyword search + FAISS, merged with Reciprocal Rank Fusion. BM25 alone puts page 355 ("The most common DDM is the Gordon growth model") at rank 1. | ✅ (retrieval, BM25 check) |
| 2 | "The textbook opens two different chapters with American Airlines..." | cross-reference | ⚠️ one of two chapters | Only one chapter's opening in the top-k and the v1 prompt refused instead of answering the part it could. | Hybrid search + diversity cap (max 2 chunks per page, so more distinct pages enter the top-k) + prompt v3 rule 4 (answer the covered part, say what is missing). | re-check with eval runner |
| 3 | "How is beta defined in the regression chapter, and how is it interpreted in the risk and return chapter?" | cross-reference | ⚠️ page 468 missed | Two chapters needed; vector search returned several overlapping chunks of page 438. | Diversity cap per page; sentence chunking (fewer, more complete chunks); prompt asks to use and cite **all** passages. | re-check |
| 4 | "What were the average annual return and standard deviation of DAL stock over 2011-2020?" | cross-reference (contradictory) | ⚠️ | The book gives two different figures (22.40% / 51.90% vs 20.87% / 51.16%); the Day-1 answer reported one silently. | Prompt v3 rule 5: report both values with citations and flag the discrepancy. | re-check |
| 5 | "What is the payback period method, what is the NPV decision rule, and what is the IRR decision rule?" | stress-test (multi-part) | ✅ | Three concepts on three pages; the answer used one source per paragraph but only as free text citations. | Numbered passages + mandatory `[n]` citations (prompt v2/v3), dynamic "list" hint. | ✅ in UI test |
| 6 | "Tell me about returns." / "Explain the model." | ambiguous | n/a | Day-1 answer picked one random interpretation (e.g. the DDM) without saying the question was vague. | Prompt v3 rule 6: short overview of the main meanings found + one clarifying question. | re-check |
| 7 | "Is a ratio of 2 good?" | ambiguous | n/a | Answered "Yes" for the current ratio only, while the question does not say which ratio. | Same as #6. | re-check |
| 8 | "What was Delta Air Lines' (DAL) stock return in 2023?" | stress-test (on-topic, unanswerable) | n/a | Correct refusal on Day 1 - must stay correct after the prompt became more "helpful". | Eval runner accepts a refusal or an explicit "the book does not cover 2023"; regression check watches it. | ✅ keep watching |
| 9 | All answers | UI | - | Citations were plain text inside the answer ("Source: principles-finance_-_WEB.pdf, Page 229"), usually one page; the Day-1 UI only listed file + page and, because of an import bug, actually showed the **hard-coded demo answers** with a single source. | Fixed the import bug (the UI imported a `Retriever` class that did not exist and `generate_answer()` returned a string, not a dict). New UI: inline clickable `[n]` citations that open the PDF at the page, a Sources list under every answer, and a Sources panel with every retrieved passage (cited / not cited, section, printed page, similarity, highlighted words). | ✅ |
| 10 | "What is the capital of Australia?" | out-of-scope | n/a | Best similarity 0.36 - the closest to our in-scope minimum (0.46). | Confidence threshold 0.40 (refuse without calling the LLM); the prompt still refuses above the threshold. Threshold slider in the UI for the demo. | ✅ (threshold) |

## Next items to check after the first full run with the API key
- Run `python scripts/eval_runner.py --judge --tag v3-hybrid` and compare with
  `--mode dense --prompt v1 --tag day1-like`.
- Add every new failed question to the table above (question, category, retrieval OK?, problem,
  fix, fixed?).

## Entries from the agentic-assistant branch (Naveen)

This document records retrieval and generation issues identified during
development and the fixes applied to the RAG system.

| Question | Category | Retrieval OK? | Problem | Fix Applied | Fixed? |
|---|---|:---:|---|---|:---:|
| "What is the current ratio?" | factual | ✅ | Initial answer did not consistently cite all relevant retrieved sources. | Updated the generation prompt to require citations for every source/page directly used in the answer. | ✅ |
| "The textbook opens two different chapters with American Airlines. What point does each chapter make with this example?" | cross-reference | ✅ | The answer required information from multiple parts of the textbook. | Updated the generation prompt to explicitly combine relevant information from multiple retrieved chunks and cite all supporting pages. | ✅ |
| "What does DDM stand for, and which DDM is the most common?" | factual | ✅ | The answer required retrieving the relevant DDM section rather than relying on a single chunk. | Tested retrieval and generation using the evaluation toolkit. | ✅ |
| "What was Delta Air Lines' (DAL) stock return in 2023?" | stress-test | ❌ | The corpus does not contain the requested 2023 data. Retrieved content only covered earlier historical data. | Kept the context-only refusal behavior so the system does not invent information outside the corpus. | ✅ |
| "What is diversification and why do investors practice it?" | factual | ✅ | Retrieval checker initially reported a false retrieval miss because the source filename in `questions.json` differed from the filename stored in the FAISS metadata. | Updated the retrieval checker to use the expected page when checking whether the expected chunk was retrieved. | ✅ |

#### Notes

### Retrieval debugging

The retrieval checker was created to inspect the top-k retrieved chunks,
including:

- similarity score
- source
- page
- retrieved text
- expected chunk status

The checker was tested using the diversification question. The expected
Page 464 chunk was retrieved at rank 1 with a similarity score of 0.6023.

### Generation debugging

The generation prompt was updated to ensure that answers using information
from multiple retrieved chunks include citations for all relevant source
and page combinations.

### Out-of-scope and unavailable information

The system is instructed to avoid using outside knowledge. When the
retrieved corpus does not contain enough information to answer a question,
the system responds:

"I don't have enough information to answer this question."

### Cost tracking

API usage tracking was added for GPT-4o-mini generation. Embedding cost
tracking is also implemented for future intentional index builds, while
the existing FAISS index was not rebuilt unnecessarily.

## Day 3 - failures found by the evaluation runs (30 Sep)

Retrieval is no longer the main problem (hit(any) 100% on the hand-written set); most remaining
failures are generation problems. Analysed from the saved runs, no re-run needed.

| Question | Category | Retrieval OK? | Problem | Fix / decision | Fixed? |
|---|---|:---:|---|---|:---:|
| "Approximately how many gallons of jet fuel does American Airlines use each year..." | factual (calculation) | ✅ | Consumption correct, extra cost computed wrongly | Calculation must go through the calculator tools; check the tool call | ❌ |
| "How is the payback period for Sam's embroidery machine calculated..." | cross-reference | ✅ | Payback given as 8 years instead of 4 | Same: force the payback_period tool for payback questions | ❌ |
| "The textbook opens two different chapters with American Airlines..." | cross-reference | ✅ | Chapter 20 point missed (commodity prices) | Both pages retrieved; prompt rule 4 (use all passages) not followed - candidate for query decomposition | ❌ |
| "Average annual return and standard deviation of DAL stock 2011-2020?" | cross-reference (contradictory) | ⚠️ | Conflicting values not both reported | Prompt rule 5 exists; model ignored it - add a contradiction example to the prompt | ❌ |
| "What does LLC stand for?" | cross-reference (contradictory) | ✅ | "Limited Liability Corporation" vs "Company" in the book, inconsistency not mentioned | Same as above | ❌ |
| "When did the US Treasury yield curve invert before the COVID-19 recession..." | stress-test (temporal) | ✅ | Wrong month for the end of the recession | Proof sentences now show the exact source sentence to check dates | ❌ |
| "Tell me about returns." / "What are the main risks?" / "How do I calculate the rate?" | ambiguous | n/a | Judge compared with "Expected answer: None" and marked answers wrong | Judge now receives the expected behaviour (overview + clarifying question); re-check with `--only-failed` | ✅ (judge) |
| "What function in R is used to calculate the sample standard deviation?" | synthetic | ❌ | Answer is in a table row ("sd( ) Calculates the sample standard deviation", p. 415); the words appear on many pages, the table page is ranked low | Table extraction (pdfplumber) planned | ❌ |
| "What is revenue...", "definition of unemployment", "NPV for the given cash flows" | synthetic | ⚠️ | Answer correct but from another page with the same content | Not a real failure (repeated information) | ✅ |
| 3 synthetic questions (fund depletion amounts, regression slope 78.075, value chain inventory) | synthetic | ✅ | Written from end-of-chapter exercises: the book has no answer | Removed; generator now skips exercise pages | ✅ |
| Reranking run identical to baseline | tooling | - | sentence-transformers not installed, option skipped silently | Install before measuring reranking | ❌ |
