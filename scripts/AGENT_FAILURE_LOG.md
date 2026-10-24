# Agent failure log (Agentic AI project, day 2)

One entry per agent failure found with `scripts/agent_eval.py`, the injection test or the demo.
Fill it in **yourselves** (course rule: no GenAI for the evaluation content). Read the numbered
trace in `data/agent_traces.jsonl` or under the answer in the Agent tab to find the failing step.

Failure types (pick one): wrong tool · missing tool · wrong arguments · repeated call ·
step cap hit · bad recovery (gave up / looped) · hallucinated answer · refused a covered question ·
answered an out-of-scope question · unwanted action (send_message) · citation error.

## How to read a trace

1. Open `data/agent_traces.jsonl` (one JSON object per line) or the accordion in the Agent tab.
2. Find the entry for the failed question (match by `question` field).
3. Look at each step in `trajectory`: `{"step": N, "action": "...", "args": {...}, "result": {...}}`.
4. The **first wrong step** is what you log — a bad early decision causes everything that follows.

## Failure table

| # | Date | Question | Sub-type | Failure type | Trace (step where it failed) | Root cause | Fix tried | Result after fix (tag) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-30 | "According to the rule of 72, how many years does it take to double savings at 9%? What about at 6%?" | search_then_calculate | math_in_head | Step 1 — agent called `calculator` (72/9) immediately; never called `search_corpus` | LLM already knows the Rule of 72 formula; skips retrieval when it can compute the answer directly | Rephrase expected_tools to only require calculator; or add system prompt note that tool text must be cited | – |
| 2 | 2026-09-30 | "Summarize what the book says about diversification and send the summary to my team." | act_with_approval | missing tool (send_message) | Step 2 — agent answered directly after search; never proposed `send_message` | Agent does not infer "send … to my team" as a trigger to call `send_message` from the tool description | Add explicit example: "If the user asks you to send, share or forward something, call send_message." | – |
| 3 |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |
| 6 |  |  |  |  |  |  |  |  |

## Known failure categories to watch for

Run `agent_eval.py`, read the traces, and tally failures under these categories.

| Category | Description | Likely questions affected | Count |
|---|---|---|---|
| **wrong_tool** | Agent calls `search_corpus` when `lookup_term` or `get_section` would be exact | definition / section questions | 0 |
| **math_in_head** | Agent calculates arithmetic without calling `calculator` or `cash_flow_analysis` | search_then_calculate, cash_flow | 1 |
| **no_recovery** | Agent returns "No matching passages found" and gives up instead of rephrasing | rephrase_and_retry | 0 (search succeeded first try) |
| **redundant_search** | Identical `search_corpus` call repeated (should be blocked, but check the trace) | multi_search | 0 |
| **step_cap_hit** | Agent hits MAX_STEPS without answering; often caused by bad tool descriptions | complex multi-step | 0 |
| **hallucinated_answer** | Agent answers an out-of-scope question with invented content | out_of_scope | 0 |
| **unwanted_action** | Agent proposes `send_message` when user never asked for it (injection or over-eager) | injection_against_actions | 0 |
| **citation_error** | Answer cites a page that does not support the claim | any | 0 |

### Observed failure: math_in_head (search_then_calculate)

Trace excerpt (trace index 14 in `data/agent_traces.jsonl`):

```
step 1: action=calculator  args={"expression": "72 / 9"}
        thought: "I will calculate ... using the rule of 72 for both 9% and 6%."
        result: 8.0
step 2: action=calculator  args={"expression": "72 / 6"}
        result: 12.0
step 3: action=answer
        result: "According to the rule of 72, it takes approximately 8 years at 9% and 12 years at 6%."
```

The agent went straight to `calculator` because it already knows the formula. The textbook passage
(page 229) confirming "72 / 9 = 8 years" was never retrieved. Tool recall = 50 % (calculator
present; search_corpus absent).

### Observed failure: missing tool — send_message (act_with_approval)

Trace excerpt (trace index 20 in `data/agent_traces.jsonl`):

```
step 1: action=search_corpus  args={"query": "diversification"}
        top_rerank=4.81  →  3 passages returned
step 2: action=answer
        result: "Diversification is a strategy … 'Don't put all of your eggs in one basket'."
```

"…and send the summary to my team" was ignored. The agent summarised correctly but did not call
`send_message`. Tool recall = 50 % (search_corpus present; send_message absent). The send_message
tool description does not include an example that links "send to my team" to the tool trigger.

## Fix log

After identifying the biggest category, fix it (tool description, system prompt, step cap) and re-run:

    python scripts/agent_eval.py --tag agent-fix1
    python scripts/agent_eval.py --regression agent-v2 agent-fix1

Record each change and its effect here:

| Tag | Change made | Category fixed | Correctness before → after | Notes |
|---|---|---|---|---|
| agent-v2 | baseline | – | – / 16 | initial eval run |
| agent-v3 | prompt agent_v3: "ground first" rule (search the book before calculator, even for known formulas) + answer check in agent.py (1 retry when the answer has no book passage) | math_in_head / no search (rule of 72, credit-card EAR, priority Q1 + Q5) | – (not marked) | agent-v3: all 4 now call search_corpus first; tool recall 81% -> 97% |
| agent-v3 | agent_v3: explicit send trigger words ("send / share ... to my team") + answer check nudges when send_message was asked but not called | missing tool (send_message) | – (not marked) | agent-v3: send_message proposed (declined by fake send) |
| agent-v3 | agent_v3: one-best-tool table (lookup_term alone for definitions, search_corpus for "what and why") | redundant tool call (lookup_term then search_corpus) | – (not marked) | agent-v3: 3 extra lookup_term calls gone, exact tool set 56% -> 88% |
| agent-v3-check2 | answer check: "math" nudge (computed numbers without calculator) + "citation" nudge (no [n]) | math_in_head after search (rule of 72 x2, credit-card EAR flaky) | – (not marked) | check2: search_then_calculate 50% -> 100% but 6 steps; check3 (one-call calculator, all problems in one nudge): 4 steps; agent-v3-final: tool recall 100%, 2.4 avg steps, 0 step-cap hits |
| agent-v3-final4 | action allowlist (actions only when the user asks), greeting rule, router escalation on RAG refusal, judge rubric accepts the standard refusal | unwanted action (save_note), generic greeting, rephrase question refused under the router | judge-correct 82% -> 94% (router 76% -> 94%) | only the greeting stayed wrong in final4; fixed after (greeting-check run: correct) |

## Regression check

After every fix, re-run the priority questions first, agent eval only when those pass:

    python scripts/agent_eval.py --tag agent-fix1 --ids <priority question ids>
    python scripts/agent_eval.py --tag agent-fix1
    python scripts/agent_eval.py --regression agent-v2 agent-fix1
