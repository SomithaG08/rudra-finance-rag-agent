"""
Offline unit tests - no API key, no network needed.

    python tests/test_offline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bm25 import BM25
from app.chunking import chunk_corpus, chunk_sentences
from app.generator import classify_question, is_refusal, parse_citations
from app.text_utils import parse_page_footer, resolve_source, tokenize


def test_footer():
    text, section, printed = parse_page_footer("Money doubles. 7.4 • Applications of TVM in Finance 215")
    assert text == "Money doubles." and section == "7.4 Applications of TVM in Finance" and printed == 215
    text, section, printed = parse_page_footer(
        "Hello. 216 7 • Time Value of Money I: Single Payment Value Access for free at openstax.org")
    assert section.startswith("Chapter 7") and printed == 216


def test_chunking_keeps_sentences_whole():
    text = " ".join(f"Sentence number {i} talks about finance and money." for i in range(200))
    chunks = chunk_sentences(text, max_tokens=60, overlap_tokens=15)
    assert len(chunks) > 5
    assert all(c.endswith(".") for c in chunks), "a chunk was cut in the middle of a sentence"
    corpus = [{"source": "a.pdf", "page": 1, "text": text}, {"source": "a.pdf", "page": 2, "text": ""}]
    out = chunk_corpus(corpus)
    assert all(c["page"] == 1 for c in out) and out[0]["chunk_id"] == 0


def test_bm25_finds_acronym():
    docs = ["The most common DDM is the Gordon growth model.",
            "Diversification reduces unsystematic risk.",
            "The current ratio divides current assets by current liabilities."]
    bm = BM25(docs)
    assert bm.search("What does DDM stand for?", 1)[0][0] == 0
    assert bm.coverage("Who won the FIFA world cup") < 0.75


def test_citations_and_refusal():
    assert parse_citations("A [1]. B [3][1]. C [2, 4]. D [9]", 5) == [1, 3, 2, 4]
    assert is_refusal("I don't have enough information to answer this question.")
    assert not is_refusal("The rule of 72 ... [1]")


def test_tools_and_language():
    from app.features import detect_language
    from app.tools import run_tool
    assert run_tool("rule_of_72", '{"rate_percent": 9}')["years_rule_of_72"] == 8.0
    ear = run_tool("effective_annual_rate",
                   '{"rate_percent": 1.5, "periods_per_year": 12, "rate_is_per_period": true}')
    assert abs(ear["effective_annual_rate_percent"] - 19.5618) < 1e-3
    assert run_tool("payback_period", '{"initial_investment": 16000, "cash_flows": [5000, 5000, 5000, 5000]}')["payback_years"] == 3.2
    assert "error" in run_tool("calculator", '{"expression": "__import__(1)"}')   # no code execution
    assert detect_language("Quelle est la règle de 72 ?") == "fr"
    assert detect_language("What is the rule of 72?") == "en"


def test_misc():
    assert resolve_source("Principles_of_Finance-WEB.pdf", ["principles-finance_-_WEB.pdf"]) \
        == "principles-finance_-_WEB.pdf"
    assert "2/10" in tokenize("terms 2/10, net 30")
    assert classify_question("Is a ratio of 2 good?") == "yes_no"
    assert classify_question("What is the difference between stocks and bonds?") == "comparison"


def test_agent_tools_and_loop():
    """Agent: tools + loop + guard rails, with a scripted 'model' instead of gpt-4o-mini (no API call)."""
    import os
    os.environ["SEND_MODE"] = "real"
    from app import agent, agent_tools as T
    from app.message_tool import send_message
    assert T.calculator("50000 / 20000")["result"] == 2.5
    assert not T.calculator("__import__('os')")["success"]              # simpleeval: no code execution
    assert "capital structure" in T.lookup_term("capital structure")["results"][0]["text"]
    assert T.lookup_term("NPV")["success"] and not T.lookup_term("zzqx")["success"]
    sec = T.get_section("7.4")
    assert sec["success"] and sec["section"].startswith("7.4")
    assert T.get_section("page 229")["results"][0]["page"] == 229
    assert not T.get_section("99.9")["success"]
    cf = T.cash_flow_analysis([-1000, 300, 400, 500], 10)
    assert cf["npv"] == -21.04 and cf["decision"].startswith("reject")
    assert not T.cash_flow_analysis("abc")["success"]
    assert send_message("team", "hi")["status"] == "approval_required"   # approved=False never sends

    script = iter([
        {"thought": "define", "action": "lookup_term", "args": {"term": "current ratio"}},
        {"action": "calculate", "expression": "50000 / 20000"},             # old name + top-level arg
        {"action": "answer", "answer": "The current ratio is current assets / current liabilities [1]. "
                                       "Here it is 2.5."},
    ])
    result = list(agent.run_agent_steps("What is the current ratio of 50000/20000?",
                                        decide=lambda *a: next(script), log=False))[-1]["result"]
    assert [s["action"] for s in result["trajectory"]] == ["lookup_term", "calculator", "answer"]
    assert result["cited_order"] == [1] and result["citations"][0]["cited"]
    assert result["trace"][1].startswith("2. calculator(") and result["stop_reason"] == "answered"

    # send_message: real mode stops and waits for a human; fake mode declines and continues
    script2 = iter([{"action": "send_message", "args": {"recipient": "team-rudra", "message": "Summary..."}}])
    r2 = agent.run_agent("Send a summary to my team", decide=lambda *a: next(script2), log=False)
    assert r2["approval_required"] and r2["pending_message"]["recipient"] == "team-rudra"
    script3 = iter([{"action": "send_message", "args": {"recipient": "x", "message": "y"}},
                    {"action": "answer", "answer": "Done."}])
    r3 = agent.run_agent("q", decide=lambda *a: next(script3), send_mode="fake", log=False)
    assert r3["send_proposals"][0]["status"] == "declined" and not r3["approval_required"]

    # repeated identical calls are blocked, then the agent stops
    r4 = agent.run_agent("x", decide=lambda *a: {"action": "calculator", "args": {"expression": "1+1"}}, log=False)
    assert r4["stop_reason"] == "repeated_calls" and "blocked" in r4["trace"][1]

    # step cap
    counter = iter(range(100))
    r5 = agent.run_agent("x", decide=lambda *a: {"action": "calculator", "args": {"expression": f"1+{next(counter)}"}},
                         log=False)
    assert r5["stop_reason"] == "max_steps" and r5["steps"] == agent.MAX_STEPS


def test_rag_extras():
    """Day-3 extras: proof sentences, section summaries, personal profile (no API call)."""
    from app.evidence import find_evidence
    from app import features, summaries
    from app.tools import run_tool
    cits = [{"n": 1, "text": "Diversification is useful. Investors diversify to reduce unsystematic risk. "
                             "The rule of 72 estimates doubling time."},
            {"n": 2, "text": "At 9% money takes 72/9, or 8, years to double. Other text here."}]
    ev = find_evidence("Investors diversify to reduce unsystematic risk [1]. At 9% it takes 8 years to double [2].", cits)
    assert ev[1][0]["evidence"].startswith("Investors diversify") and "8, years" in ev[2][0]["evidence"]

    import json
    corpus = json.loads((Path(__file__).resolve().parent.parent / "data" / "corpus.json").read_text(encoding="utf-8"))
    groups = summaries.group_pages(corpus)
    assert len(groups) > 50 and any(g["id"] == "7.4" for g in groups)
    assert summaries.is_broad_question("Summarize chapter 7") and not summaries.is_broad_question("What is NPV?")
    summaries._cache = [{"id": "7.4", "chapter": 7, "section": "7.4 Applications of TVM", "source": "a.pdf",
                         "pages": [229, 230], "summary": "Rule of 72 and TVM applications."},
                        {"id": "15.2", "chapter": 15, "section": "15.2 Risk and Return", "source": "a.pdf",
                         "pages": [465], "summary": "Diversification reduces risk."}]
    hits = summaries.search_summaries("Summarize chapter 7")
    assert hits and hits[0]["page"] == 229 and hits[0]["found_by"] == ["summary"]
    assert summaries.search_summaries("main ideas of section 15.2")[0]["page"] == 465
    summaries._cache = None

    prof = {"enabled": True, "savings": 5000, "goal_amount": 20000, "goal_years": 3, "expected_return": 4}
    assert features.profile_active(prof) and features.is_personal("How much should I save?")
    assert "Savings goal: 20000 EUR" in features.profile_instruction(prof)
    plan = run_tool("savings_plan", json.dumps({"goal_amount": 20000, "annual_rate_percent": 4, "years": 3,
                                                "current_savings": 5000}))
    grow = run_tool("savings_growth", json.dumps({"payment_per_period": plan["payment_per_period"],
                                                  "annual_rate_percent": 4, "years": 3, "current_savings": 5000}))
    assert abs(grow["future_value"] - 20000) < 1



def test_agent_v3_check_plan_router():
    """Agent day 3: answer check (grounding + send), plan-then-execute, router. Scripted model, no API."""
    from app import agent, router
    # answer from memory with calculator only -> rejected once, the agent then searches and answers
    script = iter([
        {"action": "calculator", "args": {"expression": "72 / 9"}},
        {"action": "answer", "answer": "About 8 years."},
        {"action": "lookup_term", "args": {"term": "rule of 72"}},
        {"action": "answer", "answer": "About 8 years (72 / 9) [1]."},
    ])
    r = agent.run_agent("Rule of 72 at 9%?", decide=lambda *a: next(script), checks=True, log=False)
    assert [s["action"] for s in r["trajectory"]] == ["calculator", "check", "lookup_term", "answer"]
    assert r["nudges"] == 1 and r["stop_reason"] == "answered" and "check" in r["trace"][1]
    # only ONE nudge: a second ungrounded answer is accepted
    script = iter([{"action": "answer", "answer": "x"}, {"action": "answer", "answer": "y"}])
    r = agent.run_agent("What is beta?", decide=lambda *a: next(script), checks=True, log=False)
    assert r["answer"] == "y" and r["nudges"] == 1
    # asked to send but did not -> nudged to call send_message
    script = iter([
        {"action": "lookup_term", "args": {"term": "diversification"}},
        {"action": "answer", "answer": "Diversification spreads risk [1]."},
        {"action": "send_message", "args": {"recipient": "team", "message": "Diversification spreads risk."}},
        {"action": "answer", "answer": "Prepared a message [1]."},
    ])
    r = agent.run_agent("Summarize diversification and send it to my team", decide=lambda *a: next(script),
                        send_mode="fake", checks=True, log=False)
    assert "send_message" in r["tools_used"] and r["send_proposals"][0]["status"] == "declined"
    # greetings, refusals and cash-flow answers pass the check; checks are off for v1/v2 and scripted runs
    assert agent.answer_check("Hello! What can you help me with?", [], "I can help with finance.") is None
    assert agent.answer_check("Who won the World Cup?", [], agent.REFUSAL_TEXT) is None
    assert agent.answer_check("NPV?", [{"tool": "cash_flow_analysis", "result": {"npv": 1}}], "NPV is 1.") is None
    # math in the head after a search -> "math" check; list markers / section numbers are not math
    book = [{"tool": "search_corpus", "result": {"results": [{"n": 1, "page": 229, "text": "Divide 72 by the rate."}]}}]
    assert agent.answer_check("Rule of 72 at 9%?", book, "72 / 9 = 8 years [1].")[0] == "math"
    assert agent.answer_check("Three reasons?", book, "(1) interest, (2) risk, (3) waiting [1].") is None
    assert agent.answer_check("Rule of 72?", book, "Divide 72 by the rate.")[0] == "citation"
    script = iter([{"action": "answer", "answer": "x"}])
    assert agent.run_agent("What is beta?", decide=lambda *a: next(script), log=False)["nudges"] == 0
    # plan-then-execute: the plan is step 0, not counted as a step, visible in the trace
    script = iter([{"action": "answer", "answer": "Hi!"}])
    r = agent.run_agent("Hello", decide=lambda *a: next(script), plan=True,
                        planner=lambda q, h: ["1. answer directly"], log=False)
    assert r["trajectory"][0]["action"] == "plan" and r["plan"] == ["1. answer directly"]
    assert r["steps"] == 1 and r["trace"][0].startswith("0. plan")
    # router: rule-based, $0
    R = router.route
    assert R("What does the book say about diversification?")["route"] == "rag"
    assert R("What is the time value of money and why is it important?")["route"] == "rag"
    for q in ["Rule of 72: how many years to double at 9%?", "What does section 7.4 cover?",
              "Summarize diversification and send it to my team.", "Hello! What can you help me with?",
              "What is the book's definition of capital structure?",
              "A project costs 1,000 and returns 400 per year for 3 years. NPV at 10%?"]:
        assert R(q)["route"] == "agent", q
    rr = router.rag_as_agent_result("q", {"answer": "a [1]", "citations": [{"n": 1}], "top_rerank": 3.0},
                                    "single lookup", 1.234)
    assert rr["route"] == "rag" and rr["tools_used"] == ["search_corpus"] and rr["latency_s"] == 1.23



def test_agent_extensions_day3b():
    """save_note action (approval + outcome check), adaptive planning, trajectory-judge agreement. No API."""
    import csv
    import tempfile
    from app import agent, note_tool
    # save_note in real mode stops for approval with a preview that lists the sources; nothing written
    script = iter([{"action": "lookup_term", "args": {"term": "rule of 72"}},
                   {"action": "save_note", "args": {"title": "Rule of 72", "content": "Divide 72 by the rate [1]."}}])
    r = agent.run_agent("Explain the rule of 72 and save it as a study note", decide=lambda *a: next(script),
                        send_mode="real", log=False)
    assert r["approval_required"] and r["pending_message"]["kind"] == "note"
    assert "# Rule of 72" in r["pending_message"]["message"]
    # fake mode declines and continues
    script = iter([{"action": "save_note", "args": {"title": "x", "content": "y"}}, {"action": "answer", "answer": "ok"}])
    r = agent.run_agent("save a note", decide=lambda *a: next(script), send_mode="fake", log=False)
    assert r["save_proposals"][0]["status"] == "declined" and r["stop_reason"] == "answered"
    # approved save writes the file and verifies it
    old = note_tool.NOTES_DIR
    note_tool.NOTES_DIR = Path(tempfile.mkdtemp())
    res = note_tool.save_note("Rule of 72", "# Rule of 72\n", approved=True)
    assert res["status"] == "saved" and Path(res["path"]).read_text(encoding="utf-8") == "# Rule of 72\n"
    assert note_tool.save_note("t", "x")["status"] == "approval_required"      # not approved -> nothing
    note_tool.NOTES_DIR = old
    # save intent: answer check + router
    assert agent.wants_save("Summarize NPV and save it as a study note")
    from app import router
    assert router.route("Summarize NPV and export it to a markdown file")["route"] == "agent"
    # action allowlist: an action the user never asked for is blocked (v3 harness)
    script = iter([{"action": "save_note", "args": {"title": "x", "content": "y"}}, {"action": "answer", "answer": "ok"}])
    r = agent.run_agent("What is diversification?", decide=lambda *a: next(script), checks=True, log=False)
    assert r["trajectory"][0]["result"]["status"] == "not_requested" and not r["approval_required"]
    # router v3: a RAG refusal is escalated to the agent
    from app import rag
    old_answer, old_run = rag.answer, agent.run_agent
    rag.answer = lambda *a, **k: {"answer": "I don't have enough information to answer this question.",
                                  "refused": True, "citations": []}
    agent.run_agent = lambda q, h=None, **k: {"trace": ["1. answer"], "latency_s": 1.0, "answer": "ok"}
    rr = router.run_routed("What does the book say about piggy banks?")
    assert rr["route"] == "agent" and "escalated" in rr["route_reason"]
    rag.answer, agent.run_agent = old_answer, old_run
    # adaptive planning: only multi-part questions get a plan
    assert agent.needs_plan("Rule of 72 at 9%? What about at 6%?") and not agent.needs_plan("Rule of 72 at 9%?")
    script = iter([{"action": "answer", "answer": "Hi!"}])
    r = agent.run_agent("Hello", decide=lambda *a: next(script), plan="auto", planner=lambda q, h: ["x"], log=False)
    assert r["plan"] is None and r["settings"]["plan"] == "auto:no"
    # trajectory judge agreement on a hand-labelled CSV
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import trajectory_judge as TJ
    tmp = Path(tempfile.mkdtemp()) / "tj.csv"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "sub_type", "expected_tools", "trace", "judge_verdict", "judge_reason", "human_label"])
        w.writerow(["q1", "", "", "", "sensible", "", "sensible"])
        w.writerow(["q2", "", "", "", "not_sensible", "", "sensible"])
        w.writerow(["q3", "", "", "", "unclear", "", "not_sensible"])
    import readme_tools
    readme_tools_update, TJ.update_section = TJ.update_section, lambda *a: None
    TJ.agreement(str(tmp))
    TJ.update_section = readme_tools_update


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
