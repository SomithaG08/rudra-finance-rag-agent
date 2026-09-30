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
    """Agent: tools + loop, with a scripted 'model' instead of gpt-4o-mini (no API call)."""
    from app import agent
    assert agent.calculator("50000 / 20000")["result"] == 2.5
    assert not agent.calculator("__import__('os')")["success"]          # no code execution
    assert agent.finance_tool("rule_of_72", {"rate_percent": 8})["result"]["years_rule_of_72"] == 9.0
    corpus = agent._corpus()
    page = next(e for e in corpus if len(e["text"]) > 200)
    q = agent.quote_exact("Principles of Finance", page["page"])      # title -> real file name
    assert q["success"] and q["text"] == page["text"]
    assert not agent.quote_exact(page["source"], 99999)["success"]
    assert agent.message_tool("team", "hi")["status"] == "approval_required"

    script = iter([
        {"action": "search", "query": "current ratio current assets current liabilities"},
        {"action": "calculate", "expression": "50000 / 20000"},
        {"action": "answer", "answer": "The current ratio is current assets / current liabilities [1]. "
                                       "Here it is 2.5."},
    ])
    events = list(agent.run_agent_steps("What is the current ratio of 50000/20000?",
                                        decide=lambda *a: next(script)))
    result = events[-1]["result"]
    assert [s["action"] for s in result["trajectory"]] == ["search", "calculate", "answer"]
    assert result["cited_order"] == [1] and result["citations"][0]["cited"]
    assert not result["approval_required"]

    # send_message must stop the agent and wait for a human
    script2 = iter([{"action": "send_message", "recipient": "team-rudra", "message": "Summary..."}])
    r2 = list(agent.run_agent_steps("Send a summary to my team", decide=lambda *a: next(script2)))[-1]["result"]
    assert r2["approval_required"] and r2["pending_message"]["recipient"] == "team-rudra"

    # step cap
    loop = list(agent.run_agent_steps("x", decide=lambda *a: {"action": "calculate", "expression": "1+1"}))
    assert loop[-1]["result"]["trajectory"][-1]["action"] == "max_steps_reached"


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


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
