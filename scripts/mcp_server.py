"""
mcp_server.py - The corpus tools exposed as an MCP server (Agentic AI optional extension).

Any MCP client (Claude Desktop, Cursor, the MCP Inspector, another agent) can call our tools:
    search_corpus(query)                      reranked search of Principles of Finance
    lookup_term(term)                         the book's glossary
    get_section(section)                      a chapter / section / page / "contents"
    cash_flow_analysis(cash_flows, rate)      NPV, IRR, payback, discounted payback
    calculator(expression)                    simpleeval arithmetic
send_message is NOT exposed: an action needs the human approval of our UI.

    uv pip install mcp                                    # works with mcp 1.x (FastMCP) and 2.x (MCPServer)
    python scripts/mcp_server.py                          # stdio server
    npx @modelcontextprotocol/inspector python scripts/mcp_server.py   # test it in a browser

Claude Desktop config (claude_desktop_config.json):
    {"mcpServers": {"finance-book": {"command": "python",
                                     "args": ["<repo>/scripts/mcp_server.py"]}}}
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:                                                    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as Server  # noqa: E402
except ImportError:                                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server  # noqa: E402

from app import agent_tools as T  # noqa: E402

mcp = Server("personal-finance-book")


def _text(result: dict) -> str:
    """Compact JSON for the client (long passages cut)."""
    r = dict(result)
    if "results" in r:
        r["results"] = [{"n": p["n"], "page": p["page"], "section": p.get("section"),
                         "rerank": p.get("rerank"), "text": p["text"][:1500]} for p in r["results"]]
    return json.dumps(r, ensure_ascii=False, default=str)


@mcp.tool()
def search_corpus(query: str) -> str:
    """Search the OpenStax 'Principles of Finance' textbook (hybrid BM25 + vectors, cross-encoder
    reranked). Returns the top 3 passages with page numbers, or 'No matching passages found'."""
    return _text(T.search_corpus(query))


@mcp.tool()
def lookup_term(term: str) -> str:
    """Definition of a finance term from the textbook's end-of-chapter Key Terms."""
    return _text(T.lookup_term(term))


@mcp.tool()
def get_section(section: str) -> str:
    """A section ('7.4'), chapter ('chapter 7'), page ('page 229') or 'contents' of the textbook."""
    return _text(T.get_section(section))


@mcp.tool()
def cash_flow_analysis(cash_flows: list[float], rate_percent: float | None = None) -> str:
    """NPV, IRR, payback, discounted payback and profitability index. cash_flows[0] = investment."""
    return _text(T.cash_flow_analysis(cash_flows, rate_percent))


@mcp.tool()
def calculator(expression: str) -> str:
    """Exact arithmetic (numbers, + - * / ** % parentheses, sqrt, log, exp, round)."""
    return _text(T.calculator(expression))


if __name__ == "__main__":
    mcp.run()
