# Personal Finance Assistant: RAG + Agentic AI

A Retrieval-Augmented Generation (RAG) assistant and a tool-using agent, written from scratch in Python (no LangChain / LlamaIndex), that answer questions about the OpenStax textbook *Principles of Finance*. Every answer shows **all the sources it used** as clickable citations that open the PDF at the right page.

Both run in one Gradio app: the **Chat** tab is the RAG pipeline, the **Agent (tools)** tab is the agent.

## Team Rudra

| Member | Main contribution |
|---|---|
| Somitha Gudivada | Retrieval evaluation: FAISS index, reranking config, ablation study, threshold sweep, eval metrics |
| Mekala Naveen Venkata Sai Kumar | Retrieval agent: hybrid BM25 + FAISS retrieval, RRF, ReAct agent loop, router, action tools |
| Vijaya Venkata Krishna Rudra | Data ingestion: sentence chunking, section summaries, finance tools |
| Srikanth Galipelli | UI frontend: Gradio app (7 tabs), citations, step trace, approval gate, MCP server |

## Features

**RAG pipeline**
- Page-by-page PDF extraction and sentence chunking with section titles and page numbers
- **Hybrid search**: BM25 + FAISS (`text-embedding-3-small`) merged with Reciprocal Rank Fusion
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) and a **refusal threshold** for out-of-scope questions
- `gpt-4o-mini` answers with numbered citations, max 2 chunks per page for source diversity
- Follow-up questions with query rewriting, HyDE, query expansion, French support

**Citations and trust**
- Inline `[n]` links to the cited PDF page, a Sources panel, and the exact supporting sentence highlighted
- Citation check, confidence badge, prompt-injection hardening (`<context>` wrapping)

**Interface**
- Streaming chat with memory, answer styles, quiz tab, personal savings profile, export (.md / .pdf), analytics, offline mode

**Why BM25?** The book is full of acronyms and exact terms (DDM, EAR, WACC) that embeddings blur. BM25 matches them exactly, vectors match paraphrases, and RRF keeps both.

## Agent

A plain Python loop: `gpt-4o-mini` returns one JSON action per step, max 6 steps.

| Tool | What it does |
|---|---|
| `search_corpus` | hybrid search + rerank; refuses below the threshold |
| `calculator` | exact arithmetic with simpleeval |
| `lookup_term` | the book's own glossary definition (383 terms) |
| `get_section` | a section, chapter, page or the table of contents |
| `cash_flow_analysis` | NPV, IRR, payback, profitability index in one call |
| `send_message` | Slack / Teams message, **only after human approval** |

Guard rails: limited retries, blocked repeated calls, step cap, answer check, and a numbered step trace for every run. An optional rule-based router sends simple lookups to RAG and multi-step questions to the agent.

## Architecture

```text
PDF ─► extract ─► chunk ─► embed ─► FAISS index
                                        │
question ─► (rewrite) ─► BM25 + FAISS ─► RRF ─► rerank ─► threshold? refuse
         ─► gpt-4o-mini + <context> passages ─► answer with [n] ─► citation check ─► Gradio UI

agent: question ─► gpt-4o-mini ─► JSON action ─► tool ─► ... (max 6) ─► answer + trace
```

## Installation

```bash
uv venv --python 3.12
# Windows: .venv\Scripts\activate   |   macOS / Linux: source .venv/bin/activate
uv pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```ini
OPENAI_API_KEY=sk-...
```

## Run

```bash
python scripts/extract.py        # extract the PDF (data/ is git-ignored)
python app/build_index.py        # build the index (asks YES before spending)
python app/app.py                # open http://127.0.0.1:7860
```

Evaluation:

```bash
python tests/test_offline.py
python scripts/ablation.py
python scripts/threshold_sweep.py
python scripts/eval_runner.py --judge
python scripts/agent_eval.py --compare-rag
```

## Results

<!-- ABLATION:START -->
### Retrieval ablation (59 questions, top-k = 8)

| Version | Recall@1 | Recall@5 | MRR | Full recall |
|---|---|---|---|---|
| Vector only | 86% | 97% | 0.923 | 86% |
| + reranking | 81% | 100% | 0.905 | 92% |
| + BM25 (hybrid, RRF) | 81% | 100% | 0.901 | 92% |
<!-- ABLATION:END -->

**Refusal threshold:** at a reranker score of -4.223, the assistant refuses 5 of 6 out-of-scope questions with 0 wrong refusals out of 41.

<!-- AGENT_EVAL:START -->
### Agent (17 questions)

| | Correct | Avg steps | Latency (s) | Cost |
|---|---|---|---|---|
| RAG pipeline | - | 0.8 calls | 2.4 | $0.0058 |
| Agent | 100% | 2.8 | 5.9 | $0.0151 |
<!-- AGENT_EVAL:END -->

**Prompt injection:** 0 of 3 injected instructions were obeyed, both in documents and against actions.

## Example questions

| # | Tab | Question | What it shows |
|---|---|---|---|
| 1 | Chat | What does DDM stand for, and which DDM is the most common? | hybrid search (BM25 finds the acronym), rerank scores, several citations |
| 2 | Chat | According to the rule of 72, how long does money take to double at 9%? *then* "And at 6%?" | follow-up + query rewriting |
| 3 | Chat | Who won the 2022 FIFA World Cup? | refusal from the reranker threshold, no sources |
| 4 | Chat (injection test ON) | What is diversification? | injection chunk retrieved but not obeyed |
| 5 | Agent | Define 'capital structure' using the book's glossary, then explain why it matters. | `lookup_term` + `search_corpus` |
| 6 | Agent | What is the NPV at 10% of a project costing 1,000 that returns 400 per year for 3 years? Should we accept it? | `cash_flow_analysis` |
| 7 | Agent | What does section 7.4 cover? Give its page range. | `get_section` |
| 8 | Agent | Summarize what the book says about diversification and send it to my team on Slack. | `send_message` → approval box → outcome check |
| 9 | Agent (injection ON, fake send ON) | What is diversification? | the document asks for a message; the agent ignores it |

## Source

The corpus is [*Principles of Finance*](https://openstax.org/details/books/principles-finance) by Julie Dahlquist and Rainford Knight, published by OpenStax (Rice University) under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
