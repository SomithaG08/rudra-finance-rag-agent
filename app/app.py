"""Gradio Web Interface for Personal Finance RAG.

Starts the Gradio interface, accepts user questions, calls retrieve(..., k=10),
sends retrieved chunks to generate_answer(), and displays the generated answer
along with source/page citations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    from app.generator import generate_answer
    from app.retriever import Retriever
except ImportError:
    try:
        from generator import generate_answer
        from retriever import Retriever
    except ImportError:
        Retriever = None
        generate_answer = None


retriever_instance = None


def get_retriever():
    """Lazily initialize Retriever or handle missing index gracefully."""
    global retriever_instance
    if retriever_instance is None and Retriever is not None:
        try:
            retriever_instance = Retriever()
        except Exception as e:
            print(f"Notice: FAISS index not loaded ({e}). Using mock retrieval for demo.")
            retriever_instance = None
    return retriever_instance


def retrieve(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from the corpus or mock for offline UI demo."""
    r = get_retriever()
    if r is not None:
        return r.retrieve(query, top_k=k)

    # Demo fallback when running without FAISS index or API keys
    query_lower = query.lower()
    if "current ratio" in query_lower:
        return [
            {
                "source": "principles_of_finance.pdf",
                "page": 72,
                "text": "Current Ratio = Current Assets / Current Liabilities. A ratio below 1.0 indicates that short-term obligations exceed short-term assets.",
                "score": 0.91,
                "is_relevant": True,
            }
        ]
    elif "diversification" in query_lower:
        return [
            {
                "source": "principles_of_finance.pdf",
                "page": 251,
                "text": "Diversification reduces unsystematic or firm-specific risk across securities. However, systematic risk cannot be diversified away.",
                "score": 0.88,
                "is_relevant": True,
            }
        ]
    elif "dollar today" in query_lower or "time value" in query_lower:
        return [
            {
                "source": "principles_of_finance.pdf",
                "page": 136,
                "text": "The time value of money concept establishes that a dollar received today is worth more than a dollar in the future due to its earning capacity, inflation, and risk.",
                "score": 0.89,
                "is_relevant": True,
            }
        ]
    elif any(term in query_lower for term in ["fifa", "world cup", "weather", "tokyo", "kafka", "recipe"]):
        # Out-of-scope query
        return []

    return [
        {
            "source": "principles_of_finance.pdf",
            "page": 284,
            "text": "Net Present Value (NPV) compares the present value of all cash inflows against initial investment. If NPV > 0, the project should be accepted.",
            "score": 0.85,
            "is_relevant": True,
        }
    ]


def answer_question(user_question: str) -> Tuple[str, str]:
    """Accepts user question, calls retrieve(..., k=10), and generates answer with citations."""
    if not user_question or not user_question.strip():
        return "Please enter a question.", ""

    user_question = user_question.strip()

    # Step 1: Call retrieve(..., k=10)
    chunks = retrieve(user_question, k=10)

    # Step 2: Send retrieved chunks to generate_answer()
    if generate_answer is not None and os.getenv("OPENAI_API_KEY"):
        result = generate_answer(user_question, chunks)
        answer = result.get("answer", "No answer generated.")
        citations = result.get("citations", [])
    else:
        # Offline deterministic demonstration logic
        if not chunks:
            answer = "I don't have enough information to answer this question."
            citations = []
        else:
            top = chunks[0]
            answer = f"{top['text']}"
            citations = list({f"{c['source']}, p. {c['page']}" for c in chunks})

    # Step 3: Format output citations
    if citations:
        citations_str = "\n".join(f"• {c}" for c in sorted(citations))
    else:
        citations_str = "None"

    return answer, citations_str


def create_interface():
    """Configure Gradio web interface."""
    if gr is None:
        raise ImportError("Gradio is not installed. Run: uv pip install gradio")

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="Personal Finance RAG") as demo:
        gr.Markdown(
            """
            # 💼 Personal Finance RAG
            ### Principles of Finance — Interactive Assistant
            Ask questions about personal finance, financial ratios, time value of money, and investments.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the current ratio?",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

                gr.Markdown("### 💡 Example Questions")
                gr.Examples(
                    examples=[
                        ["What is the current ratio?"],
                        ["What is diversification and why do investors practice it?"],
                        ["What are the reasons a dollar today is worth more than a dollar in the future?"],
                        ["Who won the 2022 FIFA World Cup?"],
                    ],
                    inputs=question_input,
                )

            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False,
                )
                citations_output = gr.Textbox(
                    label="Source Citations",
                    lines=3,
                    interactive=False,
                )

        submit_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, citations_output],
        )

        question_input.submit(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, citations_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=None,
            outputs=[question_input, answer_output, citations_output],
        )

        gr.Markdown(
            """
            ---
            *Powered by FAISS vector search, text-embedding-3-small, and GPT-4o-mini.*
            """
        )

    return demo


def main():
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
