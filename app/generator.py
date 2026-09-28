import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question, retrieved_chunks):
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Text: {chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """
You are a helpful assistant answering questions about a personal finance
and financial analysis textbook.

Answer the user's question using ONLY the information contained in the
provided context.

Important instructions:

1. Carefully examine ALL retrieved chunks before deciding whether the
   question can be answered.

2. If the question asks you to compare, connect, or cross-reference
   information from different parts of the textbook, combine the relevant
   information from multiple retrieved chunks.

3. Do not require the answer to appear in one single chunk. The answer may
   be constructed from several chunks.

4. If the context contains enough information to answer the question,
   provide the answer clearly and explain the relevant points.

5. Do not use outside knowledge.

6. If the context genuinely does not contain enough information to answer
   the question, say exactly:
   "I don't have enough information to answer this question."

7. Always include the relevant source and page number(s) in the answer.

8. Do not invent facts, numbers, or citations.
"""

    user_prompt = f"""
Retrieved textbook context:

{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content
