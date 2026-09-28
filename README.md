# Personal Finance RAG

## Team
**Team Name:** Rudra

**Members:**
- Somitha Gudivada
- Mekala Naveen Venkata Sai Kumar
- Vijaya Venkata Krishna Rudra
- Srikanth Galipelli

## Project Description

A Retrieval-Augmented Generation application that answers
questions using the Principles of Finance corpus.

## Features

- Corpus-based question answering
- Text chunking
- FAISS vector search
- OpenAI embeddings
- GPT-4o-mini answer generation
- Source and page citations
- Out-of-scope question handling
- Gradio web interface

## Architecture

```text
corpus.json
    ↓
Chunking
    ↓
text-embedding-3-small
    ↓
FAISS
    ↓
Top-10 retrieval
    ↓
GPT-4o-mini
    ↓
Answer + citations
    ↓
Gradio UI
```

## Installation

```bash
uv pip install openai python-dotenv tiktoken faiss-cpu numpy gradio
```

## Environment

Create a `.env` file in the project root:

```ini
OPENAI_API_KEY=your_key_here
```

> **Warning:** Never commit `.env` to Git. It is excluded via `.gitignore`.

## Run

```bash
python app/app.py
```

Then open your browser at:
`http://127.0.0.1:7860`

## Example Questions
- What is the current ratio?
- What is diversification and why do investors practice it?
- What are the reasons a dollar today is worth more than a dollar in the future?

## Out-of-Scope Example

**Question:**
> Who won the 2022 FIFA World Cup?

**Response:**
> I don't have enough information to answer this question.
