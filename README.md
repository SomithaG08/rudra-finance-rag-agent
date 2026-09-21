# Personal Finance & Financial Analysis for Students

## Team

**Team Name:** Rudra

**Members:**
- Mekala Naveen Venkata Sai Kumar
- Vijaya Venkata Krishna Rudra
- Srikanth Galipelli

## Assignment

Assignment 0 — Corpus Preparation

The goal of this assignment is to prepare a clean text corpus for use in a Retrieval-Augmented Generation (RAG) system.

## Source

The corpus was extracted from **OpenStax — Principles of Finance**.

The source PDF is stored locally in `data/sources/`.

## Project Structure

```text
personal-finance-corpus/
├── README.md
├── .gitignore
├── scripts/
│   ├── extract.py
│   └── EXTRACTION_REPORT.md
├── data/
│   ├── sources/
│   ├── corpus.json
│   └── sample.json
└── app/
```

## Extraction

The extraction script processes PDF documents page by page using PyMuPDF.

It generates `data/corpus.json` and `data/sample.json` and reports extraction statistics.

## Results

- Source files: 1
- Total entries: 629
- Total characters: 1,614,510
- Average characters per entry: 2,566.79
- Near-empty entries: 3

## Data Policy

The `data/` directory is excluded from Git using `.gitignore`.

The corpus files will be submitted separately through the required private Teams channel.

## Environment

- Python 3.12
- uv
- PyMuPDF
- pdfplumber
- BeautifulSoup4
