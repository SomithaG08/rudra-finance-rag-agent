import json
import re
from collections import Counter
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup


SOURCES = Path("data/sources")
OUTPUT_DIR = Path("data")
MIN_CHARS = 50


def clean_text(text: str) -> str:
    """Clean extracted text while preserving readable paragraphs."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Fix common PDF line-break problems:
    # "finan-\ncial" -> "financial"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace single newlines inside sentences with spaces
    text = re.sub(r"(?<![.!?:])\n(?!\n)", " ", text)

    # Keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove whitespace around lines
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


def extract_pdf(path: Path):
    """Extract one entry per PDF page."""

    doc = pymupdf.open(path)

    for page_number, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        yield page_number, text

    doc.close()


def extract_text_file(path: Path):
    """Extract plain text or Markdown files."""

    raw_text = path.read_text(encoding="utf-8")
    yield 1, clean_text(raw_text)


def extract_html(path: Path):
    """Extract readable text from HTML."""

    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that normally contain navigation/noise
    for element in soup(
        ["script", "style", "nav", "aside", "footer", "header"]
    ):
        element.decompose()

    text = soup.get_text("\n")
    yield 1, clean_text(text)


def main():
    entries = []
    near_empty = 0
    source_files = set()

    if not SOURCES.exists():
        print(f"ERROR: Source directory not found: {SOURCES}")
        return

    for path in sorted(SOURCES.iterdir()):

        if not path.is_file():
            continue

        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = extract_pdf(path)

        elif ext in {".txt", ".md"}:
            pages = extract_text_file(path)

        elif ext in {".html", ".htm"}:
            pages = extract_html(path)

        else:
            print(f"Skipping unsupported file: {path.name}")
            continue

        source_files.add(path.name)

        for page, text in pages:

            char_count = len(text)

            if char_count < MIN_CHARS:
                near_empty += 1

            entries.append(
                {
                    "source": path.name,
                    "page": page,
                    "char_count": char_count,
                    "text": text,
                }
            )

    # Save complete corpus
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = OUTPUT_DIR / "corpus.json"
    sample_path = OUTPUT_DIR / "sample.json"

    corpus_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # First 10 entries
    sample_path.write_text(
        json.dumps(entries[:10], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Statistics
    total_characters = sum(entry["char_count"] for entry in entries)

    average_characters = (
        total_characters / len(entries)
        if entries
        else 0
    )

    print()
    print("===== CORPUS EXTRACTION STATS =====")
    print(f"Source files: {len(source_files)}")
    print(f"Total entries: {len(entries)}")
    print(f"Total characters: {total_characters}")
    print(f"Average characters/entry: {average_characters:.2f}")
    print(f"Near-empty entries (<{MIN_CHARS} chars): {near_empty}")
    print("===================================")
    print()
    print(f"Created: {corpus_path}")
    print(f"Created: {sample_path}")


if __name__ == "__main__":
    main()
