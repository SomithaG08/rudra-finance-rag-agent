import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CORPUS_FILE = "data/corpus.json"
INDEX_FILE = "data/my_index.faiss"
CHUNKS_FILE = "data/chunks.json"

CHUNK_SIZE = 500
OVERLAP = 100
BATCH_SIZE = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


def chunk_corpus(corpus):
    chunks = []

    for entry in corpus:
        text_chunks = chunk_text(entry["text"])

        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "page": entry["page"],
                "source": entry["source"]
            })

    return chunks


def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [item.embedding for item in response.data]


def embed_chunks(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        print(
            f"Embedding chunks {i + 1} to "
            f"{min(i + BATCH_SIZE, len(texts))} "
            f"of {len(texts)}..."
        )

        embeddings.extend(get_embeddings(batch))

    return embeddings


def build_index(embeddings):
    vectors = np.array(embeddings).astype("float32")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def main():
    print("Loading corpus...")

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Corpus entries: {len(corpus)}")

    print("Creating chunks...")
    chunks = chunk_corpus(corpus)

    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Building FAISS index...")
    index = build_index(embeddings)

    print("Saving FAISS index...")
    faiss.write_index(index, INDEX_FILE)

    print("Saving chunks...")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print()
    print("===== INDEX BUILD COMPLETE =====")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Chunks file: {CHUNKS_FILE}")
    print("================================")


if __name__ == "__main__":
    main()
