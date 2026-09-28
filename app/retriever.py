import json
import os

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INDEX_FILE = "data/my_index.faiss"
CHUNKS_FILE = "data/chunks.json"


# Load the saved FAISS index and chunks
index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def retrieve(query, k=5):
    query_embedding = get_embedding(query)

    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Normalize so inner product represents cosine similarity
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []

    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue

        result = dict(chunks[idx])
        result["score"] = float(scores[0][i])
        results.append(result)

    return results


if __name__ == "__main__":
    question = input("Enter your question: ")

    results = retrieve(question)

    print("\n===== RETRIEVED CHUNKS =====")

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. Similarity: {result['score']:.3f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}")

    print("\n============================")

