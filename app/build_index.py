"""
build_index.py - Run ONCE (or after the corpus changes) to build the search index.

    python app/build_index.py                  # sentence chunking (recommended)
    python app/build_index.py --strategy fixed # Day-1 chunking, for comparison
    python app/build_index.py --dry-run        # no API call: chunk stats + cost estimate

EMBEDDING POLICY (professor's request: re-embed only when needed, never automatically)
  * Nothing else in the project embeds the corpus: the app, the eval scripts and the
    tests only LOAD data/my_index.faiss. Only this script embeds, and only when run by hand.
  * Guard 1: if corpus + chunking settings + model are unchanged (fingerprint in
    data/build_info.json), the script stops immediately - zero API calls.
  * Guard 2: chunks already embedded once are taken from data/embedding_cache.npz;
    only new / changed chunks are sent to the API.
  * Guard 3: before spending, it prints the token count and asks you to type YES.
  * One person builds, then shares my_index.faiss + chunks.json on Teams.

Steps
  1. read data/corpus.json (made by scripts/extract.py)
  2. chunk every page (app/chunking.py)
  3. embed every chunk with text-embedding-3-small
     -> an embedding cache (data/embedding_cache.npz) remembers every chunk
        already embedded, so re-running after adding a document only pays for
        the NEW chunks (incremental "document update pipeline")
  4. normalise vectors and store them in a FAISS IndexFlatIP
     (inner product of unit vectors = cosine similarity)
  5. save data/my_index.faiss, data/chunks.json and data/build_info.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import faiss
import numpy as np

from app.chunking import chunk_corpus
from app.config import (BUILD_INFO_FILE, CHUNK_OVERLAP_TOKENS, CHUNK_STRATEGY, CHUNK_TOKENS,
                        CHUNKS_FILE, CORPUS_FILE, EMBED_BATCH_SIZE, EMBED_CACHE_FILE, EMBED_MODEL,
                        INDEX_FILE, PRICES, has_api_key)
from app.text_utils import count_tokens


def text_hash(text: str) -> str:
    """Fingerprint of a chunk: same text -> same hash -> reuse the cached vector."""
    return hashlib.sha1(f"{EMBED_MODEL}|{text}".encode("utf-8")).hexdigest()


def load_cache() -> dict[str, np.ndarray]:
    if not EMBED_CACHE_FILE.exists():
        return {}
    data = np.load(EMBED_CACHE_FILE)
    return {h: v for h, v in zip(data["hashes"], data["vectors"])}


def save_cache(cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    hashes = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values())).astype("float32")
    np.savez_compressed(EMBED_CACHE_FILE, hashes=hashes, vectors=vectors)


def embed_all(chunks: list[dict]) -> np.ndarray:
    """Return one vector per chunk, calling the API only for chunks not in the cache."""
    from app.llm import embed_texts

    cache = load_cache()
    hashes = [text_hash(c["text"]) for c in chunks]
    missing = [i for i, h in enumerate(hashes) if h not in cache]
    print(f"Embedding cache: {len(chunks) - len(missing)} reused, {len(missing)} to embed.")

    for start in range(0, len(missing), EMBED_BATCH_SIZE):
        batch_ids = missing[start:start + EMBED_BATCH_SIZE]
        texts = [chunks[i]["text"] for i in batch_ids]
        for attempt in range(3):                       # simple retry on network errors
            try:
                vectors = embed_texts(texts, operation="embed_chunks")
                break
            except Exception as exc:                   # noqa: BLE001
                if attempt == 2:
                    save_cache(cache)                  # keep what we already paid for
                    raise
                print(f"  retrying after error: {exc}")
                time.sleep(3 * (attempt + 1))
        for i, v in zip(batch_ids, vectors):
            cache[hashes[i]] = v
        done = min(start + EMBED_BATCH_SIZE, len(missing))
        print(f"  embedded {done}/{len(missing)}")
        if done % (EMBED_BATCH_SIZE * 10) == 0:
            save_cache(cache)                          # checkpoint

    save_cache(cache)
    return np.stack([cache[h] for h in hashes]).astype("float32")


def fingerprint(corpus: list[dict], strategy: str) -> str:
    """Hash of (corpus text + chunking settings + model). Same fingerprint = same index."""
    h = hashlib.sha1()
    h.update(f"{EMBED_MODEL}|{strategy}|{CHUNK_TOKENS}|{CHUNK_OVERLAP_TOKENS}".encode())
    for e in corpus:
        h.update(f"{e.get('source')}|{e.get('page')}|{e.get('text')}".encode("utf-8"))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the FAISS index from data/corpus.json")
    parser.add_argument("--strategy", choices=["sentence", "fixed"], default=CHUNK_STRATEGY)
    parser.add_argument("--dry-run", action="store_true", help="only chunk + estimate cost")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation before paying")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the corpus and settings did not change")
    args = parser.parse_args()

    if not CORPUS_FILE.exists():
        sys.exit(f"ERROR: {CORPUS_FILE} not found. Run  python scripts/extract.py  first.")

    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
    print(f"Corpus: {len(corpus)} pages from {len({e['source'] for e in corpus})} source file(s)")

    # ---- Guard 1: nothing changed -> nothing to do (no API call at all) --------------------
    fp = fingerprint(corpus, args.strategy)
    if BUILD_INFO_FILE.exists() and INDEX_FILE.exists() and CHUNKS_FILE.exists() and not args.force:
        info = json.loads(BUILD_INFO_FILE.read_text(encoding="utf-8"))
        if info.get("fingerprint") == fp:
            print("Index is up to date (same corpus, same chunking, same model). Nothing to embed.")
            print("Use --force only if the index files are damaged.")
            return

    chunks = chunk_corpus(corpus, strategy=args.strategy)
    sizes = [c["char_count"] for c in chunks]

    # ---- Guard 2: only chunks never embedded before are paid (embedding cache) -------------
    cache = load_cache()
    new_chunks = [c for c in chunks if text_hash(c["text"]) not in cache]
    tokens = sum(count_tokens(c["text"]) for c in new_chunks)
    est = tokens * PRICES[EMBED_MODEL]["input"] / 1_000_000
    print(f"Chunks ({args.strategy}): {len(chunks)}  | avg {sum(sizes) // len(sizes)} chars "
          f"(min {min(sizes)}, max {max(sizes)})")
    print(f"Already embedded (cache): {len(chunks) - len(new_chunks)}  | to embed now: {len(new_chunks)}"
          f"  | ~{tokens:,} tokens  | est. cost ${est:.4f}")

    if args.dry_run:
        print("Dry run: nothing embedded, nothing saved.")
        return
    if new_chunks and not has_api_key():
        sys.exit("ERROR: OPENAI_API_KEY missing in .env - cannot embed.")

    # ---- Guard 3: explicit confirmation before spending ------------------------------------
    if new_chunks and not args.yes:
        answer = input(f"This will send ~{tokens:,} tokens to the embeddings API. Type YES to continue: ")
        if answer.strip() != "YES":
            print("Cancelled. Nothing embedded.")
            return

    vectors = embed_all(chunks)
    faiss.normalize_L2(vectors)                        # unit length -> inner product = cosine
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    BUILD_INFO_FILE.write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": fp,
        "strategy": args.strategy,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "chunks": len(chunks),
        "tokens_embedded_this_build": tokens,
        "sources": sorted({c["source"] for c in chunks}),
        "embedding_model": EMBED_MODEL,
    }, indent=2), encoding="utf-8")

    print("\n===== INDEX BUILD COMPLETE =====")
    print(f"Vectors in index : {index.ntotal}")
    print(f"Index file       : {INDEX_FILE}")
    print(f"Chunks file      : {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
