"""
bm25.py - Keyword search (BM25 / Okapi), written from scratch with numpy.

Why we need it: vector search understands meaning ("dividend discount model"
~ "valuing a stock from its dividends") but is weak on exact short terms such
as acronyms (DDM, WACC, LLC), tickers (DAL) or codes ("2/10, net 30").
BM25 scores a chunk higher when it contains the rare words of the question.

score(chunk, query) = sum over query words w of
    IDF(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(chunk) / avg_len))

  tf   = how many times w appears in the chunk
  IDF  = log(1 + (N - df + 0.5) / (df + 0.5))  -> rare words weigh more
  k1   = 1.5 (saturation of repeated words), b = 0.75 (length normalisation)
"""

import math
from collections import Counter, defaultdict

import numpy as np

from app.text_utils import tokenize


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(texts)
        self.doc_len = np.zeros(self.n_docs, dtype="float32")
        # inverted index: word -> list of (doc_id, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, text in enumerate(texts):
            words = tokenize(text)
            self.doc_len[doc_id] = len(words)
            for word, tf in Counter(words).items():
                self.postings[word].append((doc_id, tf))

        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.idf = {
            w: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for w, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every chunk for this query (0 for chunks without any query word)."""
        scores = np.zeros(self.n_docs, dtype="float32")
        for word in set(tokenize(query)):
            if word not in self.postings:
                continue
            idf = self.idf[word]
            for doc_id, tf in self.postings[word]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def coverage(self, query: str) -> float:
        """
        Share of the question's "information" (IDF weight) found in the corpus vocabulary.
        'Who won the 2022 FIFA World Cup?' -> 'fifa' and 'cup' never occur -> low coverage.
        Used as an out-of-scope signal in offline mode (no embeddings available).
        """
        words = set(tokenize(query))
        if not words:
            return 0.0
        max_idf = math.log(1 + (self.n_docs + 0.5) / 0.5)      # IDF of an unseen word
        total = sum(self.idf.get(w, max_idf) for w in words)
        known = sum(self.idf[w] for w in words if w in self.idf)
        return known / total if total else 0.0

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk_id, score) pairs, best first."""
        s = self.scores(query)
        if k >= len(s):
            order = np.argsort(-s)
        else:
            top = np.argpartition(-s, k)[:k]
            order = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]
