"""
evidence.py - Sentence-level proof for every citation (no API call).

A citation [n] says "this fact comes from passage n", but a passage is ~300 tokens long.
For every sentence of the answer that cites [n], we look inside passage n for the ONE
sentence that shares the most meaningful words (and numbers) with it. That sentence is
the evidence: the UI highlights it in the Sources panel, so a reader can check the claim
in two seconds instead of reading the whole passage.

Scoring (simple and explainable), on words cut to 6 letters (a light stemming):
    score = shared content words + 2 x shared numbers, divided by sqrt(length of the candidate)
Numbers weigh more because "9 years", "72" or "2/10" are the facts people check.
"""

import math
import re

from app.text_utils import split_sentences, tokenize

_CITE = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
_NUM = re.compile(r"^\d+(?:[./]\d+)*$")


def _answer_sentences(answer: str) -> list[str]:
    body = answer.split("\n\n**Sources")[0]
    body = re.sub(r"[*_`#>]", " ", body)                      # markdown symbols
    out = []
    for line in body.splitlines():
        line = line.strip(" -•\t")
        if line:
            out.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line))
    return [s.strip() for s in out if s.strip()]


def _stems(text: str) -> set[str]:
    """Words cut to 6 letters, so 'diversify' and 'diversification' match (numbers kept whole)."""
    return {t if _NUM.match(t) else t[:6] for t in tokenize(text)}


def _score(claim_tokens: set[str], sentence: str) -> tuple[float, int]:
    toks = _stems(sentence)
    shared = claim_tokens & toks
    if not shared:
        return 0.0, 0
    nums = sum(1 for t in shared if _NUM.match(t))
    raw = len(shared) + 2 * nums
    return raw / math.sqrt(len(toks) + 1), len(shared)


def find_evidence(answer: str, citations: list[dict], max_per_passage: int = 2) -> dict[int, list[dict]]:
    """
    Returns {n: [{"claim": answer sentence, "evidence": passage sentence, "score": float}, ...]}
    for every passage n cited in the answer.
    """
    by_n = {c["n"]: c for c in citations}
    sentences_cache: dict[int, list[str]] = {}
    found: dict[int, list[dict]] = {}
    for sent in _answer_sentences(answer):
        nums = []
        for m in _CITE.finditer(sent):
            nums += [int(x) for x in re.split(r"[,;]\s*", m.group(1))]
        if not nums:
            continue
        claim = re.sub(r"\s+([.,;:!?])", r"\1", _CITE.sub(" ", sent)).strip()
        claim_tokens = _stems(claim)
        if len(claim_tokens) < 2:
            continue
        for n in dict.fromkeys(nums):
            c = by_n.get(n)
            if not c:
                continue
            cands = sentences_cache.setdefault(n, split_sentences(c["text"]))
            best, best_score, best_shared = None, 0.0, 0
            for cand in cands:
                s, shared = _score(claim_tokens, cand)
                if s > best_score:
                    best, best_score, best_shared = cand, s, shared
            if best and best_shared >= 2:
                items = found.setdefault(n, [])
                if best not in [i["evidence"] for i in items] and len(items) < max_per_passage:
                    items.append({"claim": claim, "evidence": best, "score": round(best_score, 3)})
    return found
