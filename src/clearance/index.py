"""Keyword retrieval (BM25) with permissions applied before ranking.

Where the permission check sits decides whether it leaks:

* `filter="before"` (the design): only chunks the asker may read are scored at all. A
  restricted document cannot crowd allowed ones out of the top k, and its existence cannot
  be inferred from a short or empty result list.
* `filter="after"` (the common shortcut, kept to measure it): rank everything, then drop what
  the asker may not see. The top k shrinks, sometimes to nothing, exactly when restricted
  documents match best, which itself tells the asker something.
* `filter="none"`: no permission check at all, to measure what an unprotected assistant
  would expose.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from clearance.chunking import Chunk

TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = (
    "a an and are as at be by can do does for from has have how i if in is it its of "
    "on or our should that the their there this to was we what when where which who "
    "why will with you your gitlab team member members"
)
STOP = frozenset(STOP_WORDS.split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Bm25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokens(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._postings: dict[str, list[int]] = defaultdict(list)
        for i, tf in enumerate(self._tf):
            for t in tf:
                self._postings[t].append(i)

    def _score(self, i: int, query: list[str]) -> float:
        tf, length = self._tf[i], self._len[i]
        s = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                s += (
                    self._idf[t]
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * length / self._avg))
                )
        return s

    def search(
        self, query: str, groups: frozenset[str], k: int = 5, filter: str = "before"
    ) -> list[Hit]:
        q = tokens(query)
        candidates = {i for t in q for i in self._postings.get(t, ())}
        if filter == "before":
            candidates = {i for i in candidates if self.chunks[i].group in groups}
        ranked = sorted(
            (Hit(self.chunks[i], self._score(i, q)) for i in candidates),
            key=lambda h: (-h.score, h.chunk.id),
        )[:k]
        if filter == "after":
            ranked = [h for h in ranked if h.chunk.group in groups]
        return ranked
