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

Filtering before ranking is not enough on its own. BM25 weighs each query word by how rare
it is in the corpus, and the document count, word frequencies and average length all come
from the corpus. Computed over everything, they include restricted documents, so adding a
restricted page that mentions "settlement" changes the scores, and sometimes the order, of
what everyone else sees. Someone could probe for words in pages they cannot read. With
`scoped=True` (the design) those statistics come only from the groups the asker may read,
so a restricted page has no effect on their results at all. Each chunk belongs to exactly
one group, so per-group counts add up exactly and nothing is rebuilt per user.
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
        self._df: dict[str, Counter[str]] = defaultdict(Counter)  # group -> term -> chunks
        self._n: Counter[str] = Counter()  # group -> chunks
        self._words: Counter[str] = Counter()  # group -> total words
        self._postings: dict[str, list[int]] = defaultdict(list)
        for i, (chunk, tf) in enumerate(zip(chunks, self._tf, strict=True)):
            self._df[chunk.group].update(tf.keys())
            self._n[chunk.group] += 1
            self._words[chunk.group] += self._len[i]
            for t in tf:
                self._postings[t].append(i)
        self.groups = frozenset(self._n)

    def _weights(self, query: list[str], groups: frozenset[str]) -> tuple[dict[str, float], float]:
        """Inverse document frequency of each query word, and the average chunk length,
        computed over the chunks in `groups` only."""
        n = sum(self._n[g] for g in groups)
        words = sum(self._words[g] for g in groups)
        idf = {}
        for t in set(query):
            d = sum(self._df[g][t] for g in groups)
            idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))
        return idf, words / max(1, n)

    def _score(self, i: int, query: list[str], idf: dict[str, float], avg: float) -> float:
        tf, length = self._tf[i], self._len[i]
        s = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                s += (
                    idf[t]
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * length / avg))
                )
        return s

    def search(
        self,
        query: str,
        groups: frozenset[str],
        k: int = 5,
        filter: str = "before",
        scoped: bool = True,
    ) -> list[Hit]:
        q = tokens(query)
        candidates = {i for t in q for i in self._postings.get(t, ())}
        if filter == "before":
            candidates = {i for i in candidates if self.chunks[i].group in groups}
        stats_from = groups & self.groups if filter == "before" and scoped else self.groups
        idf, avg = self._weights(q, stats_from)
        ranked = sorted(
            (Hit(self.chunks[i], self._score(i, q, idf, avg)) for i in candidates),
            key=lambda h: (-h.score, h.chunk.id),
        )[:k]
        if filter == "after":
            ranked = [h for h in ranked if h.chunk.group in groups]
        return ranked
