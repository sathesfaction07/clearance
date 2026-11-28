"""One retrieval entry point for the agent and the service: keyword, semantic, or hybrid
(both, fused). Permissions are applied inside every retriever, before ranking.

Semantic is the default because it was measured best: on paraphrased questions it finds the
right page in the top 5 for 88% of questions, against 36% for keyword and 79% for hybrid
(reports/semantic.md).

If embedding the question fails (the embedding service is down or out of quota), the search
falls back to keyword results instead of failing, and says so: `retrieve` returns which
method answered, and the service reports it with every response.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from clearance.chunking import Chunk
from clearance.corpus import Access
from clearance.dense import DenseIndex, Vectors, fuse, text_hash
from clearance.index import Bm25Index, Hit

DEPTH = 20
QueryEmbedder = Callable[[str], np.ndarray]


class DenseSearch(Protocol):
    def search(self, query: np.ndarray, groups: frozenset[str], k: int = 5) -> list[Hit]: ...


class FixtureQueries:
    """Query vectors computed ahead of time, looked up by the question's text. CI has no
    embedding model, so the probe's questions come with their vectors; any other question
    raises KeyError, and the search falls back to keyword results, visibly."""

    def __init__(self, vectors: Vectors) -> None:
        self._by_hash = {h: vectors.matrix[i] for i, h in enumerate(vectors.hashes)}

    def __call__(self, query: str) -> np.ndarray:
        return self._by_hash[text_hash(query)]


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        access: Access,
        dense: DenseSearch | None = None,
        embed_query: QueryEmbedder | None = None,
        k: int = 5,
        mode: str = "semantic",
    ) -> None:
        if mode not in ("semantic", "hybrid"):
            raise ValueError(f"mode must be semantic or hybrid, not {mode!r}")
        self.mode = mode
        self.access = access
        self.k = k
        self.chunks = chunks
        self.bm25 = Bm25Index(chunks)
        self.dense = dense
        self.embed_query = embed_query
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def in_memory(
        cls,
        chunks: list[Chunk],
        access: Access,
        vectors: Vectors,
        embed_query: QueryEmbedder | None,
        k: int = 5,
        mode: str = "semantic",
    ) -> Retriever:
        return cls(chunks, access, DenseIndex(chunks, vectors), embed_query, k, mode)

    def query_vector(self, query: str) -> np.ndarray | None:
        if self.embed_query is None:
            return None
        if query not in self._cache:
            try:
                self._cache[query] = self.embed_query(query)
            except Exception:
                return None
        return self._cache[query]

    def __call__(self, query: str, user: str, k: int | None = None) -> list[Hit]:
        return self.retrieve(query, self.access.groups_of(user), k)[0]

    def retrieve(
        self, query: str, groups: frozenset[str], k: int | None = None
    ) -> tuple[list[Hit], str]:
        k = k or self.k
        vec = self.query_vector(query) if self.dense is not None else None
        if vec is None or self.dense is None:
            return self.bm25.search(query, groups, k=k), "keyword"
        if self.mode == "semantic":
            return self.dense.search(vec, groups, k=k), "semantic"
        keyword = self.bm25.search(query, groups, k=DEPTH)
        return fuse([keyword, self.dense.search(vec, groups, k=DEPTH)], k=k), "hybrid"
