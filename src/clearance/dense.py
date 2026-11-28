"""Semantic retrieval: embeddings, searched exactly in memory or approximately in pgvector.

The embedding model is EmbeddingGemma, Google's open 300M-parameter embedding model, run
locally with Ollama. (The Gemini API's `gemini-embedding-001` was the first choice; its free
tier allows 1,000 embeddings a day, so the 7,544 chunks would take eight days. Both are
supported here.) Running the model inside the cluster also means no question or handbook text
leaves it.

Embeddings are computed once and committed with a hash of the text each one came from. CI
never computes a chunk embedding: it loads the committed vectors, and a check fails if any
chunk's text no longer matches its hash (the handbook is pinned to one commit, so they match).

The permission rule is the same as for keyword search: only chunks the asker may read are
compared with the question. Dense scores need no corpus statistics (a cosine similarity
depends on the question and the chunk alone), so restricted pages cannot shift anyone's
scores here; where the filter sits is the whole question, and in an approximate index it
matters a great deal (see `pgvector.py`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clearance.chunking import Chunk
from clearance.index import Hit

MODEL = "embeddinggemma"
DIMS = 768
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
# EmbeddingGemma is trained with these prompts for retrieval (see its model card).
GEMMA_PROMPTS = {
    "RETRIEVAL_QUERY": "task: search result | query: ",
    "RETRIEVAL_DOCUMENT": "title: none | text: ",
}
BATCH = 100


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Vectors:
    """Unit-length vectors keyed by id, with the hash of the text each was computed from."""

    ids: list[str]
    hashes: list[str]
    matrix: np.ndarray  # float32, one row per id, unit length
    model: str = ""

    @classmethod
    def empty(cls, model: str = "") -> Vectors:
        return cls([], [], np.zeros((0, DIMS), dtype=np.float32), model)

    @classmethod
    def load(cls, path: Path) -> Vectors:
        if not path.exists():
            return cls.empty()
        with np.load(path) as f:
            model = str(f["model"]) if "model" in f.files else ""
            return cls(
                [str(i) for i in f["ids"]],
                [str(h) for h in f["hashes"]],
                f["vectors"].astype(np.float32),
                model,
            )

    def save(self, path: Path) -> None:
        """Stored as float16 (half the size; retrieval is unchanged at this precision)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            ids=np.array(self.ids),
            hashes=np.array(self.hashes),
            vectors=self.matrix.astype(np.float16),
            model=np.array(self.model),
        )
        os.replace(tmp, path)

    def add(self, ids: Sequence[str], hashes: Sequence[str], rows: np.ndarray) -> None:
        self.ids += list(ids)
        self.hashes += list(hashes)
        self.matrix = np.vstack([self.matrix, normalise(rows)])

    def by_id(self) -> dict[str, int]:
        return {i: n for n, i in enumerate(self.ids)}


def normalise(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


Embedder = Callable[[list[str], str], np.ndarray]


def gemma_embedder(model: str = MODEL, host: str = OLLAMA, timeout: float = 120) -> Embedder:
    """Texts to vectors with EmbeddingGemma served by Ollama (local, free, no limits)."""
    base = host if host.startswith("http") else f"http://{host}"

    def embed(texts: list[str], task: str) -> np.ndarray:
        body = json.dumps(
            {"model": model, "input": [GEMMA_PROMPTS[task] + t for t in texts], "truncate": True}
        ).encode()
        req = urllib.request.Request(
            f"{base}/api/embed", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:  # a local, configured URL
            return np.array(json.load(r)["embeddings"], dtype=np.float32)

    return embed


def gemini_embedder(model: str = "gemini-embedding-001", dims: int = DIMS) -> Embedder:
    """Texts to vectors with the Gemini API (needs GOOGLE_API_KEY; free tier is enough)."""
    from google import genai
    from google.genai import types

    client = genai.Client()

    def embed(texts: list[str], task: str) -> np.ndarray:
        r = client.models.embed_content(
            model=model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=dims),
        )
        return np.array([e.values for e in r.embeddings], dtype=np.float32)

    return embed


def embed_missing(
    items: Sequence[tuple[str, str]],
    path: Path,
    embed: Embedder,
    task: str,
    model: str = MODEL,
    batch: int = BATCH,
    pause: float = 0.0,
    log: Callable[[str], None] = print,
) -> Vectors:
    """Embed every (id, text) not already in `path` with a matching hash, saving after each
    batch, so a run stopped by a quota limit resumes where it stopped."""
    have = Vectors.load(path)
    if have.model != model:  # vectors from another model are not comparable: start again
        have = Vectors.empty(model)
    known = dict(zip(have.ids, have.hashes, strict=True))
    todo = [(i, t) for i, t in items if known.get(i) != text_hash(t)]
    stale = {i for i, _ in todo if i in known}  # the text changed: drop the old vector
    if stale:
        keep = [n for n, i in enumerate(have.ids) if i not in stale]
        have = Vectors(
            [have.ids[n] for n in keep],
            [have.hashes[n] for n in keep],
            have.matrix[keep],
            model,
        )
    for start in range(0, len(todo), batch):
        part = todo[start : start + batch]
        rows = embed([t for _, t in part], task)
        have.add([i for i, _ in part], [text_hash(t) for _, t in part], rows)
        have.save(path)
        log(f"embedded {len(have.ids)} of {len(items)}")
        if pause and start + batch < len(todo):
            time.sleep(pause)
    return have


MAX_CHARS = 6000  # the model reads 2,048 tokens; a handful of chunks are longer


def chunk_text(chunk: Chunk) -> str:
    """What is embedded for a chunk: its text (for titled chunks it starts with the page title
    and heading path), cut to what the model reads."""
    return chunk.text[:MAX_CHARS]


class DenseIndex:
    """Exact cosine search over the chunks the asker may read: the reference every
    approximate index is compared with."""

    def __init__(self, chunks: list[Chunk], vectors: Vectors) -> None:
        pos = vectors.by_id()
        missing = [c.id for c in chunks if c.id not in pos]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no vector, e.g. {missing[0]}")
        stale = [c.id for c in chunks if vectors.hashes[pos[c.id]] != text_hash(chunk_text(c))]
        if stale:
            raise ValueError(f"{len(stale)} vectors are for different text, e.g. {stale[0]}")
        self.chunks = chunks
        self.matrix = vectors.matrix[[pos[c.id] for c in chunks]]
        self._group = np.array([c.group for c in chunks])

    def search(self, query: np.ndarray, groups: frozenset[str], k: int = 5) -> list[Hit]:
        allowed = np.flatnonzero(np.isin(self._group, list(groups)))
        if not len(allowed):
            return []
        scores = self.matrix[allowed] @ normalise(query.reshape(1, -1))[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[allowed[i]], float(scores[i])) for i in top]


def fuse(rankings: Sequence[list[Hit]], k: int = 5, c: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: each list votes 1 / (c + rank) for its chunks. Needs no score
    calibration between keyword and dense scores, which are on different scales."""
    votes: dict[str, float] = {}
    chunk: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            votes[h.chunk.id] = votes.get(h.chunk.id, 0.0) + 1.0 / (c + rank)
            chunk[h.chunk.id] = h.chunk
    order = sorted(votes, key=lambda i: (-votes[i], i))[:k]
    return [Hit(chunk[i], votes[i]) for i in order]
