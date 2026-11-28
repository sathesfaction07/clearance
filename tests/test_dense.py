"""Semantic retrieval: exact search, rank fusion, resumable embedding, and pgvector's filter.

The pgvector tests need a PostgreSQL with the vector extension: CI runs them against a
pgvector service container, and locally they run when CLEARANCE_TEST_PG_DSN is set.
"""

import os

import numpy as np
import pytest

from clearance.chunking import Chunk
from clearance.dense import DenseIndex, Vectors, embed_missing, fuse, text_hash
from clearance.index import Hit

DSN = os.environ.get("CLEARANCE_TEST_PG_DSN", "")
needs_pg = pytest.mark.skipif(not DSN, reason="set CLEARANCE_TEST_PG_DSN to a pgvector database")


def _chunk(i: int, group: str, text: str = "") -> Chunk:
    return Chunk(f"doc{i}#0", f"doc{i}.md", "s", "t", "", text or f"chunk {i}", group)


def _corpus(n: int = 400, open_share: float = 0.05, seed: int = 1):
    """Restricted chunks crowd around the question; the few open ones lie further away."""
    rng = np.random.default_rng(seed)
    query = rng.normal(size=768).astype(np.float32)
    chunks, rows = [], []
    for i in range(n):
        is_open = i < n * open_share
        noise = rng.normal(size=768) * (2.0 if is_open else 0.6)
        rows.append(query + noise)
        chunks.append(_chunk(i, "open" if is_open else "secret"))
    vectors = Vectors.empty()
    vectors.add([c.id for c in chunks], [text_hash(c.text) for c in chunks], np.array(rows))
    return chunks, vectors, query


def test_exact_search_returns_only_readable_chunks_nearest_first():
    chunks, vectors, query = _corpus()
    index = DenseIndex(chunks, vectors)
    hits = index.search(query, frozenset({"open"}), k=5)
    assert len(hits) == 5 and all(h.chunk.group == "open" for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert index.search(query, frozenset(), k=5) == []
    everyone = index.search(query, frozenset({"open", "secret"}), k=5)
    assert all(h.chunk.group == "secret" for h in everyone)  # they really are nearer


def test_vectors_for_changed_text_are_refused():
    chunks, vectors, _ = _corpus(20)
    changed = [*chunks[:-1], _chunk(19, "secret", "the text changed since it was embedded")]
    with pytest.raises(ValueError, match="different text"):
        DenseIndex(changed, vectors)
    with pytest.raises(ValueError, match="no vector"):
        DenseIndex([*chunks, _chunk(99, "open")], vectors)


def test_embedding_resumes_and_only_redoes_changed_text(tmp_path):
    calls = []

    def fake(texts, task):
        calls.append(len(texts))
        return np.ones((len(texts), 768), dtype=np.float32)

    path = tmp_path / "v.npz"
    items = [(f"c{i}", f"text {i}") for i in range(250)]
    embed_missing(items, path, fake, "RETRIEVAL_DOCUMENT", "fake", log=lambda _: None)
    assert calls == [100, 100, 50]
    items[3] = ("c3", "edited")
    v = embed_missing(items, path, fake, "RETRIEVAL_DOCUMENT", "fake", log=lambda _: None)
    assert calls[-1] == 1 and len(v.ids) == 250
    assert Vectors.load(path).hashes[v.ids.index("c3")] == text_hash("edited")
    assert np.allclose(np.linalg.norm(Vectors.load(path).matrix, axis=1), 1, atol=1e-3)


def test_rank_fusion_rewards_agreement():
    a, b, c = _chunk(1, "g"), _chunk(2, "g"), _chunk(3, "g")
    fused = fuse([[Hit(a, 9), Hit(b, 8)], [Hit(b, 0.9), Hit(c, 0.8)]], k=3)
    assert [h.chunk.id for h in fused] == [b.id, a.id, c.id]


@pytest.fixture(scope="module")
def pg():
    from clearance.pgvector import PgVectorIndex

    chunks, vectors, query = _corpus()
    index = PgVectorIndex(DSN, table="test_chunks")
    index.load(chunks, vectors, ef_construction=32)
    yield index, chunks, query
    index.conn.execute("DROP TABLE IF EXISTS test_chunks")


@needs_pg
def test_no_mode_ever_returns_a_restricted_chunk(pg):
    index, _, query = pg
    modes = ["exact", "planner", "hnsw"] + (["hnsw_iterative"] if index.iterative else [])
    for mode in modes:
        hits = index.search(query, frozenset({"open"}), k=5, mode=mode, ef_search=10)
        assert all(h.id.startswith("doc") and int(h.id[3:-2]) < 20 for h in hits), mode
    assert index.count() == 400


@needs_pg
def test_the_approximate_index_starves_a_restricted_user_unless_it_scans_on(pg):
    index, _, query = pg
    exact = index.search(query, frozenset({"open"}), k=5, mode="exact")
    assert len(exact) == 5
    hnsw = index.search(query, frozenset({"open"}), k=5, mode="hnsw", ef_search=10)
    assert len(hnsw) < 5  # the graph walk found secret neighbours, then the filter removed them
    assert "embedding_idx" in index.plan(query, frozenset({"open"}), "hnsw")
    if index.iterative:
        it = index.search(query, frozenset({"open"}), k=5, mode="hnsw_iterative", ef_search=10)
        assert [h.id for h in it] == [h.id for h in exact]
