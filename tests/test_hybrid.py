"""Semantic and hybrid retrieval in the service: pgvector results (alone or fused with
keyword results), permissions intact, and a visible fallback to keyword search when a
question cannot be embedded."""

import os
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from clearance.chunking import chunk_all
from clearance.corpus import Access, load
from clearance.dense import Vectors, chunk_text, text_hash
from clearance.retrieval import FixtureQueries, Retriever
from clearance.service import State, create_app

FIXTURE = Path(__file__).parent / "fixture"
USER = "x-authenticated-user"
DSN = os.environ.get("CLEARANCE_TEST_PG_DSN", "")
needs_pg = pytest.mark.skipif(not DSN, reason="set CLEARANCE_TEST_PG_DSN to a pgvector database")
QUESTION = "what caps the settlement?"


def _setup(tmp_path):
    access = Access.load(FIXTURE / "access.yaml")
    chunks = chunk_all(load(FIXTURE, access).docs, "headings_titled")
    rng = np.random.default_rng(3)
    vectors = Vectors.empty("fake")
    vectors.add(
        [c.id for c in chunks],
        [text_hash(chunk_text(c)) for c in chunks],
        rng.normal(size=(len(chunks), 768)),
    )
    legal = next(i for i, c in enumerate(chunks) if c.group == "legal")
    queries = Vectors.empty("fake")
    queries.add(["q"], [text_hash(QUESTION)], vectors.matrix[[legal]] + 0.01)
    path = tmp_path / "chunks.npz"
    vectors.save(path)
    return access, chunks, vectors, queries, path, chunks[legal]


def test_retriever_falls_back_to_keyword_when_embedding_fails(tmp_path):
    access, chunks, vectors, queries, _, legal = _setup(tmp_path)
    r = Retriever.in_memory(chunks, access, vectors, FixtureQueries(queries))
    hits, method = r.retrieve(QUESTION, access.groups_of("tom"))
    assert method == "semantic" and hits[0].chunk.id == legal.id
    fused = Retriever.in_memory(chunks, access, vectors, FixtureQueries(queries), mode="hybrid")
    hits, method = fused.retrieve(QUESTION, access.groups_of("tom"))
    assert method == "hybrid" and legal.id in [h.chunk.id for h in hits]
    with pytest.raises(ValueError):
        Retriever(chunks, access, mode="magic")
    hits, method = r.retrieve("a question nobody embedded", access.groups_of("tom"))
    assert method == "keyword"
    sam, _ = r.retrieve(QUESTION, access.groups_of("sam"))
    assert all(h.chunk.group != "legal" for h in sam)
    assert (
        r("Northwind settlement cap", "sam")
        == r.retrieve("Northwind settlement cap", access.groups_of("sam"))[0]
    )


@needs_pg
def test_the_service_answers_from_pgvector_and_loads_the_vectors_once(tmp_path, capsys):
    from clearance.pgvector import PgVectorIndex, ensure_loaded

    _, chunks, vectors, queries, path, legal = _setup(tmp_path)
    PgVectorIndex(DSN).conn.execute("DROP TABLE IF EXISTS chunks_meta")

    def state():
        return State(
            FIXTURE,
            FIXTURE / "access.yaml",
            retrieval="semantic",
            pg_dsn=DSN,
            vectors_path=path,
            query_embedder=FixtureQueries(queries),
        )

    with TestClient(create_app(state(), background=False)) as api:
        tom = api.get("/search", params={"q": QUESTION}, headers={USER: "tom"}).json()
        sam = api.get("/search", params={"q": QUESTION}, headers={USER: "sam"}).json()
        other = api.get("/search", params={"q": "parental leave"}, headers={USER: "sam"}).json()
    assert tom["retrieval"] == "semantic" and tom["results"][0]["id"] == legal.id
    assert sam["retrieval"] == "semantic"
    assert all(not r["path"].startswith("legal/") for r in sam["results"])
    assert other["retrieval"] == "keyword"
    assert '"loaded_vectors": true' in capsys.readouterr().out
    with TestClient(create_app(state(), background=False)):
        pass  # a second replica finds the same chunks already loaded
    assert '"loaded_vectors": false' in capsys.readouterr().out
    assert ensure_loaded(PgVectorIndex(DSN), chunks, vectors) is False
