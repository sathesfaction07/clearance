"""Semantic search in PostgreSQL with pgvector, and where the permission filter sits in it.

The permission filter is a `WHERE grp = ANY(...)` clause, so no restricted chunk is ever
returned. What it does to everyone else depends on the plan Postgres chooses:

* `exact`: no vector index; every readable chunk is compared. Always complete, the reference.
* `hnsw`: the approximate (HNSW) index, which walks the graph for the `ef_search` nearest
  chunks of *any* group and only then applies the filter. When most near neighbours are
  restricted, fewer than k survive: the same starvation as filtering after ranking.
* `hnsw_iterative`: pgvector 0.8's iterative index scan (`hnsw.iterative_scan =
  strict_order`): when the filter removes too many, the scan continues through the graph
  until k readable chunks are found, in exact distance order.
* `planner`: whatever Postgres picks by itself: on a table this small it often scans every
  row exactly, which hides the problem until the data grows.

`hnsw` and `hnsw_iterative` force the vector index (sequential and bitmap scans off) so that
what is measured is the index's behaviour, not the planner's choice on a small table. There
is deliberately no index on the group column: with one, Postgres can answer from it and sort
exactly, and the measurement would no longer be of the vector index.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from clearance.chunking import Chunk
from clearance.dense import DIMS, Vectors, chunk_text, normalise, text_hash
from clearance.index import Hit

MODES = ("exact", "planner", "hnsw", "hnsw_iterative")


@dataclass
class PgHit:
    id: str
    distance: float


class PgVectorIndex:
    def __init__(self, dsn: str, table: str = "chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.table = table
        self._lock = threading.Lock()  # one connection per process, shared by requests
        self.version = self.conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]

    @property
    def iterative(self) -> bool:
        major, minor = (int(x) for x in self.version.split(".")[:2])
        return (major, minor) >= (0, 8)

    def load(self, chunks: Sequence[Chunk], vectors: Vectors, ef_construction: int = 64) -> None:
        """(Re)create the table, copy every chunk's group and vector, and build the indexes."""
        pos = vectors.by_id()
        t = self.table
        with self.conn.transaction():
            self.conn.execute(f"DROP TABLE IF EXISTS {t}")
            self.conn.execute(
                f"CREATE TABLE {t} (id text PRIMARY KEY, grp text NOT NULL, "
                f"embedding vector({DIMS}) NOT NULL)"
            )
            with (
                self.conn.cursor() as cur,
                cur.copy(f"COPY {t} (id, grp, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy,
            ):
                copy.set_types(["text", "text", "vector"])
                for c in chunks:
                    copy.write_row((c.id, c.group, vectors.matrix[pos[c.id]]))
        self.conn.execute(
            f"CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = {ef_construction})"
        )
        self.conn.execute(f"ANALYZE {t}")

    def count(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def _settings(self, mode: str, ef_search: int) -> list[str]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "hnsw_iterative" and not self.iterative:
            raise RuntimeError(f"pgvector {self.version} has no iterative scans (needs 0.8)")
        s = [f"SET LOCAL hnsw.ef_search = {int(ef_search)}"]
        if mode == "exact":
            s.append("SET LOCAL enable_indexscan = off")  # no vector index: exact by sorting
        if mode in ("hnsw", "hnsw_iterative"):
            s += ["SET LOCAL enable_seqscan = off", "SET LOCAL enable_bitmapscan = off"]
        if self.iterative:
            it = "strict_order" if mode == "hnsw_iterative" else "off"
            s.append(f"SET LOCAL hnsw.iterative_scan = {it}")
        return s

    def _query(self) -> str:
        return (
            f"SELECT id, embedding <=> %s AS d FROM {self.table} "
            "WHERE grp = ANY(%s) ORDER BY embedding <=> %s LIMIT %s"
        )

    def search(
        self,
        query: np.ndarray,
        groups: frozenset[str],
        k: int = 5,
        mode: str = "hnsw_iterative",
        ef_search: int = 40,
    ) -> list[PgHit]:
        q = normalise(query.reshape(1, -1))[0]
        with self._lock, self.conn.transaction():
            for s in self._settings(mode, ef_search):
                self.conn.execute(s)
            rows = self.conn.execute(self._query(), (q, sorted(groups), q, k)).fetchall()
        return [PgHit(i, float(d)) for i, d in rows]

    def plan(self, query: np.ndarray, groups: frozenset[str], mode: str, k: int = 5) -> str:
        """The plan Postgres uses in a mode: which index, if any, answered the search."""
        q = normalise(query.reshape(1, -1))[0]
        with self.conn.transaction():
            for s in self._settings(mode, 40):
                self.conn.execute(s)
            rows = self.conn.execute("EXPLAIN " + self._query(), (q, sorted(groups), q, k))
            return "\n".join(r[0] for r in rows.fetchall())


def fingerprint(chunks: Sequence[Chunk], vectors: Vectors) -> str:
    """Changes whenever a chunk, its group, its text or the embedding model changes."""
    h = hashlib.sha256(vectors.model.encode())
    for c in chunks:
        h.update(f"{c.id}|{c.group}|{text_hash(chunk_text(c))}\n".encode())
    return h.hexdigest()[:24]


def ensure_loaded(pg: PgVectorIndex, chunks: Sequence[Chunk], vectors: Vectors) -> bool:
    """Load the chunks unless the database already holds exactly these. Replicas starting
    together take an advisory lock, so one loads and the others wait and then skip.
    Returns whether this call loaded them."""
    want = fingerprint(chunks, vectors)
    meta = f"{pg.table}_meta"
    with pg._lock:
        pg.conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (pg.table,))
        try:
            pg.conn.execute(f"CREATE TABLE IF NOT EXISTS {meta} (fingerprint text NOT NULL)")
            row = pg.conn.execute(f"SELECT fingerprint FROM {meta}").fetchone()
            if row and row[0] == want:
                return False
            pg.load(chunks, vectors)
            pg.conn.execute(f"DELETE FROM {meta}")
            pg.conn.execute(f"INSERT INTO {meta} VALUES (%s)", (want,))
            return True
        finally:
            pg.conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (pg.table,))


class PgDense:
    """pgvector as the semantic half of hybrid retrieval, in the shipped mode."""

    def __init__(self, pg: PgVectorIndex, chunks: Sequence[Chunk]) -> None:
        self.pg = pg
        self.by_id = {c.id: c for c in chunks}
        self.mode = "hnsw_iterative" if pg.iterative else "exact"

    def search(self, query: np.ndarray, groups: frozenset[str], k: int = 5) -> list[Hit]:
        return [
            Hit(self.by_id[h.id], 1 - h.distance)
            for h in self.pg.search(query, groups, k=k, mode=self.mode)
        ]
