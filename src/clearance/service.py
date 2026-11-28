"""The retrieval service: search the handbook as the authenticated user, and nothing more.

Identity comes from the gateway in front of the service: a header set by an in-cluster
proxy, which a network policy makes trustworthy, or on GKE the token Identity-Aware Proxy
signs, which the service verifies (see `identity.py`). The caller never says which groups it
belongs to: groups are looked up here, from the access file, so a forged group list is
impossible and an unknown user reads nothing.

The index is built in the background at start-up. `/healthz` answers at once (the process is
alive); `/readyz` answers 503 until the index is built, so Kubernetes sends no traffic to a
pod that cannot search yet, and a rolling restart never drops a request.

Configuration, all from the environment:

* `CLEARANCE_CORPUS`: directory of handbook Markdown (required)
* `CLEARANCE_ACCESS`: the access file, sections to groups and users to groups (required)
* `CLEARANCE_IDENTITY`: `header` (default) or `iap`
* `CLEARANCE_USER_HEADER`: the identity header in `header` mode (default `x-authenticated-user`)
* `CLEARANCE_IAP_AUDIENCE`: in `iap` mode, required: `/projects/NUMBER/global/backendServices/ID`
* `CLEARANCE_STRATEGY`: chunking strategy (default `headings_titled`)
* `CLEARANCE_RETRIEVAL`: `keyword` (default), `semantic` (pgvector) or `hybrid` (keyword and
  semantic results fused). Semantic and hybrid also need:
  * `CLEARANCE_PG_DSN`: the database; the chunks are loaded into it at start-up unless it
    already holds exactly these (see `pgvector.ensure_loaded`)
  * `CLEARANCE_VECTORS`: the committed chunk embeddings
  * `CLEARANCE_QUERY_EMBEDDER`: `ollama` (EmbeddingGemma at `OLLAMA_HOST`) or `fixture`
    (precomputed vectors for known questions in `CLEARANCE_QUERY_VECTORS`, for CI)

Every response says which retrieval answered it: if the question cannot be embedded, the
search falls back to keyword results rather than failing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from clearance.chunking import chunk_all
from clearance.corpus import Access, load
from clearance.dense import Vectors, gemma_embedder
from clearance.identity import HeaderIdentity, IapIdentity, Identity
from clearance.retrieval import FixtureQueries, QueryEmbedder, Retriever

MAX_K = 20
MAX_QUERY = 500
SNIPPET = 400


@dataclass
class State:
    corpus_dir: Path
    access_path: Path
    strategy: str = "headings_titled"
    identity: Identity = field(default_factory=HeaderIdentity)
    retrieval: str = "keyword"
    pg_dsn: str = ""
    vectors_path: Path | None = None
    query_embedder: QueryEmbedder | None = None
    access: Access | None = None
    retriever: Retriever | None = None
    error: str = ""
    ready: threading.Event = field(default_factory=threading.Event)

    def build(self) -> None:
        started = time.perf_counter()
        try:
            access = Access.load(self.access_path)
            corpus = load(self.corpus_dir, access)
            if not corpus.docs:
                raise ValueError(f"no documents under {self.corpus_dir}")
            chunks = chunk_all(corpus.docs, self.strategy)
            dense, loaded = None, False
            if self.retrieval in ("semantic", "hybrid"):
                from clearance.pgvector import PgDense, PgVectorIndex, ensure_loaded

                if self.vectors_path is None or self.query_embedder is None:
                    raise ValueError(
                        f"{self.retrieval} retrieval needs vectors and a query embedder"
                    )
                pg = _connect(PgVectorIndex, self.pg_dsn)
                loaded = ensure_loaded(pg, chunks, Vectors.load(self.vectors_path))
                dense = PgDense(pg, chunks)
            elif self.retrieval != "keyword":
                raise ValueError(f"retrieval must be keyword, semantic or hybrid: {self.retrieval}")
            self.access = access
            mode = "hybrid" if self.retrieval == "hybrid" else "semantic"
            self.retriever = Retriever(chunks, access, dense, self.query_embedder, mode=mode)
            self.ready.set()
            _log(
                event="ready",
                docs=len(corpus.docs),
                stubs=len(corpus.stubs),
                chunks=len(chunks),
                retrieval=self.retrieval,
                loaded_vectors=loaded,
                seconds=round(time.perf_counter() - started, 2),
            )
        except Exception as e:  # stay alive and unready, and say why
            self.error = f"{type(e).__name__}: {e}"
            _log(event="index_failed", error=self.error)


def _connect(factory, dsn: str, attempts: int = 60, wait: float = 2.0):
    """The database may start after the service: keep trying for about two minutes."""
    for attempt in range(1, attempts + 1):
        try:
            return factory(dsn)
        except Exception as e:
            if attempt == attempts:
                raise
            _log(event="waiting_for_database", attempt=attempt, error=type(e).__name__)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _log(**fields: object) -> None:
    """One JSON line per event on stdout, which Cloud Logging and kubectl logs both read.
    Query text is not logged: questions about leave or pay are personal."""
    print(json.dumps(fields, sort_keys=True), file=sys.stdout, flush=True)


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(state: State, background: bool = True) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        if background:
            threading.Thread(target=state.build, daemon=True).start()
        else:
            state.build()
        yield

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "alive"})

    async def readyz(request: Request) -> JSONResponse:
        if not state.ready.is_set():
            return _error(503, state.error or "building the index")
        if not state.identity.ready():
            return _error(503, "identity keys not loaded")
        return JSONResponse({"status": "ready"})

    async def search(request: Request) -> JSONResponse:
        user, reason = state.identity.user(request.headers)
        if not user:
            _log(event="denied", mode=state.identity.mode, reason=reason)
            return _error(401, "no authenticated user")
        if not state.ready.is_set() or state.retriever is None or state.access is None:
            return _error(503, "not ready")
        query = request.query_params.get("q", "").strip()
        if not query or len(query) > MAX_QUERY:
            return _error(400, f"q is required, at most {MAX_QUERY} characters")
        try:
            k = int(request.query_params.get("k", "5"))
        except ValueError:
            return _error(400, "k must be a number")
        if not 1 <= k <= MAX_K:
            return _error(400, f"k must be between 1 and {MAX_K}")
        started = time.perf_counter()
        groups = state.access.groups_of(user)
        hits, method = await run_in_threadpool(state.retriever.retrieve, query, groups, k)
        _log(
            event="search",
            user=user,
            k=k,
            retrieval=method,
            returned=[h.chunk.id for h in hits],
            ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return JSONResponse(
            {
                "retrieval": method,
                "results": [
                    {
                        "id": h.chunk.id,
                        "path": h.chunk.path,
                        "title": h.chunk.title,
                        "heading": h.chunk.heading,
                        "score": round(h.score, 4),
                        "text": h.chunk.text[:SNIPPET],
                    }
                    for h in hits
                ],
            }
        )

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/search", search),
        ],
        lifespan=lifespan,
    )


def from_env() -> Starlette:
    """The app as uvicorn runs it: `uvicorn --factory clearance.service:from_env`."""
    return create_app(
        State(
            corpus_dir=Path(os.environ["CLEARANCE_CORPUS"]),
            access_path=Path(os.environ["CLEARANCE_ACCESS"]),
            strategy=os.environ.get("CLEARANCE_STRATEGY", "headings_titled"),
            identity=identity_from_env(),
            retrieval=os.environ.get("CLEARANCE_RETRIEVAL", "keyword"),
            pg_dsn=os.environ.get("CLEARANCE_PG_DSN", ""),
            vectors_path=_path("CLEARANCE_VECTORS"),
            query_embedder=query_embedder_from_env(),
        )
    )


def _path(name: str) -> Path | None:
    value = os.environ.get(name, "")
    return Path(value) if value else None


def query_embedder_from_env() -> QueryEmbedder | None:
    kind = os.environ.get("CLEARANCE_QUERY_EMBEDDER", "")
    if not kind:
        return None
    if kind == "fixture":
        path = _path("CLEARANCE_QUERY_VECTORS")
        if path is None:
            raise ValueError("CLEARANCE_QUERY_EMBEDDER=fixture needs CLEARANCE_QUERY_VECTORS")
        return FixtureQueries(Vectors.load(path))
    if kind == "ollama":
        embed = gemma_embedder(timeout=5)
        return lambda q: embed([q], "RETRIEVAL_QUERY")[0]
    raise ValueError(f"CLEARANCE_QUERY_EMBEDDER must be ollama or fixture, not {kind!r}")


def identity_from_env() -> Identity:
    """Fail at start-up, not per request, when the identity configuration is wrong."""
    mode = os.environ.get("CLEARANCE_IDENTITY", "header")
    if mode == "header":
        return HeaderIdentity(os.environ.get("CLEARANCE_USER_HEADER", "x-authenticated-user"))
    if mode == "iap":
        return IapIdentity(os.environ.get("CLEARANCE_IAP_AUDIENCE", ""))
    raise ValueError(f"CLEARANCE_IDENTITY must be header or iap, not {mode!r}")
