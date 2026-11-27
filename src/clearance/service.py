"""The retrieval service: search the handbook as the authenticated user, and nothing more.

Identity comes from one header set by the gateway in front of the service (Identity-Aware
Proxy on GKE, or an in-cluster proxy). The caller never says which groups it belongs to:
groups are looked up here, from the access file, so a forged group list is impossible and an
unknown user reads nothing. A network policy lets only gateway pods reach the service, so
nothing else in the cluster can set the header.

The index is built in the background at start-up. `/healthz` answers at once (the process is
alive); `/readyz` answers 503 until the index is built, so Kubernetes sends no traffic to a
pod that cannot search yet, and a rolling restart never drops a request.

Configuration, all from the environment:

* `CLEARANCE_CORPUS`: directory of handbook Markdown (required)
* `CLEARANCE_ACCESS`: the access file, sections to groups and users to groups (required)
* `CLEARANCE_USER_HEADER`: the identity header (default `x-authenticated-user`)
* `CLEARANCE_STRATEGY`: chunking strategy (default `headings_titled`)
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
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from clearance.chunking import chunk_all
from clearance.corpus import Access, load
from clearance.index import Bm25Index

MAX_K = 20
MAX_QUERY = 500
SNIPPET = 400


@dataclass
class State:
    corpus_dir: Path
    access_path: Path
    strategy: str = "headings_titled"
    user_header: str = "x-authenticated-user"
    access: Access | None = None
    index: Bm25Index | None = None
    error: str = ""
    ready: threading.Event = field(default_factory=threading.Event)

    def build(self) -> None:
        started = time.perf_counter()
        try:
            access = Access.load(self.access_path)
            corpus = load(self.corpus_dir, access)
            if not corpus.docs:
                raise ValueError(f"no documents under {self.corpus_dir}")
            self.access = access
            self.index = Bm25Index(chunk_all(corpus.docs, self.strategy))
            self.ready.set()
            _log(
                event="ready",
                docs=len(corpus.docs),
                stubs=len(corpus.stubs),
                chunks=len(self.index.chunks),
                seconds=round(time.perf_counter() - started, 2),
            )
        except Exception as e:  # stay alive and unready, and say why
            self.error = f"{type(e).__name__}: {e}"
            _log(event="index_failed", error=self.error)


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
        if state.ready.is_set():
            return JSONResponse({"status": "ready"})
        return _error(503, state.error or "building the index")

    async def search(request: Request) -> JSONResponse:
        user = request.headers.get(state.user_header, "").strip()
        if not user:
            return _error(401, "no authenticated user")
        if not state.ready.is_set() or state.index is None or state.access is None:
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
        hits = state.index.search(query, state.access.groups_of(user), k=k)
        _log(
            event="search",
            user=user,
            k=k,
            returned=[h.chunk.id for h in hits],
            ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return JSONResponse(
            {
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
                ]
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
            user_header=os.environ.get("CLEARANCE_USER_HEADER", "x-authenticated-user").lower(),
        )
    )
