"""The HTTP service enforces identity and permissions, and the probe catches it if not."""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from starlette.testclient import TestClient

from clearance import probe
from clearance.corpus import Access
from clearance.evaluate import Question
from clearance.service import State, create_app

FIXTURE = Path(__file__).parent / "fixture"
USER = "x-authenticated-user"


def _state(**kw) -> State:
    return State(corpus_dir=FIXTURE, access_path=FIXTURE / "access.yaml", **kw)


@pytest.fixture
def api():
    with TestClient(create_app(_state(), background=False)) as client:
        yield client


def test_ready_once_the_index_is_built(api):
    assert api.get("/healthz").status_code == 200
    assert api.get("/readyz").json() == {"status": "ready"}


def test_not_ready_until_built_and_says_why_it_failed():
    idle = TestClient(create_app(_state()))  # no lifespan: nothing built yet
    assert idle.get("/healthz").status_code == 200
    assert idle.get("/readyz").status_code == 503
    assert idle.get("/search?q=leave", headers={USER: "sam"}).status_code == 503
    broken = State(corpus_dir=FIXTURE / "missing", access_path=FIXTURE / "access.yaml")
    with TestClient(create_app(broken, background=False)) as client:
        r = client.get("/readyz")
    assert r.status_code == 503 and "no documents" in r.json()["error"]


def test_search_needs_an_identity(api):
    assert api.get("/search?q=parental leave").status_code == 401


def test_each_user_sees_only_their_groups(api):
    q = {"q": "Northwind settlement cap", "k": 10}
    tom = api.get("/search", params=q, headers={USER: "tom"}).json()["results"]
    sam = api.get("/search", params=q, headers={USER: "sam"}).json()["results"]
    assert tom[0]["path"] == "legal/contracts.md"
    assert all(not r["path"].startswith("legal/") for r in sam)
    assert not any("Northwind" in r["text"] for r in sam)


def test_groups_cannot_be_claimed_by_the_caller(api):
    headers = {USER: "sam", "x-groups": "legal", "x-authenticated-groups": "legal"}
    params = {"q": "Northwind settlement", "groups": "legal"}
    results = api.get("/search", params=params, headers=headers).json()["results"]
    assert all(not r["path"].startswith("legal/") for r in results)


def test_unknown_users_read_nothing(api):
    r = api.get("/search?q=parental leave", headers={USER: "mallory"})
    assert r.status_code == 200 and r.json() == {"results": []}


@pytest.mark.parametrize(
    "query", ["", "q=", "q=leave&k=0", "q=leave&k=21", "q=leave&k=x", "q=" + "a" * 501]
)
def test_bad_requests_are_refused(api, query):
    assert api.get(f"/search?{query}", headers={USER: "sam"}).status_code == 400


def test_query_text_is_not_logged(api, capsys):
    api.get("/search?q=my+sick+leave+secret", headers={USER: "sam"})
    out = capsys.readouterr().out
    assert '"event": "search"' in out and "secret" not in out


@pytest.fixture
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        create_app(_state(), background=False), host="127.0.0.1", port=port, log_level="error"
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


QUESTIONS = [
    Question("Northwind settlement legal team", "legal/contracts.md"),
    Question("severity one incident escalation", "security/incidents.md"),
    Question("parental leave weeks", "hr/leave.md"),
]


def test_probe_passes_a_correct_service(server):
    access = Access.load(FIXTURE / "access.yaml")
    r = probe.probe(server, access, QUESTIONS, USER)
    assert r["searches"] == 9 and r["failed"] == 0 and r["leaked_results"] == 0
    assert r["results"] > 0 and r["anonymous_status"] == 401 and r["unknown_user_results"] == 0


def test_probe_catches_a_service_that_leaks(server):
    # Tell the probe that sam has no groups at all: everything sam is shown is then a leak,
    # which is what a service ignoring permissions would look like.
    access = Access.load(FIXTURE / "access.yaml")
    stricter = Access(access.section_groups, {**access.user_groups, "sam": frozenset()})
    r = probe.probe(server, stricter, QUESTIONS, USER)
    assert r["leaked_results"] > 0


def test_probe_counts_requests_that_fail():
    access = Access.load(FIXTURE / "access.yaml")
    r = probe.probe("http://127.0.0.1:9", access, QUESTIONS[:1], USER)
    assert r["failed"] == r["searches"] == 3


def test_probe_cli_exit_code(server, capsys, tmp_path):
    questions = tmp_path / "questions.yaml"
    questions.write_text(
        "questions:\n  - q: parental leave weeks\n    source: hr/leave.md\n", encoding="utf-8"
    )
    args = ["--url", server, "--access", str(FIXTURE / "access.yaml")]
    assert probe.main([*args, "--questions", str(questions)]) == 0
    assert '"leaked_results": 0' in capsys.readouterr().out
