"""Ingestion, chunking and permission-filtered retrieval, on a small fixture corpus."""

from pathlib import Path

import pytest

from clearance.chunking import STRATEGIES, chunk_all, fixed, headings
from clearance.corpus import Access, clean, load
from clearance.evaluate import Question, influence, leaks, quality
from clearance.index import Bm25Index

FIXTURE = Path(__file__).parent / "fixture"


@pytest.fixture(scope="module")
def access():
    return Access.load(FIXTURE / "access.yaml")


@pytest.fixture(scope="module")
def corpus(access):
    return load(FIXTURE, access)


def test_stubs_are_dropped_and_counted(corpus):
    assert [d.path for d in corpus.docs] == [
        "hr/leave.md",
        "legal/contracts.md",
        "security/incidents.md",
    ]
    assert corpus.stubs == ["security/stub.md"]


def test_cleanup_removes_markup_but_keeps_text():
    text = clean("A <!-- hidden --> [link](http://x) ![img](a.png) {{< note >}}tip{{< /note >}}")
    assert text.split() == ["A", "link", "tip"]


def test_every_chunk_keeps_its_documents_group(corpus):
    for strategy in STRATEGIES:
        for c in chunk_all(corpus.docs, strategy):
            doc = next(d for d in corpus.docs if d.path == c.path)
            assert c.group == doc.group


def test_heading_chunks_follow_sections_and_titled_ones_say_where_they_are(corpus):
    doc = corpus.docs[0]
    chunks = headings(doc)
    assert [c.heading for c in chunks] == ["Parental leave", "Sick leave"]
    titled = headings(doc, titled=True)
    assert titled[0].text.startswith("Leave Policy > Parental leave")


def test_fixed_windows_overlap(corpus):
    doc = corpus.docs[0]
    chunks = fixed(doc, size=30, overlap=10)
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert a[-10:] == b[:10]


def test_allowed_users_find_the_page(corpus, access):
    index = Bm25Index(chunk_all(corpus.docs, "headings_titled"))
    hits = index.search("who can sign big customer contracts", access.groups_of("tom"))
    assert hits[0].chunk.path == "legal/contracts.md"


@pytest.mark.parametrize("user", ["sam", "lena"])
def test_restricted_pages_never_reach_other_users(corpus, access, user):
    index = Bm25Index(chunk_all(corpus.docs, "headings_titled"))
    hits = index.search("Northwind settlement cap", access.groups_of(user), k=10)
    assert all(h.chunk.group in access.groups_of(user) for h in hits)
    assert not any("Northwind" in h.chunk.text for h in hits)


def test_unknown_users_read_nothing(corpus, access):
    index = Bm25Index(chunk_all(corpus.docs, "headings"))
    assert index.search("parental leave", access.groups_of("mallory")) == []


def test_filter_placement_leaks_and_starves(corpus, access):
    questions = [
        Question("Northwind settlement legal team", "legal/contracts.md"),
        Question("severity one incident escalation", "security/incidents.md"),
        Question("parental leave weeks", "hr/leave.md"),
    ]
    r = leaks(corpus, access, questions, "headings", k=2)
    assert r["before"]["leaked_chunks"] == 0
    assert r["after"]["leaked_chunks"] == 0
    assert r["none"]["leaked_chunks"] > 0


def test_unreadable_pages_do_not_move_what_a_user_sees(corpus, access):
    """ "four" and "system" also appear in the legal and security pages sam cannot read."""
    chunks = chunk_all(corpus.docs, "headings_titled")
    sam = access.groups_of("sam")
    full = Bm25Index(chunks)
    own = Bm25Index([c for c in chunks if c.group in sam])
    query = "four weeks system"

    def scores(hits):
        return [(h.chunk.id, round(h.score, 9)) for h in hits]

    assert scores(full.search(query, sam)) == scores(own.search(query, sam))
    assert scores(full.search(query, sam, scoped=False)) != scores(own.search(query, sam))


def test_influence_is_measured_per_user(corpus, access):
    questions = [Question("four weeks system", "hr/leave.md")]
    r = influence(corpus, access, questions, "headings_titled")
    assert r["scoped"] == {"asked": 3, "scores_differ": 0, "order_differs": 0}
    assert r["global"]["scores_differ"] >= 1


def test_quality_is_scored_for_every_strategy(corpus, access):
    questions = [
        Question("who can sign big customer contracts", "legal/contracts.md"),
        Question("parental leave weeks", "hr/leave.md"),
        Question("a page that was never indexed", "hr/missing.md"),
    ]
    q = quality(corpus, access, questions)
    assert set(q) == set(STRATEGIES)
    for v in q.values():
        assert v["questions"] == 2  # the unindexed source is skipped, not scored as a miss
        assert v["hit@1"] == 1.0 and v["mrr"] == 1.0 and v["missed_top5"] == []
