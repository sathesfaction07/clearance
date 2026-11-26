"""Retrieval quality per chunking strategy, and permission leaks per filter placement.

Quality: for each labelled question, asked by someone allowed to read its source, is the
source page among the top k results (hit@k), and how high (mean reciprocal rank)?

Leaks: every question asked by every user. A leak is any returned chunk from a group the
user is not in. "Starved" counts questions where the user may read relevant pages but got
fewer than k results, which is what post-filtering does to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clearance.chunking import STRATEGIES, chunk_all
from clearance.corpus import Access, Corpus
from clearance.index import Bm25Index


@dataclass
class Question:
    q: str
    source: str

    @property
    def section(self) -> str:
        return self.source.split("/", 1)[0]


def load_questions(path: Path) -> list[Question]:
    return [Question(**x) for x in yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]]


def asker_for(question: Question, access: Access) -> str:
    """The least-privileged user who may read the source page."""
    group = access.section_groups[question.section]
    allowed = [u for u, g in access.user_groups.items() if group in g]
    return min(allowed, key=lambda u: (len(access.user_groups[u]), u))


def quality(
    corpus: Corpus, access: Access, questions: list[Question], ks: tuple[int, ...] = (1, 3, 5)
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    paths = {d.path for d in corpus.docs}
    usable = [q for q in questions if q.source in paths]
    for strategy in STRATEGIES:
        index = Bm25Index(chunk_all(corpus.docs, strategy))
        hits = {k: 0 for k in ks}
        rr = 0.0
        misses = []
        for q in usable:
            results = index.search(q.q, access.groups_of(asker_for(q, access)), k=max(ks) * 4)
            docs = list(dict.fromkeys(h.chunk.path for h in results))  # unique, ranked
            rank = docs.index(q.source) + 1 if q.source in docs else None
            for k in ks:
                hits[k] += bool(rank and rank <= k)
            rr += 1 / rank if rank else 0
            if not rank or rank > 5:
                misses.append(q.q)
        n = len(usable)
        out[strategy] = {
            "questions": n,
            **{f"hit@{k}": hits[k] / n for k in ks},
            "mrr": rr / n,
            "missed_top5": misses,
            "chunks": len(index.chunks),
        }
    return out


def leaks(
    corpus: Corpus, access: Access, questions: list[Question], strategy: str, k: int = 5
) -> dict[str, Any]:
    index = Bm25Index(chunk_all(corpus.docs, strategy))
    report: dict[str, Any] = {}
    for mode in ("before", "after", "none"):
        leaked = starved = asked = 0
        for groups in access.user_groups.values():
            for q in questions:
                results = index.search(q.q, groups, k=k, filter=mode)
                asked += 1
                leaked += sum(h.chunk.group not in groups for h in results)
                if len(results) < k:
                    allowed = index.search(q.q, groups, k=k, filter="before")
                    starved += len(allowed) > len(results)
        report[mode] = {"asked": asked, "leaked_chunks": leaked, "starved": starved}
    return report
