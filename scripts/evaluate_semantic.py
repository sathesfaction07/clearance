"""Keyword vs semantic vs hybrid retrieval, and permission filtering inside pgvector.

Uses the committed embeddings (data/embeddings/), so it needs no API key; run
scripts/fetch_handbook.py first. With CLEARANCE_PG_DSN set to a PostgreSQL database with
pgvector, it also loads every chunk into pgvector and measures each way of filtering there.
Writes reports/semantic.md. Exits non-zero if any search returns a restricted chunk, if
the shipped pgvector mode cuts a single search short, or if pages a user cannot read change
a single hybrid result that user sees.
"""

import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

from clearance.chunking import chunk_all
from clearance.corpus import Access, load
from clearance.dense import DenseIndex, Vectors, fuse, text_hash
from clearance.evaluate import asker_for, load_questions
from clearance.index import Bm25Index

ROOT = Path(__file__).resolve().parents[1]
SETS = {"questions.yaml": "As written", "questions-paraphrased.yaml": "Paraphrased"}
DEPTH = 20  # chunks taken from each retriever before fusing and de-duplicating pages
SHIPPED_PG = "hnsw_iterative"


def question_vectors(name, questions, vectors):
    pos = vectors.by_id()
    out = []
    for n, q in enumerate(questions):
        i = pos.get(f"{name}#{n}")
        if i is None or vectors.hashes[i] != text_hash(q.q):
            raise SystemExit(f"{name}#{n} has no current embedding; run scripts/embed.py")
        out.append(vectors.matrix[i])
    return out


class Retrievers:
    def __init__(self, chunks, vectors):
        self.bm25 = Bm25Index(chunks)
        self.dense = DenseIndex(chunks, vectors)

    def run(self, method, text, vec, groups, k=DEPTH):
        if method == "keyword":
            return self.bm25.search(text, groups, k=k)
        if method == "semantic":
            return self.dense.search(vec, groups, k=k)
        return fuse(
            [self.bm25.search(text, groups, k=DEPTH), self.dense.search(vec, groups, k=DEPTH)],
            k=k,
        )


def page_rank(hits, source):
    pages = list(dict.fromkeys(h.chunk.path for h in hits))
    return pages.index(source) + 1 if source in pages else None


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def quality(retrievers, access, paths, questions, qvecs):
    out, ranks = {}, {}
    for method in ("keyword", "semantic", "hybrid"):
        rs = []
        for q, v in zip(questions, qvecs, strict=True):
            if q.source not in paths:
                continue
            groups = access.groups_of(asker_for(q, access))
            rs.append(page_rank(retrievers.run(method, q.q, v, groups), q.source))
        n = len(rs)
        out[method] = {
            "n": n,
            **{f"hit@{k}": sum(bool(r and r <= k) for r in rs) / n for k in (1, 3, 5)},
            "mrr": sum(1 / r for r in rs if r) / n,
        }
        ranks[method] = rs
    return out, ranks


def paired(ranks, a, b):
    pairs = list(zip(ranks[a], ranks[b], strict=True))
    wins = sum(bool(x and x <= 5) and not (y and y <= 5) for x, y in pairs)
    losses = sum(bool(y and y <= 5) and not (x and x <= 5) for x, y in pairs)
    return wins, losses, sign_test(wins, losses)


def hybrid_influence(chunks, vectors, access, asked):
    """Hybrid results from the full corpus vs from only the pages each user may read."""
    full = Retrievers(chunks, vectors)
    differ = total = 0
    for groups in access.user_groups.values():
        own = Retrievers([c for c in chunks if c.group in groups], vectors)
        for text, vec in asked:
            a = full.run("hybrid", text, vec, groups, k=5)
            b = own.run("hybrid", text, vec, groups, k=5)
            total += 1
            differ += [(h.chunk.id, round(h.score, 12)) for h in a] != [
                (h.chunk.id, round(h.score, 12)) for h in b
            ]
    return total, differ


def pgvector_section(dsn, chunks, vectors, access, asked):
    from clearance.pgvector import MODES, PgVectorIndex

    pg = PgVectorIndex(dsn)
    started = time.perf_counter()
    pg.load(chunks, vectors)
    load_s = time.perf_counter() - started
    group_of = {c.id: c.group for c in chunks}
    modes = [m for m in MODES if m != "hnsw_iterative" or pg.iterative]
    exact = {}
    rows = {}
    for mode in modes:
        restricted = short = 0
        overlap, ms = [], []
        for user, groups in access.user_groups.items():
            for n, (_, vec) in enumerate(asked):
                t = time.perf_counter()
                hits = pg.search(vec, groups, k=5, mode=mode)
                ms.append((time.perf_counter() - t) * 1000)
                ids = [h.id for h in hits]
                restricted += sum(group_of[i] not in groups for i in ids)
                short += len(ids) < 5
                if mode == "exact":
                    exact[user, n] = ids
                else:
                    overlap.append(len(set(ids) & set(exact[user, n])) / 5)
        rows[mode] = {
            "searches": len(ms),
            "restricted": restricted,
            "short": short,
            "recall": statistics.fmean(overlap) if overlap else 1.0,
            "p50": statistics.median(ms),
            "p95": sorted(ms)[int(0.95 * (len(ms) - 1))],
        }
    least = min(access.user_groups.values(), key=len)
    plan = pg.plan(asked[0][1], frozenset(least), "planner")
    plan = re.sub(r"'\[[^\]]*\]'::vector", "'[question vector]'::vector", plan).splitlines()
    return pg.version, pg.count(), load_s, rows, plan


def main() -> int:
    access = Access.load(ROOT / "config" / "access.yaml")
    corpus = load(ROOT / "data" / "raw" / "handbook" / "md", access)
    chunks = chunk_all(corpus.docs, "headings_titled")
    vectors = Vectors.load(ROOT / "data" / "embeddings" / "chunks.npz")
    qv = Vectors.load(ROOT / "data" / "embeddings" / "questions.npz")
    retrievers = Retrievers(chunks, vectors)
    paths = {d.path for d in corpus.docs}

    lines = [
        "# Keyword, semantic and hybrid retrieval, and filtering in pgvector",
        "",
        f"Corpus: {len(chunks)} `headings_titled` chunks from {len(corpus.docs)} handbook pages "
        f"(the handbook pinned to one commit). Embeddings: `{vectors.model}`, 768 "
        "dimensions, computed once locally and committed. Each question is asked by "
        "the least-privileged user allowed to read its source. Regenerate with "
        "`python scripts/evaluate_semantic.py`.",
        "",
        "## Does the right page come back?",
        "",
        "Semantic: exact cosine search over the chunks the asker may read. Hybrid: reciprocal "
        f"rank fusion of the top {DEPTH} keyword and top {DEPTH} semantic chunks.",
        "",
        "| Questions | Retrieval | Source page in top 1 | Top 3 | Top 5 | Mean reciprocal rank |",
        "|:--|:--|--:|--:|--:|--:|",
    ]
    asked, compare = [], []
    for name, label in SETS.items():
        questions = load_questions(ROOT / "data" / name)
        qvecs = question_vectors(name, questions, qv)
        asked += [(q.q, v) for q, v in zip(questions, qvecs, strict=True)]
        q, ranks = quality(retrievers, access, paths, questions, qvecs)
        for method, v in q.items():
            lines.append(
                f"| {label} ({v['n']}) | {method} | {v['hit@1']:.0%} | {v['hit@3']:.0%} | "
                f"{v['hit@5']:.0%} | {v['mrr']:.2f} |"
            )
        for a, b in (("semantic", "keyword"), ("hybrid", "keyword"), ("hybrid", "semantic")):
            compare.append((label, a, b, *paired(ranks, a, b)))
    lines += [
        "",
        "Paired on the same questions (top 5): how often one finds the page and the other does "
        "not, with a two-sided sign test.",
        "",
        "| Questions | Comparison | Only the first finds it | Only the second | p |",
        "|:--|:--|--:|--:|--:|",
        *(f"| {lab} | {a} vs {b} | {w} | {lo} | {p:.2f} |" for lab, a, b, w, lo, p in compare),
    ]

    total, differ = hybrid_influence(chunks, vectors, access, asked)
    lines += [
        "",
        "## Whether unreadable pages change hybrid results",
        "",
        f"Every question as every user ({total} searches, top 5), full index against an index of "
        "only the pages that user may read.",
        "",
        f"**{differ} of {total}** hybrid searches differ. Keyword statistics are scoped to "
        "readable pages and a cosine similarity depends on nothing but the question and the "
        "chunk, so the fused ranking is exactly what the user would get if restricted pages "
        "did not exist.",
    ]

    failed = bool(differ)
    dsn = os.environ.get("CLEARANCE_PG_DSN", "")
    if dsn:
        version, count, load_s, rows, plan = pgvector_section(dsn, chunks, vectors, access, asked)
        least = min(access.user_groups.values(), key=len)
        open_share = sum(c.group in least for c in chunks) / len(chunks)
        names = {
            "exact": "Exact (no vector index)",
            "planner": "Postgres's own choice",
            "hnsw": "HNSW index, filter applied to what it finds",
            "hnsw_iterative": "HNSW with iterative scan (shipped)",
        }
        lines += [
            "",
            "## Filtering inside pgvector",
            "",
            f"pgvector {version}; {count} chunks loaded and indexed (HNSW, m 16, "
            f"ef_construction 64) in {load_s:.1f} s. Every question as every user, top 5, "
            "`hnsw.ef_search` 40 (the default). The filter is a `WHERE` clause on the chunk's "
            "group in every mode, so nothing restricted can be returned; what differs is "
            "whether users get their full results.",
            "",
            "| Mode | Searches | Restricted returned | Fewer than 5 results "
            "| Same results as exact | Median ms | p95 ms |",
            "|:--|--:|--:|--:|--:|--:|--:|",
            *(
                f"| {names[m]} | {r['searches']} | {r['restricted']} | {r['short']} | "
                f"{r['recall']:.1%} | {r['p50']:.1f} | {r['p95']:.1f} |"
                for m, r in rows.items()
            ),
            "",
            "The HNSW graph search finds the 40 nearest chunks of any group, and the filter then "
            f"removes those the user may not read. Only {open_share:.0%} of chunks are readable "
            "by everyone, "
            "so for most questions the nearest 40 hold too few readable chunks: users are "
            "starved exactly as with filtering after ranking. The iterative scan keeps "
            "walking the graph until 5 readable chunks are found.",
            "",
            "Plan Postgres chose by itself for the least-privileged user:",
            "",
            "```",
            *plan[:4],
            "```",
        ]
        shipped = rows.get(SHIPPED_PG)
        failed = failed or any(r["restricted"] for r in rows.values())
        failed = failed or shipped is None or bool(shipped["short"])
    else:
        lines += ["", "*pgvector section skipped: set CLEARANCE_PG_DSN to run it.*"]

    out = ROOT / "reports" / "semantic.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
