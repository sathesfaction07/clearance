"""Measure retrieval quality and permission leaks on the real handbook; write the report.

Run scripts/fetch_handbook.py first. Exits non-zero if the shipped design (filtering before
ranking) leaks a single chunk, so CI fails on any regression.
"""

import sys
from pathlib import Path

from clearance.corpus import Access, load
from clearance.evaluate import leaks, load_questions, quality

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = "headings_titled"


def main() -> int:
    access = Access.load(ROOT / "config" / "access.yaml")
    corpus = load(ROOT / "data" / "raw" / "handbook" / "md", access)
    questions = load_questions(ROOT / "data" / "questions.yaml")
    q = quality(corpus, access, questions)
    leak = leaks(corpus, access, questions, SHIPPED)
    n = next(iter(q.values()))["questions"]
    lines = [
        "# Retrieval and permissions report",
        "",
        f"Corpus: {len(corpus.docs)} handbook pages in 6 sections; {len(corpus.stubs)} stub "
        "pages (a link to an internal handbook and little else) dropped at ingestion.",
        f"Questions: {n}, each asked by the least-privileged user allowed to read its source. "
        "Retrieval: BM25 keyword search. Regenerate with `python scripts/evaluate.py`.",
        "",
        "## Chunking strategies",
        "",
        "| Strategy | Chunks | Source page in top 1 | Top 3 | Top 5 | Mean reciprocal rank |",
        "|:--|--:|--:|--:|--:|--:|",
        *(
            f"| {s} | {v['chunks']} | {v['hit@1']:.0%} | {v['hit@3']:.0%} | {v['hit@5']:.0%} "
            f"| {v['mrr']:.2f} |"
            for s, v in q.items()
        ),
        "",
        f"With {n} questions, one question is {1 / n:.0%}: differences of a question or two are "
        "noise, not a ranking. The questions were written by someone who had read the page "
        "headings, which favours keyword retrieval; paraphrased questions from real users are "
        "the harder test.",
        "",
        "## Where the permission check sits",
        "",
        f"Every question asked by every user ({leak['before']['asked']} searches, top 5, "
        f"`{SHIPPED}` chunks).",
        "",
        "| Filter | Restricted chunks returned | Searches cut short |",
        "|:--|--:|--:|",
        f"| Before ranking (shipped) | {leak['before']['leaked_chunks']} | "
        f"{leak['before']['starved']} |",
        f"| After ranking | {leak['after']['leaked_chunks']} | {leak['after']['starved']} |",
        f"| None | {leak['none']['leaked_chunks']} | {leak['none']['starved']} |",
        "",
        "*Cut short:* the user could read relevant pages, but got fewer than 5 results because "
        "restricted pages took the top places before being removed. Post-filtering does not "
        "leak text, but it starves users of answers and its short result lists reveal that "
        "something matched.",
    ]
    out = ROOT / "reports" / "retrieval.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 1 if leak["before"]["leaked_chunks"] else 0


if __name__ == "__main__":
    sys.exit(main())
