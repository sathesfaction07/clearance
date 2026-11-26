"""Three ways to cut documents into retrievable chunks, compared on the same questions.

* `fixed`: windows of N words with overlap. The usual first attempt; cuts mid-section.
* `headings`: split at Markdown headings, merging small sections and splitting large ones.
* `headings_titled`: the same, with "Page title > Heading > Subheading" prepended, so a chunk
  says where it comes from even when its own text never names the topic.

Every chunk carries its document's access group: permissions are attached at ingestion, and
nothing downstream can lose them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from clearance.corpus import Doc

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    section: str
    title: str
    heading: str
    text: str
    group: str


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    return [words[i : i + size] for i in range(0, max(1, len(words) - overlap), step)]


def fixed(doc: Doc, size: int = 200, overlap: int = 40) -> list[Chunk]:
    return [
        Chunk(f"{doc.path}#f{i}", doc.path, doc.section, doc.title, "", " ".join(w), doc.group)
        for i, w in enumerate(_windows(doc.text.split(), size, overlap))
    ]


def _sections(doc: Doc) -> list[tuple[str, str]]:
    """Split into (heading path, text) at headings, tracking the heading hierarchy."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    last, path = 0, ""
    for m in HEADING.finditer(doc.text):
        out.append((path, doc.text[last : m.start()]))
        level = len(m.group(1))
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, m.group(2).strip())]
        path = " > ".join(h for _, h in stack)
        last = m.end()
    out.append((path, doc.text[last:]))
    return [(p, t.strip()) for p, t in out if t.strip()]


def headings(
    doc: Doc, max_words: int = 300, min_words: int = 40, titled: bool = False
) -> list[Chunk]:
    merged: list[tuple[str, str]] = []
    for path, text in _sections(doc):
        if merged and len(merged[-1][1].split()) < min_words:
            prev_path, prev_text = merged.pop()
            merged.append((prev_path or path, f"{prev_text}\n\n{text}"))
        else:
            merged.append((path, text))
    chunks = []
    for i, (path, text) in enumerate(merged):
        for j, window in enumerate(_windows(text.split(), max_words, 40)):
            body = " ".join(window)
            if titled:
                body = f"{doc.title}{' > ' + path if path else ''}\n{body}"
            chunks.append(
                Chunk(
                    f"{doc.path}#h{i}.{j}", doc.path, doc.section, doc.title, path, body, doc.group
                )
            )
    return chunks


STRATEGIES: dict[str, Callable[[Doc], list[Chunk]]] = {
    "fixed": fixed,
    "headings": headings,
    "headings_titled": lambda d: headings(d, titled=True),
}


def chunk_all(docs: list[Doc], strategy: str) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    return [c for d in docs for c in fn(d)]
