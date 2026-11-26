"""Loading the handbook: front matter, cleanup, access groups, and stub detection.

Real handbooks are messy. These pages carry Hugo front matter and shortcodes, HTML comments,
and many "pages" that are only a link to an internal handbook. A stub like that would match
queries on its title and return nothing useful, so stubs are dropped and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
STUB_WORDS = 40


@dataclass(frozen=True)
class Access:
    section_groups: dict[str, str]
    user_groups: dict[str, frozenset[str]]

    @classmethod
    def load(cls, path: Path) -> Access:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            section_groups=dict(raw["sections"]),
            user_groups={u: frozenset(g) for u, g in raw["users"].items()},
        )

    def groups_of(self, user: str) -> frozenset[str]:
        return self.user_groups.get(user, frozenset())  # unknown users can read nothing


@dataclass
class Doc:
    path: str  # relative, with forward slashes, e.g. "legal/privacy/_index.md"
    section: str
    title: str
    text: str  # cleaned Markdown body
    group: str

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Corpus:
    docs: list[Doc]
    stubs: list[str] = field(default_factory=list)


def clean(body: str) -> str:
    body = COMMENT.sub("", body)
    body = SHORTCODE.sub("", body)
    body = IMAGE.sub("", body)
    body = LINK.sub(r"\1", body)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def parse(raw: str) -> tuple[dict, str]:
    m = FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[m.end() :]


def load(root: Path, access: Access) -> Corpus:
    docs, stubs = [], []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        section = rel.split("/", 1)[0]
        group = access.section_groups.get(section)
        if group is None:
            continue  # a section nobody has been granted is not indexed at all
        meta, body = parse(path.read_text(encoding="utf-8", errors="replace"))
        text = clean(body)
        doc = Doc(rel, section, str(meta.get("title") or Path(rel).stem), text, group)
        if doc.words < STUB_WORDS:
            stubs.append(rel)
            continue
        docs.append(doc)
    return Corpus(docs, stubs)
