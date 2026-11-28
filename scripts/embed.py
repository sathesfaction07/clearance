"""Embed the handbook chunks and the labelled questions, once, resumably.

Default: EmbeddingGemma through a local Ollama (`ollama pull embeddinggemma`). With
`--gemini`, gemini-embedding-001 through the Gemini API (needs GOOGLE_API_KEY; the free tier
allows 1,000 embeddings a day, and the script exits 3 when they run out).
Writes data/embeddings/chunks.npz and questions.npz, which are committed so CI and anyone
without the model reproduce every number. Re-running embeds only what is missing or changed.
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearance.chunking import chunk_all  # noqa: E402
from clearance.corpus import Access, load  # noqa: E402
from clearance.dense import (  # noqa: E402
    chunk_text,
    embed_missing,
    gemini_embedder,
    gemma_embedder,
)
from clearance.evaluate import load_questions  # noqa: E402

OUT = ROOT / "data" / "embeddings"
QUESTION_FILES = ["questions.yaml", "questions-paraphrased.yaml", "questions-absent.yaml"]


def patient(embed):
    """Wait out per-minute limits; stop on the daily one."""

    def call(texts, task):
        for attempt in range(8):
            try:
                return embed(texts, task)
            except Exception as e:
                text = str(e)
                if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
                    if attempt < 3:
                        time.sleep(10 * (attempt + 1))
                        continue
                    raise
                if "PerDay" in text:
                    print(text[:1500])
                    print("daily embedding quota reached; run again after it resets")
                    raise SystemExit(3) from e
                m = re.search(r"retryDelay'?: '?(\d+)", text)
                wait = int(m.group(1)) + 2 if m else 30
                print(f"rate limited, waiting {wait} s")
                time.sleep(wait)
        raise SystemExit("still rate limited after 8 tries")

    return call


def main() -> int:
    access = Access.load(ROOT / "config" / "access.yaml")
    corpus = load(ROOT / "data" / "raw" / "handbook" / "md", access)
    if "--gemini" in sys.argv:
        embed, batch, model = patient(gemini_embedder()), 100, "gemini-embedding-001"
    else:
        embed, batch, model = gemma_embedder(), 32, "embeddinggemma"
    questions = []
    for name in QUESTION_FILES:
        path = ROOT / "data" / name
        if path.exists():
            questions += [(f"{name}#{n}", q.q) for n, q in enumerate(load_questions(path))]
    embed_missing(questions, OUT / "questions.npz", embed, "RETRIEVAL_QUERY", model, batch)
    chunks = chunk_all(corpus.docs, "headings_titled")
    embed_missing(
        [(c.id, chunk_text(c)) for c in chunks], OUT / "chunks.npz", embed, "RETRIEVAL_DOCUMENT"
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
