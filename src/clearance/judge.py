"""Grading answers: what code can decide, code decides; the rest goes to a model judge.

Decided in code:
* whether the agent refused (the fixed not-found sentence);
* whether it cited a passage from the page that answers the question;
* whether any citation pointed at a passage it was never shown (removed and counted);
* for questions the asker may not read, whether the answer repeats restricted text (any
  run of 8 words from a chunk the asker cannot read).

Decided by a judge model, a different and larger model than the agent's:
* correct: the answer agrees with the page that answers the question and actually answers it;
* grounded: every claim in the answer is supported by the passages the agent was shown.
  An ungrounded claim is a hallucination, whether or not it happens to be true.

The judge is itself checked before its verdicts are used (see scripts/evaluate_answers.py):
it must reject an answer to a different question and an answer with its numbers changed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

WORDS = re.compile(r"[a-z0-9]+")
SHINGLE = 8


def _shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    w = WORDS.findall(text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def repeats_any(answer: str, texts: Iterable[str]) -> bool:
    """True if the answer contains any run of 8 words found in one of `texts`."""
    mine = _shingles(answer)
    return bool(mine) and any(mine & _shingles(t) for t in texts)


CITATION_FREE = re.compile(r"\[[^\]]*\]")


def perturb_numbers(text: str) -> str:
    """The same answer with every number changed (citations left alone): a wrong answer that
    reads exactly right."""
    parts = re.split(r"(\[[^\]]*\])", text)
    return "".join(
        p if p.startswith("[") else re.sub(r"\d+", lambda m: str(int(m.group()) + 3), p)
        for p in parts
    )


def has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", CITATION_FREE.sub("", text)))


JUDGE_PROMPT = """You are grading an answer from a company handbook assistant.

QUESTION:
{question}

REFERENCE: the handbook page that answers the question (may be cut short):
<<<
{reference}
>>>

PASSAGES the assistant was shown:
<<<
{passages}
>>>

ANSWER to grade:
<<<
{answer}
>>>

Decide two things.
- correct: true if the answer responds to the question and everything it says agrees with
  the REFERENCE; false if it answers a different question, contradicts the reference, gets
  a number, name or condition wrong, or says it could not find the answer.
- grounded: true if every factual claim in the answer is supported by the PASSAGES; false
  if any claim is not in the passages (even if it might be true elsewhere).
Return JSON only: {{"correct": true|false, "grounded": true|false, "reason": "<one sentence>"}}
"""


@dataclass
class Verdict:
    correct: bool
    grounded: bool
    reason: str


def judge_prompt(question: str, reference: str, passages: str, answer: str) -> str:
    return JUDGE_PROMPT.format(
        question=question,
        reference=reference[:8000],
        passages=passages[:12000] or "(none)",
        answer=CITATION_FREE.sub("", answer).strip(),
    )


def parse_verdict(text: str) -> Verdict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    data: dict[str, Any] = json.loads(m.group())
    return Verdict(bool(data["correct"]), bool(data["grounded"]), str(data.get("reason", "")))
