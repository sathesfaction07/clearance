"""Answer quality: is the agent right, grounded, and silent when it should be?

    python scripts/evaluate_answers.py run --out runs/answers.jsonl --resume
    python scripts/evaluate_answers.py judge --resume
    python scripts/evaluate_answers.py report runs/answers.jsonl runs/judged.jsonl

Three kinds of question, each asked by a chosen user:
* answerable: the 33 labelled questions as written and the 33 paraphrased, each asked by the
  least-privileged user who may read the page that answers it;
* restricted: every written question whose page the least-privileged user may not read,
  asked by that user. The right answer is the not-found sentence, with no restricted text;
* absent: questions the handbook does not answer, asked by the auditor (who reads
  everything), so a refusal there is not a permissions effect.

The agent runs on the Gemini API free tier (default gemini-3.1-flash-lite), retrieves with
semantic search (EmbeddingGemma query vectors from a local Ollama), and is graded in code where
possible and by a larger judge model (default gemma-4-31b-it) for correctness and
grounding. The judge is checked first on answers known to be wrong.
(gemini-3.5-flash was the first judge; its free tier allows 20 requests a day, so it
graded a sample, kept as a cross-check of the main judge.) Runs are resumable; a
quota stop exits 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearance.chunking import chunk_all  # noqa: E402
from clearance.corpus import Access, load  # noqa: E402
from clearance.dense import Vectors, gemma_embedder  # noqa: E402
from clearance.evaluate import asker_for, load_questions  # noqa: E402
from clearance.judge import (  # noqa: E402
    has_numbers,
    judge_prompt,
    parse_verdict,
    perturb_numbers,
    repeats_any,
)
from clearance.retrieval import Retriever  # noqa: E402

AGENT_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = "gemma-4-31b-it"
OUTAGE = re.compile(
    r"\b(429|500|503)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|quota|ConnectError|ReadTimeout|ReadError",
    re.IGNORECASE,
)
CONTROLS = 20


def corpus():
    access = Access.load(ROOT / "config" / "access.yaml")
    docs = load(ROOT / "data" / "raw" / "handbook" / "md", access).docs
    return access, docs, chunk_all(docs, "headings_titled")


def items(access):
    least = min(access.user_groups, key=lambda u: (len(access.user_groups[u]), u))
    auditor = max(access.user_groups, key=lambda u: (len(access.user_groups[u]), u))
    out = []
    for name, label in (("questions.yaml", "written"), ("questions-paraphrased.yaml", "para")):
        for n, q in enumerate(load_questions(ROOT / "data" / name)):
            user = asker_for(q, access)
            out.append(
                {
                    "id": f"answerable:{label}#{n}",
                    "kind": "answerable",
                    "set": label,
                    "question": q.q,
                    "source": q.source,
                    "user": user,
                }
            )
    for n, q in enumerate(load_questions(ROOT / "data" / "questions.yaml")):
        if access.section_groups[q.section] not in access.groups_of(least):
            out.append(
                {
                    "id": f"restricted:written#{n}",
                    "kind": "restricted",
                    "set": "written",
                    "question": q.q,
                    "source": q.source,
                    "user": least,
                }
            )
    for n, q in enumerate(load_questions(ROOT / "data" / "questions-absent.yaml")):
        out.append(
            {
                "id": f"absent#{n}",
                "kind": "absent",
                "set": "absent",
                "question": q.q,
                "source": "",
                "user": auditor,
            }
        )
    return out


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["id"] for x in path.read_text("utf-8").splitlines() if x}


async def run(args) -> int:
    from google.adk.models.google_llm import Gemini
    from google.genai import types

    from clearance.agent import ask, build_agent

    logging.getLogger("google_adk").setLevel(logging.ERROR)
    access, _, chunks = corpus()
    vectors = Vectors.load(ROOT / "data" / "embeddings" / "chunks.npz")
    embed = gemma_embedder()
    retriever = Retriever.in_memory(
        chunks, access, vectors, lambda q: embed([q], "RETRIEVAL_QUERY")[0]
    )
    model = Gemini(
        model=args.model, retry_options=types.HttpRetryOptions(attempts=6, initial_delay=5)
    )
    agent = build_agent(model, retriever)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items(access) if i["id"] not in done_ids(out)]
    print(f"{len(todo)} questions to ask ({len(done_ids(out))} already in {out})")
    gap = 60 / args.rpm if args.rpm else 0
    for n, item in enumerate(todo, 1):
        started = time.perf_counter()
        a = await ask(agent, item["question"], item["user"])
        if a.error and OUTAGE.search(a.error):
            print(f"stopped at [{n}/{len(todo)}]: {a.error[:200]} (run again to resume)")
            return 3
        record = {
            **item,
            "model": args.model,
            **{k: v for k, v in vars(a).items() if k not in ("question", "user")},
        }
        record["refused"] = a.refused
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        tag = "REFUSED" if a.refused else "answered"
        print(f"[{n}/{len(todo)}] {item['kind']:<10} {tag:<8} {item['id']}")
        if gap:
            await asyncio.sleep(max(0.0, gap - (time.perf_counter() - started)))
    return 0


def page_text(docs, path):
    for d in docs:
        if d.path == path:
            return d.text
    return ""


def judge_one(client, model, prompt):
    from google.genai import types

    if model.startswith("gemma"):  # open models: no JSON mode or thinking settings
        config = types.GenerateContentConfig(temperature=0)
    else:
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        )
    for attempt in range(20):  # overloaded models (503) recover within minutes
        try:
            r = client.models.generate_content(model=model, contents=prompt, config=config)
            return parse_verdict(r.text or "")
        except Exception as e:
            text = str(e)
            if "PerDay" in text or attempt == 19:
                raise
            if OUTAGE.search(text):
                m = re.search(r"retry in (\d+)", text)
                time.sleep(int(m.group(1)) + 2 if m else 30)
                continue
            raise
    raise AssertionError("unreachable")


def judge(args) -> int:
    from google import genai

    _, docs, chunks = corpus()
    by_id = {c.id: c for c in chunks}
    answers = [json.loads(x) for x in Path(args.answers).read_text("utf-8").splitlines() if x]
    real = [a for a in answers if a["kind"] == "answerable" and not a["error"]]
    tasks = []
    for a in real:
        if not a["refused"]:
            tasks.append(("real", a, a["text"]))
    answered = [a for a in real if not a["refused"]]
    for a, other in list(zip(answered, answered[7:], strict=False))[:CONTROLS]:
        if other["source"] != a["source"]:  # the answer to the question seven places on
            tasks.append(("swapped", a, other["text"]))
    numeric = [a for a in answered if has_numbers(a["text"])]
    for a in numeric[:CONTROLS]:
        tasks.append(("numbers_changed", a, perturb_numbers(a["text"])))
    out = Path(args.out)
    have = done_ids(out)
    client = genai.Client()
    todo = [t for t in tasks if f"{t[0]}:{t[1]['id']}" not in have]
    print(f"{len(todo)} verdicts to get ({len(have)} already in {out})")
    for n, (kind, a, text) in enumerate(todo, 1):
        passages = "\n\n".join(
            f"[{cid}] {by_id[cid].text[:1500]}" for cid in a["retrieved"] if cid in by_id
        )
        prompt = judge_prompt(a["question"], page_text(docs, a["source"]), passages, text)
        try:
            v = judge_one(client, args.judge, prompt)
        except Exception as e:
            print(f"stopped at [{n}/{len(todo)}]: {str(e)[:200]} (run again to resume)")
            return 3
        with out.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "id": f"{kind}:{a['id']}",
                        "kind": kind,
                        "answer_id": a["id"],
                        "judge": args.judge,
                        **vars(v),
                    }
                )
                + "\n"
            )
        print(f"[{n}/{len(todo)}] {kind:<16} correct={v.correct} grounded={v.grounded}")
    return 0


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    if not n:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} ({lo:.0%} to {hi:.0%})"


def report(args) -> int:
    access, _, chunks = corpus()
    answers = [json.loads(x) for x in Path(args.answers).read_text("utf-8").splitlines() if x]
    verdicts = {}
    if Path(args.judged).exists():
        for x in Path(args.judged).read_text("utf-8").splitlines():
            if x:
                v = json.loads(x)
                verdicts[v["id"]] = v
    model = answers[0]["model"]
    judge_model = next(iter(verdicts.values()))["judge"] if verdicts else "none"
    lines = [
        "# Answer quality",
        "",
        f"Agent: ADK, `{model}`, semantic retrieval (EmbeddingGemma vectors; keyword as the "
        f"fallback; permissions applied inside both). Judge: `{judge_model}`. Every rate has a 95% "
        "Wilson interval. Regenerate with `python scripts/evaluate_answers.py report "
        "runs/answers.jsonl runs/judged.jsonl`.",
        "",
        "## Questions the asker may read",
        "",
        "| Questions | Answered | Correct | Grounded (of answered) | Cites the right page |",
        "|:--|--:|--:|--:|--:|",
    ]
    for label, name in (("written", "As written"), ("para", "Paraphrased"), (None, "All")):
        rs = [
            a
            for a in answers
            if a["kind"] == "answerable" and not a["error"] and (label is None or a["set"] == label)
        ]
        answered = [a for a in rs if not a["refused"]]
        correct = sum(verdicts.get(f"real:{a['id']}", {}).get("correct", False) for a in answered)
        grounded = sum(verdicts.get(f"real:{a['id']}", {}).get("grounded", False) for a in answered)
        cites = sum(any(c.split("#")[0] == a["source"] for c in a["citations"]) for a in rs)
        lines.append(
            f"| {name} | {pct(len(answered), len(rs))} | {pct(correct, len(rs))} | "
            f"{pct(grounded, len(answered))} | {pct(cites, len(rs))} |"
        )
    every = [a for a in answers if not a["error"]]
    removed = sum(len(a["removed_citations"]) for a in every)
    lines += [
        "",
        "*Correct* counts a refusal as wrong. *Grounded*: every claim is in the passages the "
        "agent was shown; an ungrounded answer is a hallucination even when it happens to be "
        f"true. Citations of passages the agent was never shown, removed in code: **{removed}**.",
    ]
    wrong = [
        a
        for a in answers
        if a["kind"] == "answerable"
        and not a["error"]
        and (a["refused"] or not verdicts.get(f"real:{a['id']}", {}).get("correct"))
    ]
    if wrong:
        lines += ["", "Every answer not judged correct:", ""]
        for a in wrong:
            why = (
                "refused"
                if a["refused"]
                else verdicts.get(f"real:{a['id']}", {}).get("reason", "not judged")
            )
            lines.append(f"- {a['set']}: {a['question']} ({why})")

    least = min(access.user_groups, key=lambda u: (len(access.user_groups[u]), u))
    unreadable = [c.text for c in chunks if c.group not in access.groups_of(least)]
    restricted = [a for a in answers if a["kind"] == "restricted" and not a["error"]]
    leaked = [a for a in restricted if repeats_any(a["text"], unreadable)]
    absent = [a for a in answers if a["kind"] == "absent" and not a["error"]]
    lines += [
        "",
        "## Questions the agent should not answer",
        "",
        "| Questions | Asked by | Refused | Answer repeats restricted text |",
        "|:--|:--|--:|--:|",
        f"| About pages the asker may not read | {least} | "
        f"{pct(sum(a['refused'] for a in restricted), len(restricted))} | {len(leaked)} |",
        f"| Not in the handbook | the auditor | "
        f"{pct(sum(a['refused'] for a in absent), len(absent))} | n/a |",
    ]
    answered_anyway = [a for a in restricted + absent if not a["refused"]]
    if answered_anyway:
        lines += ["", "Answered anyway:", ""]
        lines += [f"- ({a['kind']}) {a['question']}: {a['text'][:200]}" for a in answered_anyway]

    controls = [v for v in verdicts.values() if v["kind"] != "real"]
    if controls:
        lines += [
            "",
            "## Checking the judge",
            "",
            "Before trusting its verdicts, the judge was given answers known to be wrong.",
            "",
            "| Known-wrong answer | Judged incorrect |",
            "|:--|--:|",
        ]
        for kind, name in (
            ("swapped", "An answer to a different question"),
            ("numbers_changed", "The real answer with every number changed"),
        ):
            vs = [v for v in controls if v["kind"] == kind]
            lines.append(f"| {name} | {pct(sum(not v['correct'] for v in vs), len(vs))} |")
    secs = sorted(a["seconds"] for a in every)
    calls = sorted(a["model_calls"] for a in every)
    if secs:
        lines += [
            "",
            f"Median {secs[len(secs) // 2]:.1f} s and {calls[len(calls) // 2]} model "
            "calls per question. Errors: "
            f"{sum(bool(a['error']) for a in answers)}.",
        ]
    text = "\n".join(lines) + "\n"
    Path(args.md).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default=AGENT_MODEL)
    r.add_argument("--out", default="runs/answers.jsonl")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--rpm", type=float, default=6, help="questions per minute")
    j = sub.add_parser("judge")
    j.add_argument("--judge", default=JUDGE_MODEL)
    j.add_argument("--answers", default="runs/answers.jsonl")
    j.add_argument("--out", default="runs/judged.jsonl")
    j.add_argument("--resume", action="store_true")
    s = sub.add_parser("report")
    s.add_argument("answers")
    s.add_argument("judged")
    s.add_argument("--md", default=str(ROOT / "reports" / "answers.md"))
    args = p.parse_args()
    if args.cmd == "run":
        return asyncio.run(run(args))
    return judge(args) if args.cmd == "judge" else report(args)


if __name__ == "__main__":
    sys.exit(main())
