"""The answering agent: an ADK agent that answers from the handbook pages the asker may read.

Its one tool searches as the employee who is asking. The user comes from session state,
set by the application from the verified identity, never from the model: the model can
choose what to search for, but not whose permissions apply. Retrieval enforces the
permissions (see `index.py`), so the model is never shown a passage the user may not read.

What the code checks, not the prompt: every citation in an answer must be a passage that
was retrieved for this user in this conversation. Citations of anything else are removed
and counted, so a fabricated or remembered source cannot reach the user looking cited.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from clearance.index import Hit

APP = "clearance"
NOT_FOUND = "I couldn't find that in the handbook pages you can access."
CITATION = re.compile(r"\[([^\[\]\s]+\.md#[^\[\]\s]+)\]")
PASSAGE_CHARS = 1500

INSTRUCTION = f"""You answer employees' questions using the company handbook.

1. Always call `search_handbook` before answering. You may search up to three times with
   different wording if the first results do not answer the question.
2. Answer only from the passages the tool returns. Do not use anything you know from
   elsewhere, and do not guess.
3. After each sentence that uses a passage, cite the passage id in square brackets, for
   example [legal/record-retention-policy.md#h2]. Cite only ids the tool returned.
4. If the passages do not answer the question, reply with exactly this sentence and nothing
   else: "{NOT_FOUND}"
5. Never mention, describe or speculate about documents or topics you were not shown.
6. Keep answers short: two to five sentences.
"""

Retrieve = Callable[[str, str], list[Hit]]


def build_agent(model: Any, retrieve: Retrieve) -> LlmAgent:
    def search_handbook(query: str, tool_context: ToolContext) -> dict[str, Any]:
        """Search the handbook pages this employee is allowed to read.

        Args:
            query: what to look for, in your own words.

        Returns:
            Passages, each with an id to cite, the page, its heading and its text.
        """
        user = str(tool_context.state.get("user", ""))  # set by the application
        hits = retrieve(query, user)
        seen = list(tool_context.state.get("retrieved", []))
        seen += [h.chunk.id for h in hits if h.chunk.id not in seen]
        tool_context.state["retrieved"] = seen
        return {
            "passages": [
                {
                    "id": h.chunk.id,
                    "page": h.chunk.title,
                    "heading": h.chunk.heading,
                    "text": h.chunk.text[:PASSAGE_CHARS],
                }
                for h in hits
            ]
        }

    return LlmAgent(
        name="handbook_assistant",
        model=model,
        instruction=INSTRUCTION,
        tools=[search_handbook],
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )


def check_citations(answer: str, retrieved: list[str]) -> tuple[str, list[str], list[str]]:
    """Keep citations of retrieved passages; strip and report any others."""
    allowed = set(retrieved)
    kept, removed = [], []
    for cid in CITATION.findall(answer):
        (kept if cid in allowed else removed).append(cid)
    for cid in removed:
        answer = answer.replace(f"[{cid}]", "")
    return answer.strip(), list(dict.fromkeys(kept)), removed


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower().rstrip(".") in answer.lower()


@dataclass
class Answer:
    question: str
    user: str
    text: str = ""
    citations: list[str] = field(default_factory=list)
    removed_citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    model_calls: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def refused(self) -> bool:
        return is_refusal(self.text)


async def ask(agent: LlmAgent, question: str, user: str) -> Answer:
    """One question, one fresh session, as `user`."""
    sessions = InMemorySessionService()
    runner = Runner(app_name=APP, agent=agent, session_service=sessions)
    session = await sessions.create_session(app_name=APP, user_id=user, state={"user": user})
    out = Answer(question, user)
    said: list[str] = []
    started = time.perf_counter()
    try:
        async for event in runner.run_async(
            user_id=user,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        ):
            if event.author == "user" or not event.content:
                continue
            out.model_calls += 1 if event.content.role == "model" else 0
            for part in event.content.parts or []:
                if part.function_call:
                    out.searches.append(str((part.function_call.args or {}).get("query", "")))
                elif part.text and not part.thought:
                    said.append(part.text)
    except Exception as e:  # an outage or a crash is recorded, not raised
        out.error = f"{type(e).__name__}: {e}"
    finally:
        await runner.close()
    out.seconds = round(time.perf_counter() - started, 2)
    final = await sessions.get_session(app_name=APP, user_id=user, session_id=session.id)
    out.retrieved = list((final.state if final else {}).get("retrieved", []))
    out.text, out.citations, out.removed_citations = check_citations(" ".join(said), out.retrieved)
    return out
