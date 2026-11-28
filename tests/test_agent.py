"""The answering agent, driven by a scripted model: the user comes from the session, never
from the model, and citations of anything not retrieved are removed."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

from clearance.agent import NOT_FOUND, ask, build_agent, check_citations, is_refusal
from clearance.chunking import chunk_all
from clearance.corpus import Access, load
from clearance.judge import has_numbers, parse_verdict, perturb_numbers, repeats_any
from clearance.retrieval import Retriever

FIXTURE = Path(__file__).parent / "fixture"


class ScriptedLlm(BaseLlm):
    """Searches once (trying to search as someone else), then answers with the first
    passage's id and an invented one."""

    query: str = "Northwind settlement cap"
    seen: list[Any] = Field(default_factory=list)

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        results = [
            p.function_response.response
            for c in llm_request.contents
            for p in c.parts or []
            if p.function_response
        ]
        if not results:
            call = types.FunctionCall(
                name="search_handbook", args={"query": self.query, "user": "tom"}
            )
            part = types.Part(function_call=call)
        else:
            passages = results[-1]["passages"]
            self.seen.append(passages)
            if not passages:
                part = types.Part(text=NOT_FOUND)
            else:
                part = types.Part(
                    text=f"The cap is set out here [{passages[0]['id']}]. "
                    "Also see [legal/secret-memo.md#h1]."
                )
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


@pytest.fixture
def setup():
    access = Access.load(FIXTURE / "access.yaml")
    chunks = chunk_all(load(FIXTURE, access).docs, "headings_titled")
    return access, Retriever(chunks, access)


def test_the_agent_searches_as_the_session_user_and_cannot_cite_what_it_was_not_shown(setup):
    _, retriever = setup
    llm = ScriptedLlm(model="scripted")
    tom = asyncio.run(ask(build_agent(llm, retriever), "What caps the settlement?", "tom"))
    assert tom.citations and tom.citations[0].startswith("legal/")
    assert tom.removed_citations == ["legal/secret-memo.md#h1"]
    assert "secret-memo" not in tom.text and tom.searches == ["Northwind settlement cap"]
    assert set(tom.retrieved) >= set(tom.citations)

    llm = ScriptedLlm(model="scripted")
    sam = asyncio.run(ask(build_agent(llm, retriever), "What caps the settlement?", "sam"))
    # The model asked to search as tom; the tool searched as sam, the session's user.
    assert all(not p["id"].startswith("legal/") for p in llm.seen[0])
    assert all(not c.startswith("legal/") for c in sam.citations + sam.retrieved)


def test_an_unknown_user_gets_the_not_found_answer(setup):
    _, retriever = setup
    a = asyncio.run(ask(build_agent(ScriptedLlm(model="s"), retriever), "anything", "mallory"))
    assert a.refused and a.retrieved == [] and a.model_calls == 2


def test_citation_check_and_refusal_detection():
    text, kept, removed = check_citations("A [a.md#h1]. B [b.md#h2]. C [a.md#h1].", ["a.md#h1"])
    assert kept == ["a.md#h1"] and removed == ["b.md#h2"] and "[b.md#h2]" not in text
    assert is_refusal("I couldn't find that in the handbook pages you can access.")
    assert not is_refusal("Parental leave is 16 weeks [p.md#h1].")


def test_judge_helpers():
    assert perturb_numbers("16 weeks from 2024 [a.md#h1]") == "19 weeks from 2027 [a.md#h1]"
    assert has_numbers("16 weeks") and not has_numbers("see [a.md#h1]")
    v = parse_verdict('Sure: {"correct": true, "grounded": false, "reason": "extra claim"}')
    assert v.correct and not v.grounded and v.reason == "extra claim"
    with pytest.raises(ValueError):
        parse_verdict("no verdict here")
    secret = "the settlement with Northwind is capped at two million dollars in total"
    assert repeats_any(
        "We learned the settlement with Northwind is capped at two million.", [secret]
    )
    assert not repeats_any("Settlements are handled by Legal.", [secret])
