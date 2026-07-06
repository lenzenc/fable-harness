"""Async post-turn memory extraction: a cheap Pydantic AI agent pulls
durable facts out of a completed turn, run *after* the turn finishes (not
inline) so extraction latency/cost never sits on the user-facing response
path. Entirely ablatable — callers (flows/single_agent.py,
flows/orchestrator.py) skip invoking this when `ctx.ablations.memory_on` is
False.
"""

from __future__ import annotations

from contextlib import nullcontext

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from harness.memory.store import MemoryItem

EXTRACTION_INSTRUCTIONS = """Extract durable facts worth remembering about the user or
client from this exchange — preferences, relationship details, decisions made. Skip
anything only true for this one turn (e.g. "the current time is..."). Return an empty
list if nothing durable was said."""


class ExtractedFact(BaseModel):
    content: str


class FactExtractor:
    """`test_model` override mirrors flows/single_agent.py's pattern — lets
    the eval suite and CLI --test-model runs exercise extraction with zero
    token spend and deterministic behavior."""

    def __init__(self, model_id: str, test_model: Model | None = None) -> None:
        self.model_id = model_id
        self.test_model = test_model

    async def extract(self, request: str, response: str, source: str = "post_turn_extraction") -> list[MemoryItem]:
        agent: Agent[None, list[ExtractedFact]] = Agent(
            self.model_id,
            output_type=list[ExtractedFact],
            instructions=EXTRACTION_INSTRUCTIONS,
            defer_model_check=self.test_model is not None,
        )
        override_ctx = agent.override(model=self.test_model) if self.test_model is not None else nullcontext()
        with override_ctx:
            result = await agent.run(f"User: {request}\nAssistant: {response}")
        facts = getattr(result, "output", None)
        if facts is None:
            facts = getattr(result, "data", [])
        return [MemoryItem(content=fact.content, scope="", source=source) for fact in facts]
