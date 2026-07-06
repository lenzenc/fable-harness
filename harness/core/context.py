"""RunContext: the object threaded through every layer of a single run.

Carries the request, session/scope identity, the active OTel span, and the
ablation flags that make every harness feature independently toggleable —
per the design thesis, model upgrades reprice harness strategies, so every
component (memory, critic, router tier) must be disable-able per run and
re-benchmarked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opentelemetry.trace import Span


@dataclass
class AblationFlags:
    memory_on: bool = True
    critic_on: bool = True
    # Max routing tier allowed to run: 1 = embedding fast-path only,
    # 2 = + function-call router, 3 = + LLM triage fallback.
    router_tier: int = 3


@dataclass
class RunContext:
    request: str
    session_id: str
    user_scope: str = "default"
    org_scope: str = "default"
    ablations: AblationFlags = field(default_factory=AblationFlags)
    span: "Span | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def stamp(self, key: str, value: Any) -> None:
        """Set a `harness.*` attribute on the active span, if any, and mirror
        it into metadata so it's visible even when no span is active (e.g.
        under TestModel-driven unit tests)."""
        self.metadata[key] = value
        if self.span is not None:
            self.span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
