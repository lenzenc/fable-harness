"""MemoryStore: the harness's memory seam.

Defines the `MemoryItem` shape and the `MemoryStore` Protocol shared by every
backend. The MVP backend (`harness/memory/pgvector_store.py`, Phase 3) is
Postgres+pgvector with temporal `valid_from`/`valid_to` columns — facts are
never hard-deleted, only invalidated, so "what did the agent know and when"
stays answerable (a compliance requirement, not just a nice-to-have).

`NullMemoryStore` here lets the core loop run with memory fully ablated
(`ablations.memory_on = False`) without special-casing None everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4


@dataclass
class MemoryItem:
    content: str
    scope: str  # one of {"user", "session", "org"} — enforced by convention, not type, to keep the seam thin
    embedding: list[float] | None = None
    encoder_id: str | None = None  # which encoder produced `embedding` — guards against silent dim mismatches
    source: str = "unknown"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def is_live(self) -> bool:
        return self.valid_to is None


@dataclass
class MemoryPredicate:
    """A predicate used by `invalidate` to select which live items to close
    out. `matches` receives each candidate MemoryItem and returns bool."""

    scope: str
    matches: Callable[[MemoryItem], bool]


@runtime_checkable
class MemoryStore(Protocol):
    async def remember(self, items: list[MemoryItem], scope: str) -> None: ...

    async def recall(self, query: str, scope: str, k: int = 8) -> list[MemoryItem]: ...

    async def invalidate(self, predicate: MemoryPredicate) -> int:
        """Close out (set valid_to=now) all live items matching predicate.
        Returns the count invalidated. Never hard-deletes."""
        ...


class NullMemoryStore:
    """No-op MemoryStore for `memory_on=False` ablation runs and for Phase 1
    before the pgvector backend exists."""

    async def remember(self, items: list[MemoryItem], scope: str) -> None:
        return None

    async def recall(self, query: str, scope: str, k: int = 8) -> list[MemoryItem]:
        return []

    async def invalidate(self, predicate: MemoryPredicate) -> int:
        return 0
