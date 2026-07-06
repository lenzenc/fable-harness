"""Durable session/run persistence — a couple of Postgres tables (see
harness/memory/init.sql), no ORM. This is meant to be an auditable ledger of
what ran and when, not a feature-rich session framework: every turn a
session had is a row in `run_records`, kept forever (compliance-friendly),
distinct from the temporal *memory* facts in pgvector_store.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import asyncpg


@dataclass
class RunRecord:
    session_id: str
    request: str
    output: str | None
    stop_reason: str
    success: bool
    created_at: datetime | None = None


@runtime_checkable
class SessionStore(Protocol):
    async def ensure_session(self, session_id: str, user_scope: str, org_scope: str) -> None: ...

    async def record_run(self, record: RunRecord) -> None: ...

    async def history(self, session_id: str, limit: int = 20) -> list[RunRecord]: ...


class NullSessionStore:
    """No-op SessionStore — used when no database is configured/reachable so
    the harness degrades gracefully instead of failing a run over a
    non-essential audit write."""

    async def ensure_session(self, session_id: str, user_scope: str, org_scope: str) -> None:
        return None

    async def record_run(self, record: RunRecord) -> None:
        return None

    async def history(self, session_id: str, limit: int = 20) -> list[RunRecord]:
        return []


class PostgresSessionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresSessionStore":
        pool = await asyncpg.create_pool(dsn)
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def ensure_session(self, session_id: str, user_scope: str, org_scope: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (session_id, user_scope, org_scope)
                VALUES ($1, $2, $3)
                ON CONFLICT (session_id) DO NOTHING
                """,
                session_id,
                user_scope,
                org_scope,
            )

    async def record_run(self, record: RunRecord) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_records (session_id, request, output, stop_reason, success)
                VALUES ($1, $2, $3, $4, $5)
                """,
                record.session_id,
                record.request,
                record.output,
                record.stop_reason,
                record.success,
            )

    async def history(self, session_id: str, limit: int = 20) -> list[RunRecord]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, request, output, stop_reason, success, created_at
                FROM run_records WHERE session_id = $1
                ORDER BY created_at DESC LIMIT $2
                """,
                session_id,
                limit,
            )
        return [RunRecord(**dict(r)) for r in rows]
