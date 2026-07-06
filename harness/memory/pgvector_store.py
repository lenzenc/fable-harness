"""pgvector-backed MemoryStore: Postgres + pgvector with temporal
valid_from/valid_to columns (see harness/memory/init.sql for the schema).
Facts are closed out via `invalidate` (valid_to = now()), never hard-deleted
— "what did the agent know and when" must stay answerable. The encoder id is
stored alongside every embedding as a guard against silently mixing
embedding spaces if the encoder model is ever swapped.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import asyncpg
from pgvector.asyncpg import register_vector

from harness.memory.store import MemoryItem, MemoryPredicate
from harness.routing.embeddings import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIM, get_encoder


class PgVectorMemoryStore:
    def __init__(self, pool: asyncpg.Pool, encoder_id: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.pool = pool
        self.encoder_id = encoder_id

    @classmethod
    async def connect(cls, dsn: str, encoder_id: str = DEFAULT_EMBEDDING_MODEL) -> "PgVectorMemoryStore":
        async def _init_conn(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        pool = await asyncpg.create_pool(dsn, init=_init_conn)
        return cls(pool, encoder_id=encoder_id)

    async def close(self) -> None:
        await self.pool.close()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        encoder = get_encoder(self.encoder_id)
        vectors = encoder(texts)
        for vec in vectors:
            assert len(vec) == EMBEDDING_DIM, (
                f"encoder '{self.encoder_id}' returned a {len(vec)}-dim vector, "
                f"expected {EMBEDDING_DIM} — embedding_model_id and the memory_items "
                "schema (harness/memory/init.sql) must agree"
            )
        return vectors

    async def remember(self, items: list[MemoryItem], scope: str) -> None:
        if not items:
            return
        # OpenAIEncoder.__call__ is a synchronous network round-trip — offload
        # it so it doesn't block the event loop (mirrors how router.py already
        # offloads its sync SemanticRouter.__call__ via asyncio.to_thread).
        vectors = await asyncio.to_thread(self._embed, [item.content for item in items])
        now = datetime.now(UTC)
        rows = [
            (
                uuid.UUID(item.id),
                scope,
                item.content,
                vec,
                self.encoder_id,
                item.source,
                item.valid_from or now,
            )
            for item, vec in zip(items, vectors, strict=True)
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memory_items (id, scope, content, embedding, encoder_id, source, valid_from)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
                """,
                rows,
            )

    async def recall(self, query: str, scope: str, k: int = 8) -> list[MemoryItem]:
        [vector] = await asyncio.to_thread(self._embed, [query])
        async with self.pool.acquire() as conn:
            records = await conn.fetch(
                """
                SELECT id, content, encoder_id, source, valid_from, valid_to
                FROM memory_items
                WHERE scope = $1 AND valid_to IS NULL
                ORDER BY embedding <=> $2
                LIMIT $3
                """,
                scope,
                vector,
                k,
            )
        return [self._row_to_item(r, scope, valid_to=r["valid_to"]) for r in records]

    async def invalidate(self, predicate: MemoryPredicate) -> int:
        async with self.pool.acquire() as conn, conn.transaction():
            records = await conn.fetch(
                """
                SELECT id, content, encoder_id, source, valid_from
                FROM memory_items
                WHERE scope = $1 AND valid_to IS NULL
                """,
                predicate.scope,
            )
            to_close = [r["id"] for r in records if predicate.matches(self._row_to_item(r, predicate.scope))]
            if not to_close:
                return 0
            await conn.execute("UPDATE memory_items SET valid_to = now() WHERE id = ANY($1::uuid[])", to_close)
        return len(to_close)

    @staticmethod
    def _row_to_item(row: asyncpg.Record, scope: str, valid_to: datetime | None = None) -> MemoryItem:
        return MemoryItem(
            id=str(row["id"]),
            content=row["content"],
            scope=scope,
            encoder_id=row["encoder_id"],
            source=row["source"],
            valid_from=row["valid_from"],
            valid_to=valid_to,
        )
