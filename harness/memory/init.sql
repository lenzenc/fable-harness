-- Harness memory + session schema. Facts are never hard-deleted, only
-- invalidated (valid_to set) — "what did the agent know and when" must
-- stay answerable, a compliance question in financial services.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_items (
    id UUID PRIMARY KEY,
    scope TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,  -- text-embedding-3-small; dim locked to EMBEDDING_DIM in harness/routing/embeddings.py
    encoder_id TEXT NOT NULL,         -- guards against silently mixing embedding spaces on encoder swap
    source TEXT NOT NULL DEFAULT 'unknown',
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS memory_items_embedding_idx
    ON memory_items USING hnsw (embedding vector_cosine_ops);

-- Partial index: only live (valid_to IS NULL) rows are ever queried by recall().
CREATE INDEX IF NOT EXISTS memory_items_scope_live_idx
    ON memory_items (scope) WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_scope TEXT NOT NULL,
    org_scope TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    request TEXT NOT NULL,
    output TEXT,
    stop_reason TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS run_records_session_idx ON run_records (session_id, created_at DESC);
