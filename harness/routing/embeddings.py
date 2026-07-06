"""OpenAI dense encoder factory — the tier-1 semantic router's fast path,
and the same model backs pgvector memory embeddings so there's exactly one
embedding space in the harness, not two.

This is network-backed: every call is a round-trip to OpenAI's embeddings
API and requires `OPENAI_API_KEY` to be set (see
harness/config/settings.py:get_settings(), which bridges the
HARNESS_OPENAI_API_KEY secret into that unprefixed env var — the same
pattern used for ANTHROPIC_API_KEY). `OpenAIEncoder` is a `DenseEncoder`
subclass (from semantic-router), so it drops straight into
`TieredRouter`/`SemanticRouter` with no changes on the routing side.
"""

from __future__ import annotations

from functools import lru_cache

from semantic_router.encoders import OpenAIEncoder
from semantic_router.encoders.base import DenseEncoder

# text-embedding-3-small: cheapest current-gen OpenAI embedding model,
# 1536-dim — matches the `vector(1536)` column in harness/memory/init.sql
# and semantic-router's own default score threshold (0.3) for this model.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


@lru_cache
def get_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> DenseEncoder:
    """Cached per model_name — call sites (routing, memory) should pass the
    identical string (see harness/cli.py's settings.embedding_model_id) so
    they share one cached client instance rather than each constructing
    their own."""
    return OpenAIEncoder(name=model_name)
