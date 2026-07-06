"""Stub MCP retrieval server: trivial keyword search behind the MCP tool
interface. The seam is the MCP interface itself — swap this module out for a
LlamaIndex-backed hybrid dense+BM25 server with a reranker later without
touching the harness at all (see harness/tools/mcp_client.py, the harness's
MCP client/toolset side).

Run standalone for manual testing:
    uv run python -m harness.retrieval.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("retrieval")

# Deliberately tiny in-memory demo corpus, standing in for a real document
# store — enough to make the orchestrator-workers advisor-prep flow
# (harness/flows/orchestrator.py) and citation grounding demonstrable.
_CORPUS: list[dict[str, str]] = [
    {"id": "doc-1", "text": "Client X prefers annuity discussions in person, not over email."},
    {"id": "doc-2", "text": "Client X's portfolio review cadence is quarterly; the last review was in March."},
    {"id": "doc-3", "text": "Product policy: variable annuities require a suitability questionnaire before sale."},
    {"id": "doc-4", "text": "Market context: fixed income yields have been declining through the quarter."},
]


@mcp.tool()
def search(query: str, k: int = 5) -> list[dict[str, str | int]]:
    """Keyword-match `query` against the demo corpus. Returns up to k chunks,
    each with its id, text, and a naive relevance score (matched-term
    count) — the citation-grounding primitive downstream flows attribute
    claims back to."""
    terms = [t.lower() for t in query.split() if t]
    scored = []
    for doc in _CORPUS:
        text_lower = doc["text"].lower()
        score = sum(1 for t in terms if t in text_lower)
        if score > 0:
            scored.append({**doc, "score": score})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


@mcp.tool()
def fetch_chunk(chunk_id: str) -> dict[str, str] | None:
    """Fetch a single chunk by id — lets a caller verify a citation."""
    return next((doc for doc in _CORPUS if doc["id"] == chunk_id), None)


if __name__ == "__main__":
    mcp.run(transport="stdio")
