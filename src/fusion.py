"""Reciprocal Rank Fusion (RRF) module combining vector and graph retrieval rankings."""

from typing import Any
from pydantic import BaseModel, Field


class FusedChunk(BaseModel):
    """Chunk result ranked via Reciprocal Rank Fusion."""

    chunk_id: str
    doc_id: str = Field(default="")
    content: str
    score: float = Field(description="Reciprocal Rank Fusion score")
    vector_rank: int | None = Field(default=None, description="1-based rank in vector results")
    graph_rank: int | None = Field(default=None, description="1-based rank in graph backing results")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReciprocalRankFusion:
    """Combines ranked lists from multiple retrieval systems using RRF."""

    def __init__(self, k: int = 60):
        """Initialize RRF.

        Args:
            k: Smoothing constant (standard default: 60).
        """
        self.k = k

    def fuse(
        self,
        vector_results: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        top_k: int = 8,
    ) -> list[FusedChunk]:
        """Fuse vector and graph chunk retrieval lists.

        Args:
            vector_results: List of dicts with 'chunk_id', 'content', and optional 'doc_id', 'metadata'.
            graph_results: List of backing chunk dicts with 'chunk_id', 'content', and optional 'doc_id', 'metadata'.
            top_k: Maximum number of fused items to return.

        Returns:
            Ranked list of FusedChunk objects sorted descending by RRF score.
        """
        scores: dict[str, float] = {}
        v_ranks: dict[str, int] = {}
        g_ranks: dict[str, int] = {}
        chunk_data: dict[str, dict[str, Any]] = {}

        # 1. Process Vector results (1-based rank)
        for rank, item in enumerate(vector_results, start=1):
            cid = item["chunk_id"]
            if cid not in v_ranks:
                v_ranks[cid] = rank
                scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.k + rank))
                chunk_data[cid] = item

        # 2. Process Graph results (1-based rank)
        for rank, item in enumerate(graph_results, start=1):
            cid = item["chunk_id"]
            if cid not in g_ranks:
                g_ranks[cid] = rank
                scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.k + rank))
                if cid not in chunk_data:
                    chunk_data[cid] = item

        # 3. Sort by fused RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        fused: list[FusedChunk] = []
        for cid in sorted_ids[:top_k]:
            data = chunk_data[cid]
            fused.append(
                FusedChunk(
                    chunk_id=cid,
                    doc_id=data.get("doc_id", ""),
                    content=data.get("content", ""),
                    score=scores[cid],
                    vector_rank=v_ranks.get(cid),
                    graph_rank=g_ranks.get(cid),
                    metadata=data.get("metadata", {}),
                )
            )

        return fused
