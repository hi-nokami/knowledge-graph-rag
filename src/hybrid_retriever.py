"""Unified hybrid retrieval facade coordinating routing, vector search, graph traversal, and RRF fusion."""

from typing import Any
from pydantic import BaseModel, Field

from src.config import settings
from src.embeddings import OllamaEmbeddings
from src.vector_store import PgVectorStore
from src.neo4j_client import Neo4jClient
from src.router import QueryRouter, RouteTarget, RouteResult
from src.graph_retriever import GraphRetriever
from src.fusion import ReciprocalRankFusion, FusedChunk


class HybridRetrievalResult(BaseModel):
    """Encapsulates all results produced by the hybrid retrieval pipeline."""

    query: str
    route: RouteResult
    graph_facts: list[str] = Field(default_factory=list, description="Verbalized facts from graph")
    chunks: list[FusedChunk] = Field(default_factory=list, description="Top fused/retrieved passages")
    valid_chunk_ids: list[str] = Field(default_factory=list, description="All accessible source chunk IDs")


class HybridRetriever:
    """Coordinating facade executing the hybrid KG-RAG retrieval flow."""

    def __init__(
        self,
        vector_store: PgVectorStore,
        neo4j_client: Neo4jClient,
        embeddings: OllamaEmbeddings | None = None,
        router: QueryRouter | None = None,
        fusion: ReciprocalRankFusion | None = None,
    ):
        """Initialize HybridRetriever."""
        self.vector_store = vector_store
        self.neo4j_client = neo4j_client
        self.embeddings = embeddings or OllamaEmbeddings()
        self.router = router or QueryRouter()
        self.fusion = fusion or ReciprocalRankFusion(k=settings.rrf_k)
        self.graph_retriever = GraphRetriever(neo4j_client, vector_store)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        force_route: RouteTarget | None = None,
    ) -> HybridRetrievalResult:
        """Execute hybrid retrieval for a query.

        Args:
            query: User's natural language question.
            top_k: Number of fused chunks to return (defaults to settings.top_k_fused).
            force_route: If provided, overrides semantic routing.

        Returns:
            HybridRetrievalResult with routing decision, verbalized graph facts, and ranked chunks.
        """
        top_k = top_k or settings.top_k_fused

        # 1. Route Query
        if force_route:
            route = RouteResult(target=force_route, confidence=1.0, reasoning="Forced route override")
        else:
            route = self.router.route(query)

        graph_facts: list[str] = []
        graph_backing_chunks: list[dict[str, Any]] = []
        vector_chunks: list[dict[str, Any]] = []

        # 2. Execute Graph Retrieval if ROUTE in (GRAPH, BOTH)
        if route.target in (RouteTarget.GRAPH, RouteTarget.BOTH):
            graph_res = self.graph_retriever.retrieve(query)
            graph_facts = graph_res.verbalized_facts
            graph_backing_chunks = graph_res.backing_chunks

        # 3. Execute Vector Retrieval if ROUTE in (VECTOR, BOTH)
        if route.target in (RouteTarget.VECTOR, RouteTarget.BOTH):
            q_emb = self.embeddings.embed_text(query)
            search_res = self.vector_store.similarity_search(q_emb, top_k=settings.top_k_vector)
            vector_chunks = [
                {
                    "chunk_id": sr.chunk_id,
                    "doc_id": sr.doc_id,
                    "content": sr.content,
                    "score": sr.score,
                    "metadata": sr.metadata,
                }
                for sr in search_res
            ]

        # 4. Rank / Fuse Chunks
        final_chunks: list[FusedChunk] = []

        if route.target == RouteTarget.BOTH:
            final_chunks = self.fusion.fuse(
                vector_results=vector_chunks,
                graph_results=graph_backing_chunks,
                top_k=top_k,
            )
        elif route.target == RouteTarget.VECTOR:
            # Map directly to FusedChunk
            for idx, vc in enumerate(vector_chunks[:top_k], start=1):
                final_chunks.append(
                    FusedChunk(
                        chunk_id=vc["chunk_id"],
                        doc_id=vc.get("doc_id", ""),
                        content=vc["content"],
                        score=vc.get("score", 1.0),
                        vector_rank=idx,
                        graph_rank=None,
                        metadata=vc.get("metadata", {}),
                    )
                )
        elif route.target == RouteTarget.GRAPH:
            # Map directly to FusedChunk
            for idx, gc in enumerate(graph_backing_chunks[:top_k], start=1):
                final_chunks.append(
                    FusedChunk(
                        chunk_id=gc["chunk_id"],
                        doc_id=gc.get("doc_id", ""),
                        content=gc["content"],
                        score=1.0 / (self.fusion.k + idx),
                        vector_rank=None,
                        graph_rank=idx,
                        metadata=gc.get("metadata", {}),
                    )
                )

        # Collect all valid chunk IDs available to generation context
        valid_chunk_ids = list({c.chunk_id for c in final_chunks})
        for fact in graph_facts:
            # Also extract chunk IDs embedded in graph facts [source: <id>]
            import re
            for m in re.finditer(r"\[source:\s*([a-zA-Z0-9_\-]+)\]", fact):
                valid_chunk_ids.append(m.group(1))
        valid_chunk_ids = list(set(valid_chunk_ids))

        return HybridRetrievalResult(
            query=query,
            route=route,
            graph_facts=graph_facts,
            chunks=final_chunks,
            valid_chunk_ids=valid_chunk_ids,
        )
