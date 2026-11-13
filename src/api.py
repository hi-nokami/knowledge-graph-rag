"""FastAPI application exposing REST endpoints for query, indexing, and health checks."""

from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings
from src.pipeline import KnowledgeGraphRAGPipeline, IngestionReport, QueryResponse
from src.router import RouteTarget

# Global pipeline instance
pipeline = KnowledgeGraphRAGPipeline()

app = FastAPI(
    title="Knowledge Graph RAG API",
    description="Hybrid Knowledge Graph + Vector RAG service using PostgreSQL pgvector and Neo4j",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(description="Natural language question to answer")
    force_route: Optional[RouteTarget] = Field(default=None, description="Optional forced route override (VECTOR, GRAPH, BOTH)")
    top_k: Optional[int] = Field(default=None, description="Number of fused chunks to retrieve")


class IndexRequest(BaseModel):
    """Payload for POST /index."""

    doc_id: str = Field(description="Unique document identifier")
    text: str = Field(description="Verbatim document content to chunk, embed, and extract")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional document metadata")


class HealthResponse(BaseModel):
    """Payload for GET /health."""

    status: str
    postgres_active: bool
    neo4j_active: bool
    ollama_model: str
    in_memory_mode: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verify system health and active database connections."""
    pg_active = not pipeline.vector_store.in_memory
    neo_active = not pipeline.neo4j_client.in_memory

    status = "healthy" if (pg_active and neo_active) else "degraded"
    return HealthResponse(
        status=status,
        postgres_active=pg_active,
        neo4j_active=neo_active,
        ollama_model=settings.generator_model,
        in_memory_mode=(pipeline.vector_store.in_memory or pipeline.neo4j_client.in_memory),
    )


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Execute end-to-end question answering pipeline."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = pipeline.query(
            question=request.question,
            force_route=request.force_route,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/index", response_model=IngestionReport)
def index_document(request: IndexRequest):
    """Index a document across pgvector and Neo4j."""
    if not request.doc_id.strip():
        raise HTTPException(status_code=400, detail="doc_id cannot be empty.")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty.")

    try:
        report = pipeline.ingest_document(
            doc_id=request.doc_id,
            text=request.text,
            metadata=request.metadata,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=True)
