"""Unit tests for FastAPI REST API endpoints."""

import pytest
from fastapi import HTTPException
from src.api import app, health_check, query_rag, index_document, QueryRequest, IndexRequest
import src.api


@pytest.fixture(autouse=True)
def setup_api_pipeline(mock_pipeline):
    """Ensure API uses the mock pipeline for all endpoint tests."""
    src.api.pipeline = mock_pipeline
    return mock_pipeline


def test_api_routes_registered():
    """Verify all required API endpoints are properly registered on the FastAPI app."""
    routes = {r.path: r.methods for r in app.routes if hasattr(r, "methods")}
    assert "/health" in routes
    assert "GET" in routes["/health"]
    assert "/query" in routes
    assert "POST" in routes["/query"]
    assert "/index" in routes
    assert "POST" in routes["/index"]


def test_health_endpoint():
    """Verify /health endpoint returns valid system status."""
    res = health_check()
    assert res.status in ["healthy", "degraded"]
    assert res.in_memory_mode is True
    assert res.ollama_model is not None


def test_index_and_query_endpoints():
    """Verify /index and /query logic."""
    # 1. Index document
    idx_req = IndexRequest(
        doc_id="api_doc_01",
        text="Acme Corp is a subsidiary of ParentCo. GlobalParts supplies Acme Corp.",
        metadata={"source": "api_test"},
    )
    report = index_document(idx_req)
    assert report.doc_id == "api_doc_01"
    assert report.total_chunks >= 1

    # 2. Query document
    q_req = QueryRequest(question="Who owns Acme Corp?")
    res = query_rag(q_req)
    assert res.question == "Who owns Acme Corp?"
    assert len(res.answer) > 0
    assert res.latency_ms > 0
    assert res.citation_report.is_valid is True


def test_validation_errors():
    """Verify validation errors raise HTTPException 400."""
    with pytest.raises(HTTPException) as exc_info:
        query_rag(QueryRequest(question="   "))
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        index_document(IndexRequest(doc_id="", text="Sample"))
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        index_document(IndexRequest(doc_id="doc1", text=""))
    assert exc_info.value.status_code == 400
