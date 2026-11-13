"""Unit tests for unified hybrid retriever."""

from src.hybrid_retriever import HybridRetriever
from src.router import RouteTarget


def test_hybrid_retriever_routing_and_execution(mock_pipeline):
    pipe = mock_pipeline
    retriever = pipe.retriever

    # Ingest test document first
    doc_text = "Acme Corp is a subsidiary of ParentCo. GlobalParts supplies Acme Corp and Baker Industries."
    pipe.ingest_document("test_doc", doc_text)

    # 1. Force VECTOR route
    res_vec = retriever.retrieve("Acme Corp subsidiary", force_route=RouteTarget.VECTOR)
    assert res_vec.route.target == RouteTarget.VECTOR
    assert len(res_vec.chunks) > 0
    assert len(res_vec.valid_chunk_ids) > 0

    # 2. Force GRAPH route
    res_graph = retriever.retrieve("Which supplier supplies both Acme Corp and Baker Industries?", force_route=RouteTarget.GRAPH)
    assert res_graph.route.target == RouteTarget.GRAPH
    assert len(res_graph.graph_facts) > 0

    # 3. BOTH route
    res_both = retriever.retrieve("Tell me about Acme Corp and its suppliers.", force_route=RouteTarget.BOTH)
    assert res_both.route.target == RouteTarget.BOTH
    assert len(res_both.chunks) > 0
