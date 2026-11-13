"""Unit tests for end-to-end pipeline orchestrator."""

def test_pipeline_ingestion_and_query(mock_pipeline, sample_corpus):
    pipe = mock_pipeline

    # 1. Ingest corpus
    for doc in sample_corpus:
        report = pipe.ingest_document(
            doc_id=doc["doc_id"],
            text=doc["text"],
            metadata=doc["metadata"],
        )
        assert report.total_chunks > 0
        assert report.total_entities >= 0

    # 2. Query multi-hop question
    q = "Which supplier supplies both Acme Corp and Baker Industries?"
    response = pipe.query(q)

    assert response.question == q
    assert response.latency_ms > 0
    assert len(response.answer) > 0
    # Citations report should be valid
    assert response.citation_report.is_valid is True
    # Backing chunk IDs or facts should be present
    assert len(response.retrieved_chunk_ids) > 0 or len(response.graph_facts) > 0
