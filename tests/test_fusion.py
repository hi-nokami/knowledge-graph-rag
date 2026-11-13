"""Unit tests for Reciprocal Rank Fusion (RRF)."""

from src.fusion import ReciprocalRankFusion


def test_rrf_scoring_and_corroboration():
    fusion = ReciprocalRankFusion(k=60)

    vector_results = [
        {"chunk_id": "chunk_v1", "content": "Vector top chunk"},
        {"chunk_id": "chunk_shared", "content": "Shared corroboration chunk"},
    ]
    graph_results = [
        {"chunk_id": "chunk_shared", "content": "Shared corroboration chunk"},
        {"chunk_id": "chunk_g1", "content": "Graph only chunk"},
    ]

    fused = fusion.fuse(vector_results, graph_results, top_k=5)

    assert len(fused) == 3
    # Corroborated chunk should rank #1 because it has terms from both vector and graph
    assert fused[0].chunk_id == "chunk_shared"
    assert fused[0].vector_rank == 2
    assert fused[0].graph_rank == 1
    # Check expected score: 1/(60+2) + 1/(60+1)
    expected_score = (1.0 / 62.0) + (1.0 / 61.0)
    assert round(fused[0].score, 6) == round(expected_score, 6)


def test_rrf_top_k_truncation():
    fusion = ReciprocalRankFusion(k=60)
    vec = [{"chunk_id": f"c_{i}", "content": f"Text {i}"} for i in range(10)]
    fused = fusion.fuse(vec, [], top_k=4)
    assert len(fused) == 4
    assert fused[0].chunk_id == "c_0"
