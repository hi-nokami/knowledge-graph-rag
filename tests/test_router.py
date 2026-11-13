"""Unit tests for query router."""

from src.router import QueryRouter, RouteTarget


def test_router_heuristics(mock_router):
    router = mock_router

    # Relational questions -> GRAPH
    res1 = router.route("Which supplier supplies both Acme Corp and Baker Industries?")
    assert res1.target == RouteTarget.GRAPH
    assert res1.confidence > 0.7

    res2 = router.route("Who is the parent company and subsidiary of Acme?")
    assert res2.target == RouteTarget.GRAPH

    # Definition / summary questions -> VECTOR
    res3 = router.route("What is the definition of recursive character splitting?")
    assert res3.target == RouteTarget.VECTOR

    # Mixed / complex questions -> BOTH
    res4 = router.route("Summarize the financial performance of Acme Corp and list all subsidiaries.")
    assert res4.target == RouteTarget.BOTH
