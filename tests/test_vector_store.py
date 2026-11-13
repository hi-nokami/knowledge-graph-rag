"""Unit tests for vector store module."""

from src.vector_store import PgVectorStore, _cosine_similarity
from src.chunker import TextChunk


def test_cosine_similarity_math():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert round(_cosine_similarity(v1, v2), 4) == 1.0
    assert round(_cosine_similarity(v1, v3), 4) == 0.0
    assert _cosine_similarity([], []) == 0.0


def test_vector_store_insert_and_search(memory_vector_store):
    store = memory_vector_store
    emb_1 = [1.0] + [0.0] * 767
    emb_2 = [0.0, 1.0] + [0.0] * 766

    store.insert_chunk(
        chunk_id="chunk_01",
        doc_id="doc_1",
        content="Alpha content",
        token_count=10,
        embedding=emb_1,
        metadata={"category": "tech"},
    )
    store.insert_chunk(
        chunk_id="chunk_02",
        doc_id="doc_1",
        content="Beta content",
        token_count=12,
        embedding=emb_2,
        metadata={"category": "finance"},
    )

    # Search with emb_1 as query
    results = store.similarity_search(query_embedding=emb_1, top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_01"
    assert round(results[0].score, 2) == 1.0

    # Search with metadata filter
    filtered = store.similarity_search(query_embedding=emb_1, top_k=2, filter_meta={"category": "finance"})
    assert len(filtered) == 1
    assert filtered[0].chunk_id == "chunk_02"


def test_vector_store_lookup_by_ids(memory_vector_store):
    store = memory_vector_store
    emb = [0.5] * 768

    store.insert_chunk(
        chunk_id="chunk_a",
        doc_id="doc_a",
        content="Text A",
        token_count=5,
        embedding=emb,
    )
    store.insert_chunk(
        chunk_id="chunk_b",
        doc_id="doc_b",
        content="Text B",
        token_count=6,
        embedding=emb,
    )

    item = store.get_chunk_by_id("chunk_a")
    assert item is not None
    assert item["content"] == "Text A"

    items = store.get_chunks_by_ids(["chunk_a", "chunk_b", "nonexistent"])
    assert len(items) == 2
    ids = {it["chunk_id"] for it in items}
    assert ids == {"chunk_a", "chunk_b"}
