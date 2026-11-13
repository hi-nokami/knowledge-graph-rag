"""Unit tests for recursive character text chunker."""

import pytest
from src.chunker import RecursiveCharacterChunker, TextChunk


def test_chunker_deterministic_ids():
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.split_document(doc_id="sec_doc", text=text)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"sec_doc_ch{idx:05d}"
        assert c.doc_id == "sec_doc"
        assert c.chunk_index == idx
        assert len(c.content) <= 50
        assert c.token_count > 0


def test_chunker_overlap_validation():
    with pytest.raises(ValueError, match="strictly less than"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_empty_input():
    chunker = RecursiveCharacterChunker()
    chunks = chunker.split_document(doc_id="empty", text="")
    assert chunks == []


def test_chunker_metadata_propagation():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    meta = {"source": "sec_filing", "year": 2023}
    chunks = chunker.split_document(doc_id="meta_doc", text="Sample text for testing metadata propagation.", metadata=meta)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["source"] == "sec_filing"
    assert chunk.metadata["year"] == 2023
    assert chunk.metadata["doc_id"] == "meta_doc"
    assert chunk.metadata["chunk_index"] == 0
