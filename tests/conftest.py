"""Pytest fixtures for knowledge-graph-rag test suite."""

import sys
from pathlib import Path
import pytest

# Ensure source and _shared directory are on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ROOT_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = ROOT_DIR.parent / "_shared"

for p in [str(ROOT_DIR), str(SRC_DIR), str(SHARED_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.vector_store import PgVectorStore
from src.neo4j_client import Neo4jClient
from src.embeddings import OllamaEmbeddings
from src.extractor import EntityExtractor
from src.router import QueryRouter
from src.generator import AnswerGenerator
from src.citation_validator import CitationValidator
from src.hybrid_retriever import HybridRetriever
from src.pipeline import KnowledgeGraphRAGPipeline


@pytest.fixture
def memory_vector_store():
    """Provides an in-memory PgVectorStore without requiring active PostgreSQL."""
    store = PgVectorStore(in_memory=True)
    store.init_db()
    return store


@pytest.fixture
def memory_neo4j_client():
    """Provides an in-memory Neo4jClient without requiring active Neo4j."""
    client = Neo4jClient(in_memory=True)
    client.init_schema()
    return client


@pytest.fixture
def mock_embeddings():
    """Provides OllamaEmbeddings operating with deterministic offline embeddings."""
    return OllamaEmbeddings(use_fallback_if_unavailable=True, offline_mode=True)


@pytest.fixture
def mock_extractor():
    """Provides EntityExtractor operating with deterministic heuristic extraction."""
    return EntityExtractor(use_heuristics_fallback=True, offline_mode=True)


@pytest.fixture
def mock_router():
    """Provides QueryRouter operating with deterministic heuristic classification."""
    return QueryRouter(use_heuristics_fallback=True, offline_mode=True)


@pytest.fixture
def mock_generator():
    """Provides AnswerGenerator operating with deterministic heuristic synthesis."""
    return AnswerGenerator(use_fallback_if_unavailable=True, offline_mode=True)


@pytest.fixture
def citation_validator():
    """Provides a CitationValidator instance."""
    return CitationValidator()


@pytest.fixture
def mock_pipeline(memory_vector_store, memory_neo4j_client, mock_embeddings, mock_extractor, mock_router, mock_generator, citation_validator):
    """Provides an end-to-end KnowledgeGraphRAGPipeline wired with in-memory stores."""
    retriever = HybridRetriever(
        vector_store=memory_vector_store,
        neo4j_client=memory_neo4j_client,
        embeddings=mock_embeddings,
        router=mock_router,
    )
    pipe = KnowledgeGraphRAGPipeline(
        vector_store=memory_vector_store,
        neo4j_client=memory_neo4j_client,
        embeddings=mock_embeddings,
        extractor=mock_extractor,
        retriever=retriever,
        generator=mock_generator,
        validator=citation_validator,
    )
    pipe.initialize_storage()
    return pipe


@pytest.fixture
def sample_corpus():
    """Sample corporate documents for testing multi-hop reasoning."""
    return [
        {
            "doc_id": "doc_acme_corp",
            "text": "Acme Corp is a subsidiary of ParentCo. GlobalParts entered into a multi-year supplier agreement with both Acme Corp and Baker Industries in 2022. Acme Corp produces advanced energy turbines.",
            "metadata": {"source": "sec_filing_2022"},
        },
        {
            "doc_id": "doc_baker_ind",
            "text": "Baker Industries is an aerospace manufacturing firm. GlobalParts supplies critical hydraulic components to Baker Industries.",
            "metadata": {"source": "annual_report_2023"},
        },
        {
            "doc_id": "doc_parent_co",
            "text": "ParentCo acquired Acme Corp in 2021 to expand its renewable energy footprint across North America.",
            "metadata": {"source": "press_release"},
        },
    ]
