"""End-to-end ingestion and question-answering pipeline orchestrator."""

import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.chunker import RecursiveCharacterChunker, TextChunk
from src.embeddings import OllamaEmbeddings
from src.vector_store import PgVectorStore
from src.neo4j_client import Neo4jClient
from src.extractor import EntityExtractor
from src.entity_resolver import EntityResolver
from src.hybrid_retriever import HybridRetriever, HybridRetrievalResult
from src.generator import AnswerGenerator
from src.citation_validator import CitationValidator, CitationValidationResult
from src.router import RouteTarget, RouteResult


class IngestionReport(BaseModel):
    """Metrics produced during document ingestion."""

    doc_id: str
    total_chunks: int
    total_entities: int
    total_relationships: int
    duration_seconds: float


class QueryResponse(BaseModel):
    """Full end-to-end response for a user query."""

    question: str
    answer: str
    route: RouteResult
    graph_facts: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    citation_report: CitationValidationResult
    latency_ms: float


class KnowledgeGraphRAGPipeline:
    """Master pipeline orchestrator combining Vector RAG and Graph RAG."""

    def __init__(
        self,
        vector_store: PgVectorStore | None = None,
        neo4j_client: Neo4jClient | None = None,
        embeddings: OllamaEmbeddings | None = None,
        extractor: EntityExtractor | None = None,
        retriever: HybridRetriever | None = None,
        generator: AnswerGenerator | None = None,
        validator: CitationValidator | None = None,
    ):
        """Initialize all pipeline components."""
        self.vector_store = vector_store or PgVectorStore()
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.embeddings = embeddings or OllamaEmbeddings()
        self.chunker = RecursiveCharacterChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.extractor = extractor or EntityExtractor()
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            neo4j_client=self.neo4j_client,
            embeddings=self.embeddings,
        )
        self.generator = generator or AnswerGenerator()
        self.validator = validator or CitationValidator()

    def initialize_storage(self) -> dict[str, bool]:
        """Initialize both PostgreSQL schema and Neo4j constraints."""
        pg_ok = self.vector_store.init_db()
        neo_ok = self.neo4j_client.init_schema()
        return {"postgres_initialized": pg_ok, "neo4j_initialized": neo_ok}

    def ingest_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionReport:
        """Execute dual-store ingestion for a document text."""
        start_time = time.perf_counter()

        # 1. Chunk text
        chunks = self.chunker.split_document(doc_id, text, metadata=metadata)
        if not chunks:
            return IngestionReport(
                doc_id=doc_id,
                total_chunks=0,
                total_entities=0,
                total_relationships=0,
                duration_seconds=0.0,
            )

        # 2. Embed chunks and insert into pgvector
        chunk_texts = [c.content for c in chunks]
        embeddings = self.embeddings.embed_batch(chunk_texts)
        self.vector_store.insert_chunks(chunks, embeddings)

        # 3. Extract knowledge graph elements from each chunk
        total_entities = 0
        total_relationships = 0

        for chunk in chunks:
            extract_res = self.extractor.extract_from_chunk(chunk.chunk_id, chunk.content)

            # Ingest entities into Neo4j
            mentioned_canons = []
            for entity in extract_res.entities:
                canon = self.neo4j_client.merge_entity(entity)
                mentioned_canons.append((entity.type, canon))
                total_entities += 1

            # Ingest relationships into Neo4j
            for rel in extract_res.relationships:
                self.neo4j_client.merge_relationship(rel, source_chunk_id=chunk.chunk_id)
                total_relationships += 1

            # Link DocumentChunk node in graph
            self.neo4j_client.merge_document_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=doc_id,
                mentioned_canonical_entities=mentioned_canons,
            )

            # Record entity mentions in vector store
            mentions_for_store = [(canon, etype.value) for etype, canon in mentioned_canons]
            self.vector_store.record_entity_mentions(chunk.chunk_id, mentions_for_store)

        duration = time.perf_counter() - start_time
        return IngestionReport(
            doc_id=doc_id,
            total_chunks=len(chunks),
            total_entities=total_entities,
            total_relationships=total_relationships,
            duration_seconds=round(duration, 3),
        )

    def query(
        self,
        question: str,
        force_route: RouteTarget | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Execute end-to-end grounded question answering."""
        start_time = time.perf_counter()

        # 1. Retrieve
        retrieval_res: HybridRetrievalResult = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            force_route=force_route,
        )

        # 2. Generate
        raw_answer = self.generator.generate_answer(
            question=question,
            graph_facts=retrieval_res.graph_facts,
            chunks=retrieval_res.chunks,
        )

        # 3. Validate citations
        citation_report = self.validator.validate(
            answer=raw_answer,
            valid_chunk_ids=retrieval_res.valid_chunk_ids,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return QueryResponse(
            question=question,
            answer=citation_report.cleaned_answer,
            route=retrieval_res.route,
            graph_facts=retrieval_res.graph_facts,
            retrieved_chunk_ids=retrieval_res.valid_chunk_ids,
            citation_report=citation_report,
            latency_ms=round(latency_ms, 2),
        )
