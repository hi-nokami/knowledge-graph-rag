"""PostgreSQL + pgvector vector store client with HNSW indexing and fallback in-memory capability."""

import json
import math
from typing import Any, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.chunker import TextChunk


class SearchResult(BaseModel):
    """Result item returned by vector similarity search."""

    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    score: float = Field(description="Cosine similarity score (0.0 to 1.0)")
    metadata: dict[str, Any] = Field(default_factory=dict)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class PgVectorStore:
    """Vector database client using PostgreSQL with pgvector extension."""

    def __init__(
        self,
        dsn: str | None = None,
        dimension: int = 768,
        in_memory: bool = False,
    ):
        """Initialize the vector store.

        Args:
            dsn: PostgreSQL connection string (defaults to settings.postgres_dsn).
            dimension: Vector dimension (defaults to 768).
            in_memory: If True, operates in in-memory mode without PostgreSQL.
        """
        self.dsn = dsn or settings.postgres_dsn
        self.dimension = dimension
        self.in_memory = in_memory
        
        # In-memory storage structures for testing or fallback
        self._memory_chunks: dict[str, dict[str, Any]] = {}
        self._memory_entity_mentions: list[dict[str, Any]] = []

    def _get_connection(self):
        """Establish connection to PostgreSQL."""
        if self.in_memory:
            return None
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            conn = psycopg2.connect(self.dsn, connect_timeout=2)
            register_vector(conn)
            return conn
        except Exception:
            return None

    def init_db(self) -> bool:
        """Initialize database schema, tables, and HNSW cosine index.

        Returns:
            True if schema initialized in Postgres, False if running in-memory fallback.
        """
        if self.in_memory:
            return False

        conn = self._get_connection()
        if conn is None:
            self.in_memory = True
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(128) UNIQUE NOT NULL,
                    doc_id VARCHAR(128) NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_mentions (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(128) NOT NULL REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
                    entity_canonical_name VARCHAR(256) NOT NULL,
                    entity_type VARCHAR(64) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)

                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id ON document_chunks(doc_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_chunks_metadata ON document_chunks USING gin (metadata);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_canonical ON entity_mentions(entity_canonical_name);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk_id ON entity_mentions(chunk_id);")

                cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding_hnsw 
                ON document_chunks 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
                """)
                conn.commit()
            return True
        except Exception:
            conn.rollback()
            self.in_memory = True
            return False
        finally:
            conn.close()

    def insert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        token_count: int,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert a single chunk and its embedding into vector storage."""
        metadata = metadata or {}

        if self.in_memory:
            self._memory_chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "token_count": token_count,
                "embedding": list(embedding),
                "metadata": metadata,
            }
            return

        conn = self._get_connection()
        if conn is None:
            self.in_memory = True
            self.insert_chunk(chunk_id, doc_id, content, token_count, embedding, metadata)
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO document_chunks (chunk_id, doc_id, content, token_count, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    token_count = EXCLUDED.token_count,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
                """, (chunk_id, doc_id, content, token_count, embedding, json.dumps(metadata)))
                conn.commit()
        except Exception:
            conn.rollback()
            self.in_memory = True
            self.insert_chunk(chunk_id, doc_id, content, token_count, embedding, metadata)
        finally:
            conn.close()

    def insert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Batch insert multiple chunks with their corresponding embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have identical lengths")

        for chunk, emb in zip(chunks, embeddings):
            self.insert_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                token_count=chunk.token_count,
                embedding=emb,
                metadata=chunk.metadata,
            )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_meta: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform cosine similarity search returning top-k nearest chunks."""
        if self.in_memory:
            scored = []
            for item in self._memory_chunks.values():
                # Apply metadata filter if specified
                if filter_meta:
                    match = all(item["metadata"].get(k) == v for k, v in filter_meta.items())
                    if not match:
                        continue
                score = _cosine_similarity(query_embedding, item["embedding"])
                scored.append(
                    SearchResult(
                        chunk_id=item["chunk_id"],
                        doc_id=item["doc_id"],
                        content=item["content"],
                        token_count=item["token_count"],
                        score=score,
                        metadata=item["metadata"],
                    )
                )
            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:top_k]

        conn = self._get_connection()
        if conn is None:
            self.in_memory = True
            return self.similarity_search(query_embedding, top_k, filter_meta)

        try:
            with conn.cursor() as cur:
                # 1 - (embedding <=> query_vector) computes cosine similarity
                cur.execute("SET hnsw.ef_search = 40;")

                query = """
                SELECT chunk_id, doc_id, content, token_count,
                       1 - (embedding <=> %s) AS similarity,
                       metadata
                FROM document_chunks
                """
                params: list[Any] = [query_embedding]

                if filter_meta:
                    query += " WHERE metadata @> %s"
                    params.append(json.dumps(filter_meta))

                query += " ORDER BY embedding <=> %s ASC LIMIT %s;"
                params.extend([query_embedding, top_k])

                cur.execute(query, params)
                rows = cur.fetchall()

                return [
                    SearchResult(
                        chunk_id=r[0],
                        doc_id=r[1],
                        content=r[2],
                        token_count=r[3],
                        score=float(r[4]),
                        metadata=r[5] if isinstance(r[5], dict) else json.loads(r[5]),
                    )
                    for r in rows
                ]
        except Exception:
            self.in_memory = True
            return self.similarity_search(query_embedding, top_k, filter_meta)
        finally:
            conn.close()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Retrieve chunk record by its chunk_id."""
        if self.in_memory:
            return self._memory_chunks.get(chunk_id)

        conn = self._get_connection()
        if conn is None:
            return self._memory_chunks.get(chunk_id)

        try:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT chunk_id, doc_id, content, token_count, metadata
                FROM document_chunks WHERE chunk_id = %s;
                """, (chunk_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "content": row[2],
                    "token_count": row[3],
                    "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                }
        finally:
            conn.close()

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve multiple chunks by their chunk_ids."""
        if not chunk_ids:
            return []

        if self.in_memory:
            return [self._memory_chunks[cid] for cid in chunk_ids if cid in self._memory_chunks]

        conn = self._get_connection()
        if conn is None:
            return [self._memory_chunks[cid] for cid in chunk_ids if cid in self._memory_chunks]

        try:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT chunk_id, doc_id, content, token_count, metadata
                FROM document_chunks WHERE chunk_id = ANY(%s);
                """, (chunk_ids,))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "content": row[2],
                        "token_count": row[3],
                        "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                    })
                return results
        finally:
            conn.close()

    def record_entity_mentions(
        self,
        chunk_id: str,
        mentions: list[tuple[str, str]],
    ) -> None:
        """Record entity mentions linking entity canonical names to chunk_id."""
        if self.in_memory:
            for canon, etype in mentions:
                self._memory_entity_mentions.append({
                    "chunk_id": chunk_id,
                    "entity_canonical_name": canon,
                    "entity_type": etype,
                })
            return

        conn = self._get_connection()
        if conn is None:
            return

        try:
            with conn.cursor() as cur:
                for canon, etype in mentions:
                    cur.execute("""
                    INSERT INTO entity_mentions (chunk_id, entity_canonical_name, entity_type)
                    VALUES (%s, %s, %s);
                    """, (chunk_id, canon, etype))
                conn.commit()
        finally:
            conn.close()
