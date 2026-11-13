"""Configuration settings for Knowledge Graph RAG."""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Base project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    """System settings loaded from environment variables with sensible defaults."""

    # Neo4j Settings
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password123"))

    # PostgreSQL / pgvector Settings
    postgres_host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    postgres_port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))
    postgres_db: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "kgrag"))

    # Ollama / Local LLM Settings
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    openai_compat_url: str = Field(default_factory=lambda: os.getenv("OPENAI_COMPAT_URL", "http://localhost:11434/v1"))
    
    # Model Configurations per AGENTS.md
    router_model: str = Field(default_factory=lambda: os.getenv("ROUTER_MODEL", "qwen2.5:1.5b"))
    extraction_model: str = Field(default_factory=lambda: os.getenv("EXTRACTION_MODEL", "llama3.1:8b"))
    generator_model: str = Field(default_factory=lambda: os.getenv("GENERATOR_MODEL", "qwen3:8b"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    embedding_dim: int = Field(default=768)

    # Chunking & Retrieval Parameters
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    top_k_vector: int = Field(default=5)
    top_k_graph: int = Field(default=5)
    top_k_fused: int = Field(default=8)
    rrf_k: int = Field(default=60)

    # API Configuration
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    @property
    def postgres_dsn(self) -> str:
        """Construct PostgreSQL connection string."""
        return (
            f"dbname={self.postgres_db} "
            f"user={self.postgres_user} "
            f"password={self.postgres_password} "
            f"host={self.postgres_host} "
            f"port={self.postgres_port}"
        )


settings = Settings()
