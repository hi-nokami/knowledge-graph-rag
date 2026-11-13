"""Ontology definitions for Knowledge Graph nodes and edges."""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Supported entity node types in corporate & financial ontology."""

    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    LOCATION = "Location"
    FINANCIAL_METRIC = "FinancialMetric"
    DOCUMENT_CHUNK = "DocumentChunk"


class RelationType(str, Enum):
    """Supported directed relationship types in ontology."""

    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    SUPPLIES = "SUPPLIES"
    ACQUIRED = "ACQUIRED"
    INVESTED_IN = "INVESTED_IN"
    EMPLOYS = "EMPLOYS"
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"
    PRODUCES = "PRODUCES"
    LOCATED_IN = "LOCATED_IN"
    REPORTED = "REPORTED"
    MENTIONED_IN = "MENTIONED_IN"


# --- Node Models ---

class BaseEntity(BaseModel):
    """Base model for all knowledge graph entities."""

    id: Optional[str] = None
    name: str = Field(description="Observed entity name in text")
    canonical_name: str = Field(description="Normalized canonical identifier")
    aliases: list[str] = Field(default_factory=list, description="Known alternative names")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompanyNode(BaseEntity):
    """Company entity node."""

    sector: Optional[str] = Field(default=None, description="Industry sector (e.g. Energy, Tech)")


class PersonNode(BaseEntity):
    """Person entity node."""

    role: Optional[str] = Field(default=None, description="Primary known role or title")


class ProductNode(BaseEntity):
    """Product entity node."""

    category: Optional[str] = Field(default=None, description="Product category")


class LocationNode(BaseEntity):
    """Geographic location node."""

    location_type: Optional[str] = Field(default=None, description="City, Country, HQ, etc.")


class FinancialMetricNode(BaseModel):
    """Financial metric node."""

    id: Optional[str] = None
    canonical_name: str
    metric_type: str = Field(description="Revenue, NetIncome, EBITDA, FundingRound, etc.")
    amount: float
    currency: str = Field(default="USD")
    period: Optional[str] = Field(default=None, description="Q1, Q2, Q3, Q4, Annual")
    year: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkNode(BaseModel):
    """Document chunk reference node in graph."""

    chunk_id: str
    doc_id: str
    section: Optional[str] = None
    content_hash: Optional[str] = None


# --- Relationship Model ---

class Relationship(BaseModel):
    """Directed edge between two entities in the knowledge graph."""

    source_canonical_name: str
    source_type: EntityType
    target_canonical_name: str
    target_type: EntityType
    relation_type: RelationType
    source_chunk_id: str = Field(description="ID of the chunk from which this fact was extracted")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    properties: dict[str, Any] = Field(default_factory=dict, description="Edge properties (year, amount, etc.)")


# --- Extraction Schemas ---

class ExtractedEntity(BaseModel):
    """Raw entity extracted by LLM."""

    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    sector: Optional[str] = None
    role: Optional[str] = None
    category: Optional[str] = None
    location_type: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = "USD"
    year: Optional[int] = None


class ExtractedRelation(BaseModel):
    """Raw relationship extracted by LLM."""

    source_name: str
    source_type: EntityType
    target_name: str
    target_type: EntityType
    relation_type: RelationType
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Structured extraction output for a single chunk."""

    chunk_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelation] = Field(default_factory=list)
