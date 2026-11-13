"""Unit tests for ontology models and entity resolver."""

import pytest
from src.ontology import (
    EntityType,
    RelationType,
    CompanyNode,
    PersonNode,
    Relationship,
    ExtractedEntity,
    ExtractedRelation,
)
from src.entity_resolver import EntityResolver


def test_entity_resolver_normalization():
    resolver = EntityResolver()

    assert resolver.normalize_name("Acme Corp, Inc.") == "acme"
    assert resolver.normalize_name("Apple Computer, Inc.") == "apple computer"
    assert resolver.normalize_name("Dr. Jane Doe, Ph.D.") == "jane doe"
    assert resolver.normalize_name("GlobalParts LLC") == "globalparts"
    assert resolver.normalize_name("Apex Solutions Ltd.") == "apex solutions"


def test_entity_resolver_alias_clustering():
    resolver = EntityResolver()

    canon1 = resolver.resolve("Acme Corporation")
    canon2 = resolver.resolve("Acme Corp.")
    canon3 = resolver.resolve("Acme Inc.")

    assert canon1 == "acme"
    assert canon2 == "acme"
    assert canon3 == "acme"

    aliases = resolver.get_aliases("acme")
    assert "Acme Corporation" in aliases
    assert "Acme Corp." in aliases


def test_ontology_models():
    company = CompanyNode(
        name="Acme Corp",
        canonical_name="acme",
        sector="Energy",
        aliases=["ACME"],
    )
    assert company.sector == "Energy"
    assert company.canonical_name == "acme"

    rel = Relationship(
        source_canonical_name="acme",
        source_type=EntityType.COMPANY,
        target_canonical_name="parentco",
        target_type=EntityType.COMPANY,
        relation_type=RelationType.SUBSIDIARY_OF,
        source_chunk_id="chunk_001",
        confidence=0.98,
        properties={"since": 2021},
    )
    assert rel.relation_type == RelationType.SUBSIDIARY_OF
    assert rel.properties["since"] == 2021
