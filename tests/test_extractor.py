"""Unit tests for entity and relationship extraction module."""

from src.extractor import EntityExtractor
from src.ontology import EntityType, RelationType


def test_heuristic_extractor_entities_and_relationships(mock_extractor):
    text = "GlobalParts supplies Acme Corp and Baker Industries. Acme Corp is a subsidiary of ParentCo."
    res = mock_extractor.extract_from_chunk("test_chunk_01", text)

    assert res.chunk_id == "test_chunk_01"
    ent_names = [e.name.lower() for e in res.entities]
    assert any("globalparts" in n for n in ent_names)
    assert any("acme" in n for n in ent_names)
    assert any("baker" in n for n in ent_names)

    rel_types = [r.relation_type for r in res.relationships]
    assert RelationType.SUPPLIES in rel_types
    assert RelationType.SUBSIDIARY_OF in rel_types


def test_extractor_json_parsing(mock_extractor):
    raw_json = """```json
    {
      "entities": [
        {"name": "AlphaCorp", "type": "Company", "sector": "Technology"}
      ],
      "relationships": [
        {"source_name": "AlphaCorp", "source_type": "Company", "target_name": "BetaGroup", "target_type": "Company", "relation_type": "SUBSIDIARY_OF", "confidence": 0.9}
      ]
    }
    ```"""
    parsed = mock_extractor._parse_json_result("chunk_99", raw_json)
    assert len(parsed.entities) == 1
    assert parsed.entities[0].name == "AlphaCorp"
    assert parsed.entities[0].type == EntityType.COMPANY
    assert len(parsed.relationships) == 1
    assert parsed.relationships[0].relation_type == RelationType.SUBSIDIARY_OF
