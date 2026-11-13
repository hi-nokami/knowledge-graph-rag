"""Unit tests for Neo4j client and templated graph queries."""

from src.ontology import ExtractedEntity, ExtractedRelation, EntityType, RelationType


def test_neo4j_client_merge_and_templates(memory_neo4j_client):
    client = memory_neo4j_client

    # 1. Merge entities
    client.merge_entity(ExtractedEntity(name="GlobalParts", type=EntityType.COMPANY, sector="Manufacturing"))
    client.merge_entity(ExtractedEntity(name="Acme Corp", type=EntityType.COMPANY, sector="Energy"))
    client.merge_entity(ExtractedEntity(name="Baker Industries", type=EntityType.COMPANY, sector="Aerospace"))

    # 2. Merge relationships
    client.merge_relationship(
        ExtractedRelation(
            source_name="GlobalParts",
            source_type=EntityType.COMPANY,
            target_name="Acme Corp",
            target_type=EntityType.COMPANY,
            relation_type=RelationType.SUPPLIES,
            properties={"product": "Turbines"},
        ),
        source_chunk_id="chunk_101",
    )
    client.merge_relationship(
        ExtractedRelation(
            source_name="GlobalParts",
            source_type=EntityType.COMPANY,
            target_name="Baker Industries",
            target_type=EntityType.COMPANY,
            relation_type=RelationType.SUPPLIES,
            properties={"product": "Valves"},
        ),
        source_chunk_id="chunk_102",
    )

    # 3. Test shared_suppliers template
    params = {"company_a": "acme", "company_b": "baker"}
    records = client.execute_template("shared_suppliers", params)
    assert len(records) == 1
    assert records[0]["supplier_canonical"] == "globalparts"
    assert records[0]["product_a"] == "Turbines"
    assert records[0]["product_b"] == "Valves"

    # 4. Test neighborhood_expansion template
    nbrs = client.execute_template("neighborhood_expansion", {"entity_name": "acme"})
    assert len(nbrs) >= 1
    assert nbrs[0]["neighbor_canonical"] == "globalparts"
