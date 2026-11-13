"""Parameterized Cypher template registry and safe execution interface."""

from typing import Any
from pydantic import BaseModel, Field


class CypherTemplate(BaseModel):
    """Specification of a pre-approved parameterized Cypher query."""

    name: str
    description: str
    query: str
    required_params: list[str] = Field(default_factory=list)


# --- Template Definitions ---

TEMPLATE_SHARED_SUPPLIERS = CypherTemplate(
    name="shared_suppliers",
    description="Find common supplier companies that supply both company A and company B.",
    query="""
    MATCH (s:Company)-[r1:SUPPLIES]->(a:Company {canonical_name: $company_a}),
          (s:Company)-[r2:SUPPLIES]->(b:Company {canonical_name: $company_b})
    RETURN s.name AS supplier,
           s.canonical_name AS supplier_canonical,
           s.sector AS sector,
           r1.product AS product_a,
           r2.product AS product_b,
           r1.source_chunk_id AS source_a,
           r2.source_chunk_id AS source_b
    LIMIT 20;
    """,
    required_params=["company_a", "company_b"],
)

TEMPLATE_MULTI_HOP_PATHS = CypherTemplate(
    name="multi_hop_paths",
    description="Find 1 to 3 hop relational paths connecting two entities.",
    query="""
    MATCH p = (start {canonical_name: $start_name})-[*1..3]-(target {canonical_name: $target_name})
    RETURN [n in nodes(p) | coalesce(n.name, n.canonical_name)] AS node_names,
           [n in nodes(p) | labels(n)[0]] AS node_labels,
           [r in relationships(p) | type(r)] AS rel_types,
           [r in relationships(p) | r.source_chunk_id] AS source_chunks,
           length(p) AS hop_count
    ORDER BY length(p) ASC
    LIMIT 10;
    """,
    required_params=["start_name", "target_name"],
)

TEMPLATE_SUBSIDIARY_NETWORK = CypherTemplate(
    name="subsidiary_network",
    description="Traverse subsidiary relationships and attached financial metrics.",
    query="""
    MATCH (parent:Company {canonical_name: $parent_name})<-[r:SUBSIDIARY_OF*1..2]-(sub:Company)
    OPTIONAL MATCH (sub)-[rm:REPORTED]->(m:FinancialMetric)
    RETURN sub.name AS subsidiary,
           sub.canonical_name AS subsidiary_canonical,
           sub.sector AS sector,
           m.metric_type AS metric_type,
           m.amount AS amount,
           m.currency AS currency,
           m.year AS year,
           coalesce(r[0].source_chunk_id, rm.source_chunk_id) AS source_chunk_id
    LIMIT 25;
    """,
    required_params=["parent_name"],
)

TEMPLATE_NEIGHBORHOOD_EXPANSION = CypherTemplate(
    name="neighborhood_expansion",
    description="Retrieve 1-hop relationships and connected neighbors for an entity.",
    query="""
    MATCH (e {canonical_name: $entity_name})-[r]-(neighbor)
    WHERE type(r) <> 'MENTIONED_IN'
    RETURN type(r) AS relation_type,
           startNode(r) = e AS is_outgoing,
           labels(neighbor)[0] AS neighbor_label,
           coalesce(neighbor.name, neighbor.canonical_name) AS neighbor_name,
           neighbor.canonical_name AS neighbor_canonical,
           r.source_chunk_id AS source_chunk_id,
           properties(r) AS edge_props
    LIMIT 20;
    """,
    required_params=["entity_name"],
)

TEMPLATE_CHUNK_BACKING_RELATIONS = CypherTemplate(
    name="chunk_backing_relations",
    description="Retrieve all relationship triples extracted from a specific chunk.",
    query="""
    MATCH (s)-[r {source_chunk_id: $chunk_id}]->(t)
    RETURN labels(s)[0] AS source_type,
           coalesce(s.name, s.canonical_name) AS source_name,
           type(r) AS relation_type,
           labels(t)[0] AS target_type,
           coalesce(t.name, t.canonical_name) AS target_name,
           r.confidence AS confidence,
           properties(r) AS properties
    LIMIT 50;
    """,
    required_params=["chunk_id"],
)


class CypherTemplateRegistry:
    """Safe registry ensuring only verified, parameterized queries are executed."""

    _TEMPLATES: dict[str, CypherTemplate] = {
        "shared_suppliers": TEMPLATE_SHARED_SUPPLIERS,
        "multi_hop_paths": TEMPLATE_MULTI_HOP_PATHS,
        "subsidiary_network": TEMPLATE_SUBSIDIARY_NETWORK,
        "neighborhood_expansion": TEMPLATE_NEIGHBORHOOD_EXPANSION,
        "chunk_backing_relations": TEMPLATE_CHUNK_BACKING_RELATIONS,
    }

    @classmethod
    def get(cls, template_name: str) -> CypherTemplate:
        """Fetch template by name or raise KeyError."""
        if template_name not in cls._TEMPLATES:
            raise KeyError(
                f"Unknown Cypher template '{template_name}'. "
                f"Available templates: {list(cls._TEMPLATES.keys())}"
            )
        return cls._TEMPLATES[template_name]

    @classmethod
    def list_templates(cls) -> list[str]:
        """List registered template names."""
        return list(cls._TEMPLATES.keys())

    @classmethod
    def validate_params(cls, template_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Verify that all required parameters are provided."""
        template = cls.get(template_name)
        missing = [p for p in template.required_params if p not in params]
        if missing:
            raise ValueError(
                f"Template '{template_name}' requires parameters {missing}, but got {list(params.keys())}"
            )
        return params
