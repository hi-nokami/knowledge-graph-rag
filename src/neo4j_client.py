"""Neo4j graph database client with constraints, idempotent MERGE queries, and in-memory mock."""

from typing import Any, Optional
from pydantic import BaseModel

from src.config import settings
from src.ontology import (
    EntityType,
    RelationType,
    ExtractedEntity,
    ExtractedRelation,
    Relationship,
)
from src.cypher_templates import CypherTemplateRegistry
from src.entity_resolver import EntityResolver


class Neo4jClient:
    """Client for managing Neo4j connection, schema, ingestion, and templated querying."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        in_memory: bool = False,
    ):
        """Initialize Neo4j client.

        Args:
            uri: Bolt protocol URI (e.g. bolt://localhost:7687).
            user: Neo4j username.
            password: Neo4j password.
            in_memory: If True, operates in in-memory mode without Neo4j server.
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.in_memory = in_memory
        self.resolver = EntityResolver()

        # In-memory graph representation for offline testing
        self._memory_nodes: dict[tuple[str, str], dict[str, Any]] = {}  # (type, canonical_name) -> props
        self._memory_edges: list[dict[str, Any]] = []

        self._driver = None

    def _get_driver(self):
        """Lazy initialization of Neo4j driver."""
        if self.in_memory:
            return None
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    connection_timeout=2.0,
                )
            except Exception:
                self._driver = None
        return self._driver

    def close(self):
        """Close Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def init_schema(self) -> bool:
        """Create uniqueness constraints and indexes for ontology entities and chunks."""
        if self.in_memory:
            return False

        driver = self._get_driver()
        if driver is None:
            self.in_memory = True
            return False

        queries = [
            "CREATE CONSTRAINT unique_company_canonical IF NOT EXISTS FOR (c:Company) REQUIRE c.canonical_name IS UNIQUE;",
            "CREATE CONSTRAINT unique_person_canonical IF NOT EXISTS FOR (p:Person) REQUIRE p.canonical_name IS UNIQUE;",
            "CREATE CONSTRAINT unique_product_canonical IF NOT EXISTS FOR (pr:Product) REQUIRE pr.canonical_name IS UNIQUE;",
            "CREATE CONSTRAINT unique_location_canonical IF NOT EXISTS FOR (l:Location) REQUIRE l.canonical_name IS UNIQUE;",
            "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (dc:DocumentChunk) REQUIRE dc.chunk_id IS UNIQUE;",
            "CREATE INDEX idx_company_name IF NOT EXISTS FOR (c:Company) ON (c.name);",
            "CREATE INDEX idx_person_name IF NOT EXISTS FOR (p:Person) ON (p.name);",
        ]

        try:
            with driver.session() as session:
                for q in queries:
                    session.run(q)
            return True
        except Exception:
            self.in_memory = True
            return False

    def merge_entity(self, entity: ExtractedEntity) -> str:
        """Idempotently MERGE an entity node into the graph, returning its canonical_name."""
        canon = self.resolver.resolve(entity.name)

        if self.in_memory:
            key = (entity.type.value, canon)
            existing = self._memory_nodes.get(key, {})
            aliases = list(set(existing.get("aliases", []) + entity.aliases + [entity.name]))
            props = {
                "canonical_name": canon,
                "name": entity.name,
                "type": entity.type.value,
                "aliases": aliases,
                "sector": entity.sector or existing.get("sector"),
                "role": entity.role or existing.get("role"),
                "category": entity.category or existing.get("category"),
                "location_type": entity.location_type or existing.get("location_type"),
            }
            self._memory_nodes[key] = props
            return canon

        driver = self._get_driver()
        if driver is None:
            self.in_memory = True
            return self.merge_entity(entity)

        label = entity.type.value
        query = f"""
        MERGE (e:{label} {{canonical_name: $canonical_name}})
        ON CREATE SET
            e.name = $name,
            e.aliases = $aliases,
            e.sector = $sector,
            e.role = $role,
            e.category = $category,
            e.location_type = $location_type,
            e.created_at = datetime()
        ON MATCH SET
            e.aliases = [x IN e.aliases + $aliases WHERE x IS NOT NULL]
        RETURN e.canonical_name AS canon;
        """
        params = {
            "canonical_name": canon,
            "name": entity.name,
            "aliases": list(set(entity.aliases + [entity.name])),
            "sector": entity.sector,
            "role": entity.role,
            "category": entity.category,
            "location_type": entity.location_type,
        }

        try:
            with driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                return record["canon"] if record else canon
        except Exception:
            self.in_memory = True
            return self.merge_entity(entity)

    def merge_relationship(
        self,
        rel: ExtractedRelation,
        source_chunk_id: str,
    ) -> bool:
        """Idempotently MERGE a directed relationship between entities, storing source_chunk_id."""
        source_canon = self.resolver.resolve(rel.source_name)
        target_canon = self.resolver.resolve(rel.target_name)
        rel_type = rel.relation_type.value
        source_label = rel.source_type.value
        target_label = rel.target_type.value

        if self.in_memory:
            # Ensure endpoints exist
            self._memory_nodes.setdefault((source_label, source_canon), {
                "canonical_name": source_canon, "name": rel.source_name, "type": source_label, "aliases": []
            })
            self._memory_nodes.setdefault((target_label, target_canon), {
                "canonical_name": target_canon, "name": rel.target_name, "type": target_label, "aliases": []
            })
            # Deduplicate parallel identical edges in graph
            for existing in self._memory_edges:
                if (existing["source_canon"] == source_canon and
                    existing["target_canon"] == target_canon and
                    existing["rel_type"] == rel_type):
                    existing["confidence"] = max(existing["confidence"], rel.confidence)
                    return True

            edge = {
                "source_canon": source_canon,
                "source_type": source_label,
                "target_canon": target_canon,
                "target_type": target_label,
                "rel_type": rel_type,
                "source_chunk_id": source_chunk_id,
                "confidence": rel.confidence,
                "properties": rel.properties or {},
            }
            self._memory_edges.append(edge)
            return True

        driver = self._get_driver()
        if driver is None:
            self.in_memory = True
            return self.merge_relationship(rel, source_chunk_id)

        query = f"""
        MATCH (s:{source_label} {{canonical_name: $source_canon}})
        MATCH (t:{target_label} {{canonical_name: $target_canon}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET
            r.source_chunk_id = $source_chunk_id,
            r.confidence = $confidence,
            r += $props
        ON MATCH SET
            r.source_chunk_id = $source_chunk_id,
            r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
            r += $props
        RETURN count(r) AS cnt;
        """
        params = {
            "source_canon": source_canon,
            "target_canon": target_canon,
            "source_chunk_id": source_chunk_id,
            "confidence": rel.confidence,
            "props": rel.properties or {},
        }

        try:
            with driver.session() as session:
                session.run(query, params)
            return True
        except Exception:
            self.in_memory = True
            return self.merge_relationship(rel, source_chunk_id)

    def merge_document_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        mentioned_canonical_entities: list[tuple[EntityType, str]] | None = None,
    ) -> None:
        """Create DocumentChunk node and link mentioned entities via [:MENTIONED_IN]."""
        mentioned = mentioned_canonical_entities or []

        if self.in_memory:
            self._memory_nodes[("DocumentChunk", chunk_id)] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "type": "DocumentChunk",
            }
            for etype, canon in mentioned:
                self._memory_edges.append({
                    "source_canon": canon,
                    "source_type": etype.value,
                    "target_canon": chunk_id,
                    "target_type": "DocumentChunk",
                    "rel_type": "MENTIONED_IN",
                    "source_chunk_id": chunk_id,
                    "confidence": 1.0,
                    "properties": {},
                })
            return

        driver = self._get_driver()
        if driver is None:
            self.in_memory = True
            self.merge_document_chunk(chunk_id, doc_id, mentioned_canonical_entities)
            return

        query = """
        MERGE (dc:DocumentChunk {chunk_id: $chunk_id})
        ON CREATE SET dc.doc_id = $doc_id;
        """
        try:
            with driver.session() as session:
                session.run(query, {"chunk_id": chunk_id, "doc_id": doc_id})
                for etype, canon in mentioned:
                    link_q = f"""
                    MATCH (e:{etype.value} {{canonical_name: $canon}})
                    MATCH (dc:DocumentChunk {{chunk_id: $chunk_id}})
                    MERGE (e)-[r:MENTIONED_IN]->(dc)
                    ON CREATE SET r.source_chunk_id = $chunk_id;
                    """
                    session.run(link_q, {"canon": canon, "chunk_id": chunk_id})
        except Exception:
            self.in_memory = True
            self.merge_document_chunk(chunk_id, doc_id, mentioned_canonical_entities)

    def execute_template(self, template_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Safely execute a pre-approved Cypher template."""
        template = CypherTemplateRegistry.get(template_name)
        validated = CypherTemplateRegistry.validate_params(template_name, params)

        if self.in_memory:
            return self._execute_memory_template(template_name, validated)

        driver = self._get_driver()
        if driver is None:
            self.in_memory = True
            return self._execute_memory_template(template_name, validated)

        try:
            with driver.session() as session:
                result = session.run(template.query, validated)
                return [dict(record) for record in result]
        except Exception:
            self.in_memory = True
            return self._execute_memory_template(template_name, validated)

    def _execute_memory_template(self, template_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Mock execution of templates against in-memory graph."""
        results = []

        def match_canon(q_name: str) -> str:
            norm = self.resolver.normalize_name(q_name)
            # Exact match
            for (ntype, canon) in self._memory_nodes:
                if canon == norm:
                    return canon
            # Prefix / alias match
            for (ntype, canon) in self._memory_nodes:
                if canon.startswith(norm) or norm.startswith(canon):
                    return canon
            return norm

        if template_name == "shared_suppliers":
            co_a = match_canon(params["company_a"])
            co_b = match_canon(params["company_b"])
            # Suppliers to co_a
            supp_a = {e["source_canon"]: e for e in self._memory_edges if e["rel_type"] == "SUPPLIES" and e["target_canon"] == co_a}
            # Suppliers to co_b
            for e in self._memory_edges:
                if e["rel_type"] == "SUPPLIES" and e["target_canon"] == co_b:
                    supp = e["source_canon"]
                    if supp in supp_a:
                        ea = supp_a[supp]
                        node = self._memory_nodes.get(("Company", supp), {})
                        results.append({
                            "supplier": node.get("name", supp.title()),
                            "supplier_canonical": supp,
                            "sector": node.get("sector", "Unknown"),
                            "product_a": ea["properties"].get("product"),
                            "product_b": e["properties"].get("product"),
                            "source_a": ea.get("source_chunk_id", ""),
                            "source_b": e.get("source_chunk_id", ""),
                        })

        elif template_name == "neighborhood_expansion":
            target = match_canon(params["entity_name"])
            for e in self._memory_edges:
                if e.get("rel_type") == "MENTIONED_IN":
                    continue
                if e["source_canon"] == target:
                    neighbor_node = self._memory_nodes.get((e["target_type"], e["target_canon"]), {})
                    results.append({
                        "relation_type": e["rel_type"],
                        "is_outgoing": True,
                        "neighbor_label": e["target_type"],
                        "neighbor_name": neighbor_node.get("name", e["target_canon"]),
                        "neighbor_canonical": e["target_canon"],
                        "source_chunk_id": e.get("source_chunk_id", ""),
                        "edge_props": e.get("properties", {}),
                    })
                elif e["target_canon"] == target:
                    neighbor_node = self._memory_nodes.get((e["source_type"], e["source_canon"]), {})
                    results.append({
                        "relation_type": e["rel_type"],
                        "is_outgoing": False,
                        "neighbor_label": e["source_type"],
                        "neighbor_name": neighbor_node.get("name", e["source_canon"]),
                        "neighbor_canonical": e["source_canon"],
                        "source_chunk_id": e.get("source_chunk_id", ""),
                        "edge_props": e.get("properties", {}),
                    })

        elif template_name == "subsidiary_network":
            parent = match_canon(params["parent_name"])
            for e in self._memory_edges:
                if e["rel_type"] == "SUBSIDIARY_OF" and e["target_canon"] == parent:
                    sub_canon = e["source_canon"]
                    sub_node = self._memory_nodes.get(("Company", sub_canon), {})
                    results.append({
                        "subsidiary": sub_node.get("name", sub_canon.title()),
                        "subsidiary_canonical": sub_canon,
                        "sector": sub_node.get("sector"),
                        "metric_type": None,
                        "amount": None,
                        "currency": "USD",
                        "year": None,
                        "source_chunk_id": e.get("source_chunk_id", ""),
                    })

        elif template_name == "multi_hop_paths":
            start = match_canon(params["start_name"])
            target = match_canon(params["target_name"])
            # Breadth-first search up to 3 hops
            queue = [([start], [("Company", start)], [], [])]
            visited = {start}
            while queue and len(results) < 5:
                curr_path, curr_typed, curr_rels, curr_sources = queue.pop(0)
                curr_node = curr_path[-1]

                if curr_node == target and len(curr_rels) > 0:
                    results.append({
                        "node_names": [self._memory_nodes.get(tn, {}).get("name", tn[1]) for tn in curr_typed],
                        "node_labels": [tn[0] for tn in curr_typed],
                        "rel_types": curr_rels,
                        "source_chunks": curr_sources,
                        "hop_count": len(curr_rels),
                    })
                    continue

                if len(curr_rels) < 3:
                    for e in self._memory_edges:
                        nxt = None
                        nxt_typed = None
                        if e["source_canon"] == curr_node and e["target_canon"] not in curr_path:
                            nxt = e["target_canon"]
                            nxt_typed = (e["target_type"], nxt)
                        elif e["target_canon"] == curr_node and e["source_canon"] not in curr_path:
                            nxt = e["source_canon"]
                            nxt_typed = (e["source_type"], nxt)

                        if nxt:
                            queue.append((
                                curr_path + [nxt],
                                curr_typed + [nxt_typed],
                                curr_rels + [e["rel_type"]],
                                curr_sources + [e.get("source_chunk_id", "")],
                            ))

        elif template_name == "chunk_backing_relations":
            cid = params["chunk_id"]
            for e in self._memory_edges:
                if e.get("source_chunk_id") == cid:
                    s_node = self._memory_nodes.get((e["source_type"], e["source_canon"]), {})
                    t_node = self._memory_nodes.get((e["target_type"], e["target_canon"]), {})
                    results.append({
                        "source_type": e["source_type"],
                        "source_name": s_node.get("name", e["source_canon"]),
                        "relation_type": e["rel_type"],
                        "target_type": e["target_type"],
                        "target_name": t_node.get("name", e["target_canon"]),
                        "confidence": e.get("confidence", 1.0),
                        "properties": e.get("properties", {}),
                    })

        return results
