"""Knowledge graph retrieval, path verbalization, and backing chunk resolution."""

import re
from typing import Any
from pydantic import BaseModel, Field

from src.neo4j_client import Neo4jClient
from src.vector_store import PgVectorStore
from src.entity_resolver import EntityResolver


class GraphRetrievalResult(BaseModel):
    """Result of graph retrieval containing natural language facts and backing text chunks."""

    verbalized_facts: list[str] = Field(default_factory=list, description="Natural language statements with citations")
    source_chunk_ids: list[str] = Field(default_factory=list, description="Extracted source_chunk_ids from graph edges")
    backing_chunks: list[dict[str, Any]] = Field(default_factory=list, description="Full chunk contents fetched by ID")
    raw_graph_records: list[dict[str, Any]] = Field(default_factory=list, description="Raw Cypher records")


class GraphRetriever:
    """Coordinates entity resolution on queries, template execution, verbalization, and backing chunk fetching."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        vector_store: PgVectorStore,
        resolver: EntityResolver | None = None,
    ):
        """Initialize GraphRetriever."""
        self.graph_client = neo4j_client
        self.vector_store = vector_store
        self.resolver = resolver or neo4j_client.resolver

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract candidate entity names mentioned in query."""
        # Find capitalized words or phrases (e.g. Acme Corp, Baker Industries, GlobalParts)
        candidates = re.findall(
            r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b",
            query,
        )
        # Filter out common query question words
        stop_words = {"Who", "What", "Which", "Where", "When", "Why", "How", "Are", "Is", "Does", "Do", "Can"}
        cleaned = [c.strip() for c in candidates if c.strip() not in stop_words]
        return cleaned

    def retrieve(self, query: str, top_k_facts: int = 10) -> GraphRetrievalResult:
        """Execute graph retrieval for query, verbalize facts, and fetch backing text chunks."""
        entities = self._extract_query_entities(query)
        q_lower = query.lower()

        verbalized_facts: list[str] = []
        source_chunk_ids: set[str] = set()
        raw_records: list[dict[str, Any]] = []

        # 1. Template Selection Strategy
        # Shared suppliers: requires explicit "both" or multiple targets joined by "and"
        if len(entities) >= 2 and ("both" in q_lower or (" and " in q_lower and ("supplier" in q_lower or "supplies" in q_lower))):
            # Template: Shared suppliers
            params = {
                "company_a": self.resolver.resolve(entities[0]),
                "company_b": self.resolver.resolve(entities[1]),
            }
            records = self.graph_client.execute_template("shared_suppliers", params)
            raw_records.extend(records)
            for r in records:
                supp = r.get("supplier", "Unknown")
                p_a = r.get("product_a") or "products"
                p_b = r.get("product_b") or "products"
                src_a = r.get("source_a", "")
                src_b = r.get("source_b", "")
                
                src_tag = ""
                if src_a and src_b and src_a == src_b:
                    src_tag = f" [source: {src_a}]"
                    source_chunk_ids.add(src_a)
                else:
                    tags = []
                    if src_a:
                        tags.append(src_a)
                        source_chunk_ids.add(src_a)
                    if src_b:
                        tags.append(src_b)
                        source_chunk_ids.add(src_b)
                    if tags:
                        src_tag = f" [source: {', '.join(tags)}]"

                statement = f"{supp} supplies {entities[0]} ({p_a}) and {entities[1]} ({p_b}){src_tag}."
                verbalized_facts.append(statement)

        elif len(entities) >= 2 and ("path" in q_lower or "connection" in q_lower or "relationship" in q_lower or "related" in q_lower):
            # Template: Multi-hop paths
            params = {
                "start_name": self.resolver.resolve(entities[0]),
                "target_name": self.resolver.resolve(entities[1]),
            }
            records = self.graph_client.execute_template("multi_hop_paths", params)
            raw_records.extend(records)
            for r in records:
                nodes = r.get("node_names", [])
                rels = r.get("rel_types", [])
                sources = [s for s in r.get("source_chunks", []) if s]
                source_chunk_ids.update(sources)

                src_tag = f" [source: {', '.join(sources)}]" if sources else ""
                steps = []
                for i in range(len(rels)):
                    steps.append(f"{nodes[i]} -[{rels[i]}]-> {nodes[i+1]}")
                path_str = " -> ".join(steps)
                verbalized_facts.append(f"Connection path: {path_str}{src_tag}.")

        # 2. Subsidiary / hierarchy check
        if entities and ("subsidiary" in q_lower or "parent" in q_lower or "owns" in q_lower or "acquired" in q_lower):
            for ent in entities:
                params = {"parent_name": self.resolver.resolve(ent)}
                records = self.graph_client.execute_template("subsidiary_network", params)
                raw_records.extend(records)
                for r in records:
                    sub = r.get("subsidiary", "Unknown")
                    src = r.get("source_chunk_id", "")
                    if src:
                        source_chunk_ids.add(src)
                    src_tag = f" [source: {src}]" if src else ""
                    verbalized_facts.append(f"{sub} is a subsidiary of {ent}{src_tag}.")

        # 3. Neighborhood expansion on all query entities
        if entities:
            for ent in entities:
                params = {"entity_name": self.resolver.resolve(ent)}
                records = self.graph_client.execute_template("neighborhood_expansion", params)
                raw_records.extend(records)
                for r in records:
                    rel = r.get("relation_type", "RELATED_TO")
                    neighbor = r.get("neighbor_name", "Unknown")
                    is_out = r.get("is_outgoing", True)
                    src = r.get("source_chunk_id", "")
                    if src:
                        source_chunk_ids.add(src)
                    src_tag = f" [source: {src}]" if src else ""

                    if is_out:
                        stmt = f"{ent} {rel.replace('_', ' ').lower()} {neighbor}{src_tag}."
                    else:
                        stmt = f"{neighbor} {rel.replace('_', ' ').lower()} {ent}{src_tag}."
                    if stmt not in verbalized_facts:
                        verbalized_facts.append(stmt)

        # 4. Multi-hop 2nd hop expansion for relational queries
        if any(w in q_lower for w in ["investor", "invested", "parent", "owns", "path", "connection", "subsidiary", "supplies", "supplier"]):
            first_hop_canons = list(dict.fromkeys(r.get("neighbor_canonical") for r in raw_records if r.get("neighbor_canonical")))
            for neighbor_canon in first_hop_canons[:8]:
                nbr_records = self.graph_client.execute_template("neighborhood_expansion", {"entity_name": neighbor_canon})
                for nr in nbr_records:
                    n_rel = nr.get("relation_type", "RELATED_TO")
                    n_neighbor = nr.get("neighbor_name", "Unknown")
                    n_is_out = nr.get("is_outgoing", True)
                    n_src = nr.get("source_chunk_id", "")
                    if n_src:
                        source_chunk_ids.add(n_src)
                    src_tag = f" [source: {n_src}]" if n_src else ""
                    if n_is_out:
                        stmt = f"{neighbor_canon.title()} {n_rel.replace('_', ' ').lower()} {n_neighbor}{src_tag}."
                    else:
                        stmt = f"{n_neighbor} {n_rel.replace('_', ' ').lower()} {neighbor_canon.title()}{src_tag}."
                    if stmt not in verbalized_facts:
                        verbalized_facts.append(stmt)

        # Sort verbalized facts by query relevance (matching terms) to prioritize direct answers
        q_tokens = set(re.findall(r"\w+", q_lower))
        verbalized_facts.sort(
            key=lambda f: len(set(re.findall(r"\w+", f.lower())) & q_tokens),
            reverse=True,
        )

        # 5. Fetch backing text chunks from PostgreSQL via vector_store
        chunk_ids_list = list(source_chunk_ids)
        backing_chunks = self.vector_store.get_chunks_by_ids(chunk_ids_list)

        return GraphRetrievalResult(
            verbalized_facts=verbalized_facts[:top_k_facts],
            source_chunk_ids=chunk_ids_list,
            backing_chunks=backing_chunks,
            raw_graph_records=raw_records,
        )
