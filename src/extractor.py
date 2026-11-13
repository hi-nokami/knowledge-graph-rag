"""Structured entity and relationship extraction using local Ollama LLMs."""

import json
import re
import sys
from pathlib import Path
from typing import Optional

_SHARED_DIR = Path(__file__).resolve().parent.parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.append(str(_SHARED_DIR))

from src.config import settings
from src.ontology import (
    EntityType,
    RelationType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)

EXTRACTION_SYSTEM_PROMPT = """You are an expert financial and corporate knowledge graph extractor.
Analyze the provided text passage and extract all key entities and relationships strictly conforming to the following ontology.

Allowed Entity Types:
- Company (name, sector, aliases)
- Person (name, role, aliases)
- Product (name, category)
- Location (name, location_type)
- FinancialMetric (name, metric_type, amount, currency, year)

Allowed Relationship Types:
- SUBSIDIARY_OF (source: Company, target: Company)
- SUPPLIES (source: Company, target: Company, properties: {product: str})
- ACQUIRED (source: Company, target: Company, properties: {year: int, amount: float})
- INVESTED_IN (source: Company|Person, target: Company, properties: {amount: float})
- EMPLOYS (source: Company, target: Person, properties: {title: str})
- BOARD_MEMBER_OF (source: Person, target: Company)
- PRODUCES (source: Company, target: Product)
- LOCATED_IN (source: Company|Person, target: Location)
- REPORTED (source: Company, target: FinancialMetric)

You MUST respond strictly with a valid JSON object matching this schema:
{
  "entities": [
    {"name": "string", "type": "Company|Person|Product|Location|FinancialMetric", "aliases": [], "sector": null, "role": null, "category": null, "location_type": null, "amount": null, "currency": "USD", "year": null}
  ],
  "relationships": [
    {"source_name": "string", "source_type": "Company|Person|Product|Location", "target_name": "string", "target_type": "Company|Person|Product|Location", "relation_type": "RELATION_NAME", "confidence": 0.95, "properties": {}}
  ]
}
Do not include any conversational preamble or Markdown fences other than the raw JSON."""


class EntityExtractor:
    """Extracts entities and relations from text chunks adhering to the domain ontology."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        use_heuristics_fallback: bool = True,
        offline_mode: bool = False,
    ):
        """Initialize extractor.

        Args:
            model_name: Ollama model name (defaults to llama3.1:8b).
            base_url: Ollama API URL.
            use_heuristics_fallback: If True, falls back to rule-based regex extraction when Ollama is unavailable.
            offline_mode: If True, skips network client and uses heuristics directly.
        """
        self.model_name = model_name or settings.extraction_model
        self.base_url = base_url or settings.openai_compat_url
        self.use_heuristics_fallback = use_heuristics_fallback
        self.offline_mode = offline_mode
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI-compatible client."""
        if self.offline_mode:
            return None
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="ollama",
                    timeout=1.5,
                )
            except Exception:
                self._client = None
        return self._client

    def extract_from_chunk(self, chunk_id: str, text: str) -> ExtractionResult:
        """Extract entities and relations from a text chunk.

        Args:
            chunk_id: Identifier of the source chunk.
            text: Text content.

        Returns:
            ExtractionResult model containing validated entities and relations.
        """
        client = self._get_client()
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract entities and relationships from this text:\n\n{text}"},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw_json = response.choices[0].message.content or "{}"
                return self._parse_json_result(chunk_id, raw_json)
            except Exception:
                if not self.use_heuristics_fallback:
                    raise

        if self.use_heuristics_fallback:
            return self._heuristic_extraction(chunk_id, text)

        return ExtractionResult(chunk_id=chunk_id, entities=[], relationships=[])

    def _parse_json_result(self, chunk_id: str, raw_json: str) -> ExtractionResult:
        """Parse, clean, and validate raw JSON output into an ExtractionResult."""
        # Strip potential markdown formatting if returned
        cleaned = raw_json.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            return self._heuristic_extraction(chunk_id, raw_json)

        valid_entities: list[ExtractedEntity] = []
        for ent in data.get("entities", []):
            try:
                etype_str = ent.get("type", "").strip()
                etype = EntityType(etype_str)
                name = ent.get("name", "").strip()
                if name:
                    valid_entities.append(
                        ExtractedEntity(
                            name=name,
                            type=etype,
                            aliases=ent.get("aliases", []),
                            sector=ent.get("sector"),
                            role=ent.get("role"),
                            category=ent.get("category"),
                            location_type=ent.get("location_type"),
                            amount=ent.get("amount"),
                            currency=ent.get("currency", "USD"),
                            year=ent.get("year"),
                        )
                    )
            except Exception:
                continue

        valid_relations: list[ExtractedRelation] = []
        for rel in data.get("relationships", []):
            try:
                rtype = RelationType(rel.get("relation_type", "").strip())
                stype = EntityType(rel.get("source_type", "").strip())
                ttype = EntityType(rel.get("target_type", "").strip())
                sname = rel.get("source_name", "").strip()
                tname = rel.get("target_name", "").strip()
                if sname and tname:
                    conf = float(rel.get("confidence", 0.9))
                    valid_relations.append(
                        ExtractedRelation(
                            source_name=sname,
                            source_type=stype,
                            target_name=tname,
                            target_type=ttype,
                            relation_type=rtype,
                            confidence=max(0.0, min(1.0, conf)),
                            properties=rel.get("properties", {}),
                        )
                    )
            except Exception:
                continue

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=valid_entities,
            relationships=valid_relations,
        )

    def _heuristic_extraction(self, chunk_id: str, text: str) -> ExtractionResult:
        """Deterministic heuristic extraction fallback using patterns for offline execution."""
        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelation] = []
        seen_entities: dict[str, ExtractedEntity] = {}

        def add_entity(raw_name: str, etype: EntityType = EntityType.COMPANY):
            clean = raw_name.strip()
            # Clean common trailing clauses
            for stop in [" to ", " for ", " in ", " and "]:
                if stop in clean.lower():
                    clean = clean[:clean.lower().find(stop)].strip()
            if len(clean) >= 2 and clean.lower() not in seen_entities:
                ent = ExtractedEntity(name=clean, type=etype)
                seen_entities[clean.lower()] = ent
                entities.append(ent)
            return clean

        # 1. Supplier patterns
        supp_matches = re.finditer(
            r"([A-Z][a-zA-Z0-9\s]+?)\s+(?:supplies|is a supplier to|entered into (?:an?|a) (?:[\w-]+\s+)*(?:supplier|supply) (?:agreement|contract) (?:with|to supply(?:\s+[a-zA-Z0-9\s]+?)?\s+to|with)|partnered with)\s+([A-Z][a-zA-Z0-9\s]+)",
            text,
            re.IGNORECASE,
        )
        for sm in supp_matches:
            src = add_entity(sm.group(1).strip())
            tgt_clause = sm.group(2).strip()
            # Split target if "both X and Y"
            targets = [tgt_clause]
            if " and " in tgt_clause:
                targets = [t.strip().replace("both ", "") for t in tgt_clause.split(" and ")]

            for tgt_raw in targets:
                tgt = add_entity(tgt_raw)
                relationships.append(
                    ExtractedRelation(
                        source_name=src,
                        source_type=EntityType.COMPANY,
                        target_name=tgt,
                        target_type=EntityType.COMPANY,
                        relation_type=RelationType.SUPPLIES,
                        confidence=0.95,
                    )
                )

        # 2. Subsidiary patterns: "X is a subsidiary of Y"
        sub_matches = re.finditer(
            r"([A-Z][a-zA-Z0-9\s]+?)\s+is a subsidiary of\s+([A-Z][a-zA-Z0-9\s]+)",
            text,
            re.IGNORECASE,
        )
        for sb in sub_matches:
            src = add_entity(sb.group(1).strip())
            tgt = add_entity(sb.group(2).strip())
            relationships.append(
                ExtractedRelation(
                    source_name=src,
                    source_type=EntityType.COMPANY,
                    target_name=tgt,
                    target_type=EntityType.COMPANY,
                    relation_type=RelationType.SUBSIDIARY_OF,
                    confidence=0.98,
                )
            )

        # 3. Investor patterns: "X invested in Y"
        inv_matches = re.finditer(
            r"([A-Z][a-zA-Z0-9\s]+?)\s+invested in\s+([A-Z][a-zA-Z0-9\s]+)",
            text,
            re.IGNORECASE,
        )
        for im in inv_matches:
            src = add_entity(im.group(1).strip())
            tgt = add_entity(im.group(2).strip())
            relationships.append(
                ExtractedRelation(
                    source_name=src,
                    source_type=EntityType.COMPANY,
                    target_name=tgt,
                    target_type=EntityType.COMPANY,
                    relation_type=RelationType.INVESTED_IN,
                    confidence=0.95,
                )
            )

        # 4. Product production patterns: "X produces [advanced] Y"
        prod_matches = re.finditer(
            r"([A-Z][a-zA-Z0-9\s]+?)\s+produces(?:\s+advanced)?\s+([A-Z][a-zA-Z0-9\s]+)",
            text,
            re.IGNORECASE,
        )
        for pm in prod_matches:
            src = add_entity(pm.group(1).strip())
            prod_name = add_entity(pm.group(2).strip(), etype=EntityType.PRODUCT)
            relationships.append(
                ExtractedRelation(
                    source_name=src,
                    source_type=EntityType.COMPANY,
                    target_name=prod_name,
                    target_type=EntityType.PRODUCT,
                    relation_type=RelationType.PRODUCES,
                    confidence=0.95,
                )
            )

        # 5. Acquisition patterns: "X acquired Y in [year]"
        acq_matches = re.finditer(
            r"([A-Z][a-zA-Z0-9\s]+?)\s+acquired\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s+in\s+(\d{4}))?",
            text,
            re.IGNORECASE,
        )
        for am in acq_matches:
            src = add_entity(am.group(1).strip())
            tgt = add_entity(am.group(2).strip())
            props = {}
            if am.group(3):
                props["year"] = int(am.group(3))
            relationships.append(
                ExtractedRelation(
                    source_name=src,
                    source_type=EntityType.COMPANY,
                    target_name=tgt,
                    target_type=EntityType.COMPANY,
                    relation_type=RelationType.ACQUIRED,
                    confidence=0.95,
                    properties=props,
                )
            )

        # Also extract standalone corporate entities using generalized corporate patterns
        corp_suffixes = (
            r"Corp|Corporation|Inc|Incorporated|Industries|Technologies|Technology|"
            r"Systems|Energy|Holdings|Solutions|Motors|Logistics|Capital|Dynamics|"
            r"Ventures|Partners|Auto|Power|Tech|Grid|Supply|Enterprises|Group|"
            r"International|Parts|Aero|Labs?|Co|LLC|Ltd|PLC|GmbH|SA|AG"
        )
        company_patterns = [
            rf"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*(?:\s+(?:{corp_suffixes}))+)\b",
            # Generalized PascalCase / CamelCase corporate brand names
            r"\b([A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*)\b",
        ]
        for pattern in company_patterns:
            for match in re.finditer(pattern, text):
                add_entity(match.group(1).strip())

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            relationships=relationships,
        )
