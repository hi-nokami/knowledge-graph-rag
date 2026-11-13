"""Entity resolution and canonical name normalization."""

import re
from typing import Optional

# Common legal entity and professional suffixes
LEGAL_SUFFIXES = [
    "inc",
    "corp",
    "corporation",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "co",
    "company",
    "group",
    "holdings",
    "plc",
    "ag",
    "gmbh",
    "sa",
    "se",
    "nv",
    "phd",
    "ph.d",
    "md",
    "m.d",
    "mba",
    "cfa",
    "cpa",
]

_SUFFIX_REGEX = re.compile(
    r"(?:[\s,]+|\b)(?:" + "|".join(LEGAL_SUFFIXES) + r")\.?\s*$",
    re.IGNORECASE,
)


class EntityResolver:
    """Normalizes entity names, maintains alias mappings, and clusters canonical entities."""

    def __init__(self):
        """Initialize entity resolver with canonical mappings."""
        # Maps alias / normalized name -> canonical name
        self._canonical_map: dict[str, str] = {}
        # Maps canonical name -> set of aliases
        self._alias_groups: dict[str, set[str]] = {}

    @staticmethod
    def normalize_name(raw_name: str) -> str:
        """Strip punctuation, excessive whitespace, and legal suffixes to produce a canonical key.

        Examples:
            'Acme Corp, Inc.' -> 'acme'
            '  Apple Computer, Inc. ' -> 'apple computer'
            'Dr. Jane Doe, Ph.D.' -> 'jane doe'
        """
        if not raw_name:
            return ""

        text = raw_name.strip()

        # Remove honorific titles
        text = re.sub(r"^(?:mr\.|ms\.|mrs\.|dr\.|prof\.)\s+", "", text, flags=re.IGNORECASE)

        # Remove trailing legal suffixes iteratively
        prev = None
        while prev != text:
            prev = text
            text = _SUFFIX_REGEX.sub("", text).strip()

        # Clean punctuation and normalize whitespace
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text.lower()

    def resolve(self, raw_name: str) -> str:
        """Resolve a raw observed name to its canonical entity name.

        If already observed, returns the existing canonical name.
        Otherwise, registers the normalized name as a new canonical entity.
        """
        if not raw_name:
            return ""

        normalized = self.normalize_name(raw_name)
        if not normalized:
            return raw_name.lower().strip()

        if normalized in self._canonical_map:
            canon = self._canonical_map[normalized]
            self._canonical_map[raw_name.lower().strip()] = canon
            if canon not in self._alias_groups:
                self._alias_groups[canon] = {canon}
            self._alias_groups[canon].add(raw_name.strip())
            return canon

        # Register as a new canonical entity
        self._canonical_map[normalized] = normalized
        self._canonical_map[raw_name.lower().strip()] = normalized
        if normalized not in self._alias_groups:
            self._alias_groups[normalized] = {raw_name.strip(), normalized}
        else:
            self._alias_groups[normalized].add(raw_name.strip())

        return normalized

    def register_alias(self, canonical_name: str, alias: str) -> None:
        """Explicitly register an alias for an existing canonical name."""
        canon = self.normalize_name(canonical_name)
        norm_alias = self.normalize_name(alias)

        self._canonical_map[norm_alias] = canon
        self._canonical_map[alias.lower().strip()] = canon

        if canon not in self._alias_groups:
            self._alias_groups[canon] = {canonical_name.strip(), canon}
        self._alias_groups[canon].add(alias.strip())

    def get_aliases(self, canonical_name: str) -> list[str]:
        """Return all known aliases for a given canonical entity."""
        canon = self.normalize_name(canonical_name)
        return sorted(list(self._alias_groups.get(canon, set())))
