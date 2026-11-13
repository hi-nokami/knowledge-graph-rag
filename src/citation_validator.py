"""Post-generation citation validator and sanitization engine."""

import re
from pydantic import BaseModel, Field

_CITATION_PATTERN = re.compile(r"\[source:\s*([a-zA-Z0-9_\-]+)\]", re.IGNORECASE)


class CitationValidationResult(BaseModel):
    """Validation report for citations present in an answer."""

    is_valid: bool = Field(description="True if all citations are grounded in the retrieved context")
    valid_citations: list[str] = Field(default_factory=list, description="Citations that match retrieved chunk IDs")
    invalid_citations: list[str] = Field(default_factory=list, description="Fabricated or missing chunk IDs")
    total_citations_count: int = Field(default=0)
    citation_precision: float = Field(default=1.0, ge=0.0, le=1.0)
    cleaned_answer: str = Field(description="Answer with invalid citations stripped or marked")


class CitationValidator:
    """Validates that citations produced by the generator map to actual retrieved context."""

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """Extract all cited chunk IDs from text."""
        return _CITATION_PATTERN.findall(text)

    def validate(self, answer: str, valid_chunk_ids: list[str]) -> CitationValidationResult:
        """Validate answer citations against the set of ground-truth retrieved chunk IDs.

        Args:
            answer: Generated text containing [source: <chunk_id>] markers.
            valid_chunk_ids: List of chunk IDs present in the generation context.

        Returns:
            CitationValidationResult with validity flags and citation metrics.
        """
        valid_set = set(valid_chunk_ids)
        found_citations = self.extract_citations(answer)

        valid_cites: list[str] = []
        invalid_cites: list[str] = []

        for cite in found_citations:
            if cite in valid_set:
                valid_cites.append(cite)
            else:
                invalid_cites.append(cite)

        total = len(found_citations)
        precision = (len(valid_cites) / total) if total > 0 else 1.0
        is_valid = len(invalid_cites) == 0

        # Create cleaned answer by removing invalid citations
        def replace_cite(match):
            cid = match.group(1)
            if cid in valid_set:
                return match.group(0)
            return ""  # Strip hallucinated citation

        cleaned_answer = _CITATION_PATTERN.sub(replace_cite, answer)
        # Normalize double spaces created by removal
        cleaned_answer = re.sub(r" +", " ", cleaned_answer).strip()

        return CitationValidationResult(
            is_valid=is_valid,
            valid_citations=valid_cites,
            invalid_citations=invalid_cites,
            total_citations_count=total,
            citation_precision=precision,
            cleaned_answer=cleaned_answer,
        )
