"""Unit tests for citation validator and sanitizer."""

from src.citation_validator import CitationValidator


def test_citation_validator_valid():
    validator = CitationValidator()
    answer = "Acme Corp is owned by ParentCo [source: doc1_ch00000] and supplied by GlobalParts [source: doc1_ch00001]."
    valid_ids = ["doc1_ch00000", "doc1_ch00001", "doc1_ch00002"]

    report = validator.validate(answer, valid_ids)
    assert report.is_valid is True
    assert report.total_citations_count == 2
    assert report.citation_precision == 1.0
    assert report.invalid_citations == []
    assert report.cleaned_answer == answer


def test_citation_validator_detects_and_strips_hallucinations():
    validator = CitationValidator()
    answer = "Acme Corp is owned by ParentCo [source: doc1_ch00000] and located in Mars [source: fake_chunk_999]."
    valid_ids = ["doc1_ch00000"]

    report = validator.validate(answer, valid_ids)
    assert report.is_valid is False
    assert report.invalid_citations == ["fake_chunk_999"]
    assert report.valid_citations == ["doc1_ch00000"]
    assert report.total_citations_count == 2
    assert report.citation_precision == 0.5
    # The hallucinated citation is stripped from the cleaned answer
    assert "fake_chunk_999" not in report.cleaned_answer
    assert "[source: doc1_ch00000]" in report.cleaned_answer
