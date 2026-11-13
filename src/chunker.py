"""Recursive character text chunker with deterministic chunk IDs."""

from typing import Any
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents a text chunk produced from document splitting."""

    chunk_id: str = Field(description="Deterministic unique ID: {doc_id}_ch{index:05d}")
    doc_id: str = Field(description="Identifier of the source document")
    content: str = Field(description="Verbatim text content of the chunk")
    token_count: int = Field(description="Estimated or calculated token count")
    chunk_index: int = Field(description="Zero-based index of this chunk in the document")
    start_char: int = Field(default=0, description="Start character offset in document")
    end_char: int = Field(default=0, description="End character offset in document")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata dict")


class RecursiveCharacterChunker:
    """Recursively splits text into chunks using hierarchical separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum character length per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            separators: Hierarchical list of separators (defaults to paragraphs, sentences, words, chars).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split text recursively using the given separators list."""
        final_chunks: list[str] = []

        # Find the first separator present in the text
        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: list[str] = []
        for s in splits:
            if separator and good_splits:
                # Re-attach separator to avoid losing structure
                piece = separator + s if separator != "\n\n" and separator != "\n" else s
            else:
                piece = s

            if not piece.strip() and separator in ["\n\n", "\n"]:
                continue

            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if new_separators:
                    sub_splits = self._split_text(piece, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(piece)

        # Merge splits up to chunk_size with chunk_overlap
        current_chunk: list[str] = []
        current_length = 0

        for s in good_splits:
            s_len = len(s)
            if current_length + s_len > self.chunk_size and current_chunk:
                merged_text = "".join(current_chunk).strip()
                if merged_text:
                    final_chunks.append(merged_text)

                # Keep overlap from the end of current_chunk
                overlap_chunks: list[str] = []
                overlap_length = 0
                for prev in reversed(current_chunk):
                    if overlap_length + len(prev) <= self.chunk_overlap:
                        overlap_chunks.insert(0, prev)
                        overlap_length += len(prev)
                    else:
                        break
                current_chunk = overlap_chunks
                current_length = overlap_length

            current_chunk.append(s)
            current_length += s_len

        if current_chunk:
            merged_text = "".join(current_chunk).strip()
            if merged_text:
                final_chunks.append(merged_text)

        return final_chunks

    def split_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split document text into a list of TextChunk objects.

        Args:
            doc_id: Identifier for the source document.
            text: Raw document text.
            metadata: Base metadata dictionary to attach to each chunk.

        Returns:
            List of TextChunk models with deterministic chunk_id formatted as {doc_id}_ch{idx:05d}.
        """
        raw_chunks = self._split_text(text, self.separators)
        results: list[TextChunk] = []

        current_search_idx = 0
        base_meta = metadata.copy() if metadata else {}

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_ch{idx:05d}"
            
            # Find approximate char offset in original text
            start_char = text.find(chunk_text[:min(50, len(chunk_text))], current_search_idx)
            if start_char == -1:
                start_char = current_search_idx
            end_char = start_char + len(chunk_text)
            current_search_idx = max(start_char, current_search_idx)

            # Heuristic token count: ~4 chars per token
            token_count = max(1, len(chunk_text) // 4)

            chunk_meta = {
                **base_meta,
                "doc_id": doc_id,
                "chunk_index": idx,
                "start_char": start_char,
                "end_char": end_char,
            }

            results.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    token_count=token_count,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_meta,
                )
            )

        return results
