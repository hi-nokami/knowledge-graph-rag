"""Embedding generation using nomic-embed-text via Ollama."""

import hashlib
import math
import sys
from pathlib import Path
from typing import Optional

# Access _shared/ollama_helper.py without duplicating code
_SHARED_DIR = Path(__file__).resolve().parent.parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.append(str(_SHARED_DIR))

try:
    import ollama_helper
except ImportError:
    ollama_helper = None

from src.config import settings


def _generate_deterministic_fallback_embedding(text: str, dim: int = 768) -> list[float]:
    """Generate deterministic normalized pseudo-embedding for testing or when Ollama is offline."""
    vec = []
    # Seed with SHA256 hashes of text slices
    for i in range(dim):
        h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
        vec.append(val)
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class OllamaEmbeddings:
    """Embeddings client connecting to Ollama OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        dimension: int | None = None,
        use_fallback_if_unavailable: bool = True,
        offline_mode: bool = False,
    ):
        """Initialize embedding client.

        Args:
            model_name: Name of embedding model (defaults to nomic-embed-text).
            base_url: Ollama base URL or OpenAI compatibility URL.
            dimension: Expected vector dimension (defaults to 768).
            use_fallback_if_unavailable: Whether to use fallback pseudo-embeddings when Ollama is unreachable.
            offline_mode: If True, operates offline using deterministic embeddings.
        """
        self.model_name = model_name or settings.embedding_model
        self.base_url = base_url or settings.openai_compat_url
        self.dimension = dimension or settings.embedding_dim
        self.use_fallback_if_unavailable = use_fallback_if_unavailable
        self.offline_mode = offline_mode
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
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

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text string."""
        if not text:
            return [0.0] * self.dimension

        client = self._get_client()
        if client is not None:
            try:
                resp = client.embeddings.create(
                    model=self.model_name,
                    input=text,
                )
                embedding = resp.data[0].embedding
                # Validate dimension or pad/truncate if necessary
                if len(embedding) == self.dimension:
                    return embedding
                elif len(embedding) > self.dimension:
                    return embedding[: self.dimension]
                else:
                    return embedding + [0.0] * (self.dimension - len(embedding))
            except Exception:
                if not self.use_fallback_if_unavailable:
                    raise

        if self.use_fallback_if_unavailable:
            return _generate_deterministic_fallback_embedding(text, self.dimension)
        raise RuntimeError(f"Could not connect to Ollama embedding service at {self.base_url}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text strings."""
        if not texts:
            return []

        client = self._get_client()
        if client is not None:
            try:
                resp = client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                # Ensure ordered output by index
                ordered = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                return [d.embedding for d in ordered]
            except Exception:
                if not self.use_fallback_if_unavailable:
                    raise

        return [self.embed_text(t) for t in texts]
