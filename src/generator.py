"""Answer generation with structured context formatting and claim citations."""

import sys
from pathlib import Path
from typing import Any

_SHARED_DIR = Path(__file__).resolve().parent.parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.append(str(_SHARED_DIR))

from src.config import settings
from src.fusion import FusedChunk

SYSTEM_PROMPT = """You are a factual research assistant. Answer the user question concisely in 1-2 direct sentences strictly using the provided context.
For every factual claim you make, you MUST cite the source chunk using the format [source: <chunk_id>].
If the provided context does not contain sufficient information, state that the question cannot be answered."""


class AnswerGenerator:
    """Generates grounded answers citing retrieved graph facts and text chunks."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        use_fallback_if_unavailable: bool = True,
        offline_mode: bool = False,
    ):
        """Initialize AnswerGenerator."""
        self.model_name = model_name or settings.generator_model
        self.base_url = base_url or settings.openai_compat_url
        self.use_fallback_if_unavailable = use_fallback_if_unavailable
        self.offline_mode = offline_mode
        self._client = None

    def _get_client(self):
        if self.offline_mode:
            return None
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="ollama",
                    timeout=60.0,
                )
            except Exception:
                self._client = None
        return self._client

    def build_prompt_context(
        self,
        graph_facts: list[str],
        chunks: list[FusedChunk],
    ) -> str:
        """Format retrieval results into structured, labeled context sections."""
        parts = []

        if graph_facts:
            parts.append("=== Graph-Derived Relationships ===")
            for fact in graph_facts:
                parts.append(f"- {fact}")
            parts.append("")

        if chunks:
            parts.append("=== Retrieved Text Passages ===")
            for c in chunks:
                parts.append(f"--- Chunk {c.chunk_id} ---")
                parts.append(c.content.strip())
                parts.append("")

        return "\n".join(parts).strip()

    def generate_answer(
        self,
        question: str,
        graph_facts: list[str],
        chunks: list[FusedChunk],
    ) -> str:
        """Generate answer given question and retrieved context."""
        context_str = self.build_prompt_context(graph_facts, chunks)

        user_content = f"""{context_str}

=== User Question ===
{question}
"""

        client = self._get_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content.strip() and getattr(msg, "reasoning", None):
                    # In reasoning models (like qwen3:8b), if answer was truncated to reasoning buffer
                    r_lines = [l.strip() for l in msg.reasoning.strip().split("\n") if l.strip()]
                    if r_lines:
                        content = r_lines[-1]
                if content.strip():
                    return content.strip()
            except Exception:
                if not self.use_fallback_if_unavailable:
                    raise

        if self.use_fallback_if_unavailable:
            return self._heuristic_answer(question, graph_facts, chunks)

        raise RuntimeError("LLM generation failed and fallback is disabled.")

    def _heuristic_answer(
        self,
        question: str,
        graph_facts: list[str],
        chunks: list[FusedChunk],
    ) -> str:
        """Deterministic synthesized answer fallback for offline execution."""
        lines = []

        if graph_facts:
            for gf in graph_facts:
                lines.append(gf)

        if not lines and chunks:
            top_chunk = chunks[0]
            lines.append(f"According to the documentation, {top_chunk.content[:200]}... [source: {top_chunk.chunk_id}]")

        if not lines:
            return "The provided context does not contain sufficient information to answer the question."

        return " ".join(lines)
