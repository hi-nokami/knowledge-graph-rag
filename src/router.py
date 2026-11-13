"""Query routing module classifying questions into VECTOR, GRAPH, or BOTH retrieval paths."""

import re
import sys
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

_SHARED_DIR = Path(__file__).resolve().parent.parent.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.append(str(_SHARED_DIR))

from src.config import settings


class RouteTarget(str, Enum):
    """Retrieval targets."""

    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    BOTH = "BOTH"


class RouteResult(BaseModel):
    """Classification outcome."""

    target: RouteTarget
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


ROUTER_PROMPT = """You are a high-speed semantic query router in a hybrid retrieval architecture.
Your job is to classify incoming questions into one of three retrieval modes:

1. VECTOR: Single-fact lookups, general definitions, policies, quotes, or questions answerable from an isolated text passage.
2. GRAPH: Multi-hop relational connections ("Who supplies both X and Y?"), ownership chains ("subsidiaries of X"), corporate hierarchies, executive roles, or structural network paths.
3. BOTH: Complex, open-ended, comparative questions, or queries requiring both broad factual context and multi-hop entity relationships.

Respond strictly in this format:
ROUTE: [VECTOR|GRAPH|BOTH]
REASON: [Brief 1-sentence reason]
"""


class QueryRouter:
    """Fast semantic classifier routing queries to optimal retrieval pipelines."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        use_heuristics_fallback: bool = True,
        offline_mode: bool = False,
    ):
        """Initialize QueryRouter."""
        self.model_name = model_name or settings.router_model
        self.base_url = base_url or settings.openai_compat_url
        self.use_heuristics_fallback = use_heuristics_fallback
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
                    timeout=1.5,
                )
            except Exception:
                self._client = None
        return self._client

    def route(self, query: str) -> RouteResult:
        """Classify user query into VECTOR, GRAPH, or BOTH."""
        client = self._get_client()
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": ROUTER_PROMPT},
                        {"role": "user", "content": f"Query: {query}"},
                    ],
                    temperature=0.0,
                    max_tokens=60,
                )
                text = resp.choices[0].message.content or ""
                match = re.search(r"ROUTE:\s*(VECTOR|GRAPH|BOTH)", text, re.IGNORECASE)
                if match:
                    route_val = match.group(1).upper()
                    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
                    reason = reason_match.group(1).strip() if reason_match else "Model classification"
                    return RouteResult(
                        target=RouteTarget(route_val),
                        confidence=0.9,
                        reasoning=reason,
                    )
            except Exception:
                if not self.use_heuristics_fallback:
                    raise

        if self.use_heuristics_fallback:
            return self._heuristic_route(query)

        return RouteResult(target=RouteTarget.BOTH, confidence=0.5, reasoning="Default fallback to BOTH")

    def _heuristic_route(self, query: str) -> RouteResult:
        """Deterministic rule-based routing fallback."""
        q = query.lower()

        # Multi-hop and graph keywords
        graph_patterns = [
            r"\bsuppl(?:ies|y|ier|iers)\b",
            r"\bsubsidiar(?:y|ies)\b",
            r"\bparent\s+company\b",
            r"\bwho\s+owns\b",
            r"\bwho\s+acquired\b",
            r"\bacquisition\b",
            r"\bconnected\s+to\b",
            r"\bconnection\s+between\b",
            r"\brelationship\s+between\b",
            r"\bboth\s+.+\s+and\b",
            r"\bshare(?:d)?\s+supplier\b",
            r"\bhow\s+is\s+.+\s+related\s+to\b",
            r"\bpath\s+between\b",
            r"\bboard\s+member\b",
            r"\bceo\s+of\b",
        ]

        # Vector-only keywords
        vector_patterns = [
            r"\bwhat\s+is\s+the\s+definition\b",
            r"\bdefine\b",
            r"\bpolicy\b",
            r"\bexplain\s+what\b",
            r"\bdescribe\s+the\s+mission\b",
            r"\bsummarize\b",
        ]

        has_graph = any(re.search(pat, q) for pat in graph_patterns)
        has_vector = any(re.search(pat, q) for pat in vector_patterns)

        if has_graph and not has_vector:
            return RouteResult(
                target=RouteTarget.GRAPH,
                confidence=0.85,
                reasoning="Query contains multi-hop relational cues.",
            )
        elif has_vector and not has_graph:
            return RouteResult(
                target=RouteTarget.VECTOR,
                confidence=0.85,
                reasoning="Query requests textual definition or summary.",
            )
        else:
            return RouteResult(
                target=RouteTarget.BOTH,
                confidence=0.75,
                reasoning="Query is open-ended or contains mixed relational and textual cues.",
            )
