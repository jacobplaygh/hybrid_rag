from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RetrievalRoute:
    strategy: str
    alpha: float
    reason: str


class RetrievalRouter:
    """Choose the best retrieval strategy for a query."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def route(self, query: str, analysis: Optional[Dict[str, Any]] = None) -> RetrievalRoute:
        if not self.enabled:
            return RetrievalRoute(strategy="hybrid", alpha=0.7, reason="dynamic routing disabled")

        analysis = analysis or {}
        intent = str(analysis.get("intent", "FACTUAL")).upper()
        entities = analysis.get("entities") or []
        normalized = query.lower()

        if any(token in normalized for token in [" compare ", " compared to ", " versus ", " vs ", " difference between ", " tradeoff "]):
            return RetrievalRoute(
                strategy="hybrid",
                alpha=0.75,
                reason="comparative queries need semantic breadth and lexical precision",
            )

        if intent == "NAVIGATIONAL" or any(term in normalized for term in ["part number", "product id", "version", "error code", "section", "document", "guide", "file name"]):
            return RetrievalRoute(
                strategy="keyword",
                alpha=0.0,
                reason="identifier and navigation queries benefit from lexical matching",
            )

        if intent == "COMPLEX":
            return RetrievalRoute(
                strategy="hybrid",
                alpha=0.6,
                reason="complex queries need broader recall before synthesis",
            )

        if intent == "COMPARATIVE":
            return RetrievalRoute(
                strategy="hybrid",
                alpha=0.8,
                reason="comparative query requires balanced retrieval across related terms",
            )

        if entities and len(entities) > 0:
            return RetrievalRoute(
                strategy="vector",
                alpha=1.0,
                reason="entity-rich factual queries benefit from semantic matching",
            )

        if any(ch.isdigit() for ch in query):
            return RetrievalRoute(
                strategy="keyword",
                alpha=0.2,
                reason="numeric identifiers often match exact lexical text",
            )

        return RetrievalRoute(
            strategy="hybrid",
            alpha=0.7,
            reason="balanced retrieval is the default for general factual queries",
        )
