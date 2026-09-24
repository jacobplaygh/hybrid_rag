import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple


@dataclass
class GraphNode:
    name: str
    document_ids: Set[str] = field(default_factory=set)
    neighbors: Set[str] = field(default_factory=set)


class KnowledgeGraph:
    """Lightweight GraphRAG implementation for entity-centric queries."""

    KNOWN_TERMS = (
        "FastAPI",
        "Hybrid RAG",
        "GraphRAG",
        "BM25",
        "LangSmith",
        "Prometheus",
        "LlamaIndex",
        "vector store",
        "semantic cache",
        "Reranker",
        "Context Manager",
        "Query Understanding",
    )

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    def index_documents(self, documents: Iterable[Any]) -> None:
        for document in documents:
            if isinstance(document, dict):
                doc_id = str(document.get("id") or document.get("doc_id") or len(self.nodes))
                text = str(document.get("content") or document.get("text") or "")
            else:
                doc_id = str(getattr(document, "id", getattr(document, "doc_id", len(self.nodes))))
                text = str(getattr(document, "content", getattr(document, "text", "")))

            entities = self._extract_entities(text)
            if not entities:
                continue

            for entity in entities:
                node = self.nodes.setdefault(entity, GraphNode(name=entity))
                node.document_ids.add(doc_id)

            for source, target, relation in self._extract_relations(text):
                if source in self.nodes and target in self.nodes:
                    self.nodes[source].neighbors.add(target)
                    self.nodes[target].neighbors.add(source)
                    self.edges[source].add((target, relation))
                    self.edges[target].add((source, relation))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        entities = self._extract_entities(query)
        if not entities:
            return []

        results: List[Dict[str, Any]] = []
        visited: Set[str] = set()
        queue: List[str] = []

        for entity in entities:
            if entity in self.nodes and entity not in visited:
                visited.add(entity)
                queue.append(entity)

        while queue and len(results) < top_k:
            entity = queue.pop(0)
            neighbors = sorted(self.nodes.get(entity, GraphNode(entity)).neighbors)
            context_parts = [f"{entity}"]
            for neighbor in neighbors[:3]:
                rels = sorted({rel for other, rel in self.edges.get(entity, set()) if other == neighbor})
                if rels:
                    context_parts.append(f"related to {neighbor} via {', '.join(rels)}")
                else:
                    context_parts.append(f"related to {neighbor}")
            results.append({
                "entity": entity,
                "content": "; ".join(context_parts),
                "source": entity,
                "score": 0.8,
                "metadata": {
                    "graph_rag": True,
                    "neighbors": neighbors,
                    "document_ids": sorted(self.nodes[entity].document_ids),
                },
            })

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return results[:top_k]

    def _extract_entities(self, text: str) -> List[str]:
        tokens = []
        title_case_pattern = r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b"
        for match in re.finditer(title_case_pattern, text):
            value = match.group(0).strip()
            if value and value.lower() not in {"the", "a", "an"}:
                tokens.append(value)

        for term in self.KNOWN_TERMS:
            if term.lower() in text.lower():
                tokens.append(term)

        normalized = []
        seen = set()
        for value in tokens:
            key = value.lower()
            if key not in seen:
                normalized.append(value)
                seen.add(key)
        return normalized

    def _extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        relations: List[Tuple[str, str, str]] = []
        known_entities = sorted(self.KNOWN_TERMS, key=len, reverse=True)
        known_pattern = "|".join(re.escape(entity) for entity in known_entities)
        known_relation_pattern = (
            rf"\b({known_pattern})\b\s+"
            rf"(?:uses|integrates with|connects to|works with|depends on|builds on|extends)\s+"
            rf"\b({known_pattern})\b"
        )
        patterns = [
            (known_relation_pattern, "related"),
            (r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+(?:uses|integrates with|connects to|works with|depends on|builds on|extends)\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\b", "related"),
            (r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+is\s+powered by\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\b", "powered_by"),
            (r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\s+and\s+([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*)\b", "related"),
        ]

        for pattern, relation in patterns:
            flags = re.IGNORECASE if pattern == known_relation_pattern else 0
            for match in re.finditer(pattern, text, flags=flags):
                left, right = match.groups()
                if left and right:
                    relations.append((
                        self._canonical_entity(left.strip()),
                        self._canonical_entity(right.strip()),
                        relation,
                    ))
        return relations

    def _canonical_entity(self, value: str) -> str:
        """Normalize known entities while preserving their display spelling."""
        for entity in self.KNOWN_TERMS:
            if entity.lower() == value.lower():
                return entity
        return value
