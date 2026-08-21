import logging
import tiktoken
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Handles token budgeting, truncation, and context window management 
    for LLM prompts to prevent context overflow.
    """
    def __init__(
        self,
        max_tokens: int = 120000,
        system_prompt_reserve: int = 1000,
        encoding_name: str = "cl100k_base",
        max_docs_per_source: Optional[int] = 2,
    ):
        self.max_tokens = max_tokens
        self.system_prompt_reserve = system_prompt_reserve
        self.max_docs_per_source = max_docs_per_source
        self.last_selection_report: Dict[str, Any] = {}
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer {encoding_name}: {e}. Falling back to cl100k_base.")
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(self, text: str) -> int:
        """
        Calculate precise token count using tiktoken.
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def truncate_context(self, context_docs: List[Any], query: Optional[str], history: Optional[str] = None) -> List[Any]:
        """
        Truncates the list of retrieved documents to fit within the token budget,
        prioritizing the most relevant (top) documents.
        """
        # Calculate current overhead (query + history + reserve)
        overhead = self.estimate_tokens(query or "")
        if history:
            overhead += self.estimate_tokens(history)
        # Keep small explicit budgets usable instead of reserving the entire window.
        reserve = min(self.system_prompt_reserve, self.max_tokens // 4)
        overhead += reserve

        available_tokens = self.max_tokens - overhead
        self.last_selection_report = {
            "selected": [],
            "dropped": [],
            "available_tokens": max(available_tokens, 0),
            "used_tokens": 0,
            "ordering": "retrieval_order",
        }
        if available_tokens <= 0:
            logger.warning("Context window exhausted by query and history alone.")
            self.last_selection_report["reason"] = "context_overhead"
            for doc in context_docs:
                self._record_dropped(doc, "context_overhead")
            return []

        ordered_docs = self._order_documents(context_docs)
        if ordered_docs is not context_docs:
            self.last_selection_report["ordering"] = "score_descending"

        selected_docs = []
        current_tokens = 0
        seen_keys = set()
        source_counts = {}

        for doc in ordered_docs:
            content = doc.content if hasattr(doc, "content") else doc.get("content", "")
            if content is None:
                content = ""
            doc_key = self._document_key(doc, content)
            doc_id = self._document_id(doc, content)
            if doc_key in seen_keys:
                self._record_dropped(doc, "duplicate")
                continue
            seen_keys.add(doc_key)
            source = self._document_source(doc)
            if (
                source is not None
                and self.max_docs_per_source is not None
                and source_counts.get(source, 0) >= self.max_docs_per_source
            ):
                self._record_dropped(doc, "source_limit")
                continue
            doc_tokens = self.estimate_tokens(content)
            
            if current_tokens + doc_tokens <= available_tokens:
                selected_docs.append(doc)
                current_tokens += doc_tokens
                self.last_selection_report["selected"].append({
                    "id": doc_id,
                    "source": source,
                    "truncated": False,
                })
                if source is not None:
                    source_counts[source] = source_counts.get(source, 0) + 1
            else:
                # Try to fit a truncated version of the last document
                remaining = available_tokens - current_tokens
                if remaining > 100: # Only bother if we can fit a meaningful snippet
                    truncated_content = self._hard_truncate(content, remaining)
                    # Create a copy of the doc with truncated content
                    if hasattr(doc, "content"):
                        # This assumes doc is a LlamaIndex/LangChain object; 
                        # in a real scenario, we'd handle this more robustly.
                        import copy
                        new_doc = copy.copy(doc)
                        new_doc.content = truncated_content
                        selected_docs.append(new_doc)
                    else:
                        new_doc = doc.copy()
                        new_doc["content"] = truncated_content
                        selected_docs.append(new_doc)
                    self.last_selection_report["selected"].append({
                        "id": doc_id,
                        "source": source,
                        "truncated": True,
                    })
                    current_tokens += self.estimate_tokens(truncated_content)
                    if source is not None:
                        source_counts[source] = source_counts.get(source, 0) + 1
                else:
                    self._record_dropped(doc, "token_budget")
                break
        self.last_selection_report["used_tokens"] = current_tokens
        return selected_docs

    def _order_documents(self, context_docs: List[Any]) -> List[Any]:
        """Sort scored documents by relevance while preserving stable ties."""
        scored = []
        for position, doc in enumerate(context_docs):
            score = getattr(doc, "score", None)
            if score is None and isinstance(doc, dict):
                score = doc.get("score")
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                continue
            scored.append((position, numeric_score, doc))

        if not scored:
            return context_docs

        return [
            doc
            for _, _, doc in sorted(
                scored,
                key=lambda item: (-item[1], item[0]),
            )
        ] + [
            doc
            for position, doc in enumerate(context_docs)
            if not any(scored_position == position for scored_position, _, _ in scored)
        ]

    def _record_dropped(self, doc: Any, reason: str) -> None:
        """Record a document excluded from the current context."""
        content = doc.content if hasattr(doc, "content") else doc.get("content", "")
        self.last_selection_report["dropped"].append({
            "id": self._document_id(doc, content or ""),
            "reason": reason,
        })

    def _document_id(self, doc: Any, content: str) -> str:
        """Return a stable human-readable identifier for selection reports."""
        doc_id = getattr(doc, "doc_id", None)
        if doc_id is None and isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id")
        if doc_id:
            return str(doc_id)
        source = self._document_source(doc)
        return source or content[:80]

    def _document_key(self, doc: Any, content: str) -> tuple:
        """Return a stable key for removing duplicate retrieved chunks."""
        doc_id = getattr(doc, "doc_id", None)
        if doc_id is None and isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id")
        if doc_id:
            return ("id", str(doc_id))

        source = self._document_source(doc)
        return ("content", str(source or ""), content)

    def _document_source(self, doc: Any) -> Optional[str]:
        """Return a source identifier when the document provides one."""
        source = getattr(doc, "source", None)
        if source is None and isinstance(doc, dict):
            source = doc.get("source")
        if not source or str(source) == "document":
            return None
        return str(source)

    def _hard_truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit a token budget using tiktoken."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Leave room for the truncation marker
        marker = "... [Truncated]"
        marker_tokens = len(self.encoder.encode(marker))
        
        if max_tokens <= marker_tokens:
            return "" # Not enough room even for the marker
            
        truncated_tokens = tokens[:max_tokens - marker_tokens]
        return self.encoder.decode(truncated_tokens) + marker
