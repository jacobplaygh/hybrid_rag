import logging
import tiktoken
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Handles token budgeting, truncation, and context window management 
    for LLM prompts to prevent context overflow.
    """
    def __init__(self, max_tokens: int = 120000, system_prompt_reserve: int = 1000, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.system_prompt_reserve = system_prompt_reserve
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
        overhead += self.system_prompt_reserve

        available_tokens = self.max_tokens - overhead
        if available_tokens <= 0:
            logger.warning("Context window exhausted by query and history alone.")
            return []

        selected_docs = []
        current_tokens = 0

        for doc in context_docs:
            content = doc.content if hasattr(doc, "content") else doc.get("content", "")
            if content is None:
                content = ""
            doc_tokens = self.estimate_tokens(content)
            
            if current_tokens + doc_tokens <= available_tokens:
                selected_docs.append(doc)
                current_tokens += doc_tokens
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
                break
        
        return selected_docs

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
