"""Semantic caching for RAG queries using embeddings."""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Implements semantic caching by storing query embeddings and 
    retrieving responses based on cosine similarity.
    """
    
    def __init__(self, embedding_model, threshold: float = 0.9):
        """
        Initialize semantic cache.
        
        Args:
            embedding_model: Model used to generate embeddings for queries.
            threshold: Similarity threshold for a cache hit.
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        # Store as: { embedding_vector: { "query": str, "response": dict, "timestamp": str } }
        self.cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}

    def _get_embedding(self, text: str):
        """Generate embedding for the given text."""
        try:
            # Try common embedding method names
            for method_name in ["get_embedding", "embed_query", "embed"]:
                method = getattr(self.embedding_model, method_name, None)
                if callable(method):
                    embedding = method(text)
                    return tuple(embedding)

            raise AttributeError(f"Embedding model {type(self.embedding_model)} has no compatible embedding method")
        except Exception as exc:
            logger.error(f"Failed to generate embedding for cache: {exc}")
            return None

    def _cosine_similarity(self, v1: Tuple[float, ...], v2: Tuple[float, ...]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(v1)
        b = np.array(v2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if a semantically similar query exists.
        
        Returns:
            The cached response dictionary if a hit is found, else None.
        """
        query_emb = self._get_embedding(query)
        if query_emb is None or not self.cache:
            return None

        best_sim = -1.0
        best_match = None

        for cached_emb, data in self.cache.items():
            sim = self._cosine_similarity(query_emb, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_match = data

        if best_sim >= self.threshold:
            logger.info(f"🎯 Semantic cache hit! Similarity: {best_sim:.4f}")
            return best_match
        
        return None

    def set(self, query: str, response: Dict[str, Any]):
        """Store a query and its response in the cache."""
        query_emb = self._get_embedding(query)
        if query_emb is not None:
            self.cache[query_emb] = {
                "query": query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            logger.debug(f"💾 Cached response for query: {query[:50]}...")
