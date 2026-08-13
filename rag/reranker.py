"""Cross-Encoder Reranking for improving retrieval precision."""

from typing import List, Dict, Any
import numpy as np
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """A document with a refined reranked score."""
    doc: Dict[str, Any]
    rerank_score: float

class CrossEncoderReranker:
    """
    Reranks retrieved documents using a Cross-Encoder model.
    
    Unlike Bi-Encoders (used in vector search), Cross-Encoders process the 
    query and document simultaneously, allowing for much higher precision.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: The HuggingFace model name for the cross-encoder.
        """
        self.model_name = model_name
        # Lazy loading of the model to avoid importing torch/transformers 
        # until actually needed.
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the model and tokenizer from HuggingFace."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            try:
                # Try loading from local cache first to avoid network calls
                self._model = CrossEncoder(self.model_name, local_files_only=True)
                logger.info(f"Loaded reranker model {self.model_name} from local cache.")
            except Exception as e:
                logger.info(f"Local model not found or error loading: {e}. Attempting to download from HuggingFace...")
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Successfully downloaded and loaded reranker model {self.model_name}.")
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their actual relevance to the query.
        
        Args:
            query: The user's original query.
            documents: List of documents retrieved by the hybrid retriever.
            top_k: Number of top results to return.
            
        Returns:
            The top-k documents re-sorted by the cross-encoder score.
        """
        if not documents:
            return []

        self._load_model()
        
        # Prepare pairs for the cross-encoder: [[query, doc1], [query, doc2], ...]
        pairs = []
        for doc in documents:
            # Handle both dictionary and object (RetrievedDoc) types
            content = doc['content'] if isinstance(doc, dict) else getattr(doc, 'content', "")
            pairs.append([query, content])
        
        # Predict scores (higher is more relevant)
        start_time = time.perf_counter()
        scores = self._model.predict(pairs)
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        logger.info(f"Reranking {len(pairs)} documents took {processing_time:.4f} seconds")
        
        # Combine scores with documents
        reranked = []
        for i, score in enumerate(scores):
            doc = documents[i]
            # Create a copy if it's a dict, otherwise we'll just store the score in a wrapper or the object
            if isinstance(doc, dict):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(score)
                reranked.append(doc_copy)
            else:
                # For objects, we can't easily add a new attribute without modifying the class,
                # so we wrap it in a dict or just use the object and accept that rerank_score 
                # might be handled separately. To keep it consistent with the API, we'll use a dict.
                reranked.append({
                    "content": getattr(doc, 'content', ""),
                    "source": getattr(doc, 'source', "unknown"),
                    "score": getattr(doc, 'score', 0.0),
                    "rerank_score": float(score),
                    "metadata": getattr(doc, 'metadata', {})
                })
        
        # Sort by rerank_score descending
        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Debug: Log the top scores to see why documents might be filtered out later
        if reranked:
            logger.info(f"Top rerank score: {reranked[0]['rerank_score']:.4f}")
            logger.info(f"Bottom rerank score: {reranked[-1]['rerank_score']:.4f}")
        
        return reranked[:top_k]
