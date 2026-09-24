"""Advanced retrieval with semantic + keyword search."""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime, timezone

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logging.warning("rank_bm25 not installed; keyword search unavailable")

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logging.warning("ChatNVIDIA not installed; reranking unavailable")

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDoc:
    """Retrieved document chunk."""
    doc_id: str
    content: str
    source: str
    score: float
    retrieval_method: str = "hybrid"  # semantic, keyword, or hybrid
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata or {},
        }


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Semantic search (vector similarity)
    - Keyword search (BM25)
    - Optional LLM reranking
    """
    
    def __init__(self, vector_store, use_reranker: bool = False, reranker: Optional[ChatNVIDIA] = None):
        """Initialize hybrid retriever."""
        self.vector_store = vector_store
        self.use_reranker = use_reranker and HAS_LANGCHAIN and reranker is not None
        self.reranker = reranker if self.use_reranker else None
        self.retrieval_history: Dict[str, Dict[str, Any]] = {}
        
        # Initialize BM25 with documents
        self.all_documents: List[Dict[str, Any]] = []
        self.bm25 = None
        
        logger.info("🔍 Initializing HybridRetriever")
        if self.use_reranker and not self.reranker:
            logger.warning("Reranking enabled but reranker model unavailable")
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for keyword search."""
        self.all_documents = documents
        
        if HAS_BM25:
            # Prepare corpus for BM25
            corpus = [doc.get("content", "").split() for doc in documents]
            self.bm25 = BM25Okapi(corpus)
            logger.info(f"✅ Indexed {len(documents)} documents for BM25")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 3,
        alpha: float = 0.5,
    ) -> tuple[List[RetrievedDoc], str]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Balance between semantic (1.0) and keyword (0.0)
        
        Returns:
            Tuple of (retrieved_docs, query_id)
        """
        query_id = str(uuid.uuid4())
        retrieval_time_start = datetime.now(timezone.utc)
        
        logger.debug(f"Retrieving for query: {query}")
        
        # Semantic search
        semantic_results = await self.semantic_retrieve(query, top_k)
        
        # Keyword search
        keyword_results = await self.keyword_retrieve(query, top_k) if HAS_BM25 else []
        
        # Combine results
        combined = self._combine_results(semantic_results, keyword_results, alpha, top_k)
        
        # Optional reranking
        if self.use_reranker and self.reranker:
            combined = await self.rerank_results(query, combined, top_k)
        
        # Track retrieval
        retrieval_time = (datetime.now(timezone.utc) - retrieval_time_start).total_seconds() * 1000
        self.retrieval_history[query_id] = {
            "query": query,
            "results_count": len(combined),
            "retrieval_time_ms": retrieval_time,
            "alpha": alpha,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"✅ Retrieved {len(combined)} documents in {retrieval_time:.1f}ms")
        return combined[:top_k], query_id
    
    async def semantic_retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievedDoc]:
        """Semantic search via vector store."""
        try:
            results = await self.vector_store.search(query, top_k=top_k)
            
            retrieved = []
            for result in results:
                doc = RetrievedDoc(
                    doc_id=result.get("doc_id", "unknown"),
                    content=result.get("content", ""),
                    source=result.get("metadata", {}).get("source", "unknown"),
                    score=result.get("score", 0),
                    retrieval_method="semantic",
                    metadata=result.get("metadata", {}),
                )
                retrieved.append(doc)
            
            logger.debug(f"Semantic retrieval: {len(retrieved)} results")
            return retrieved
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    async def keyword_retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[RetrievedDoc]:
        """Keyword search with BM25."""
        if not HAS_BM25 or not self.bm25:
            return []
        
        try:
            query_tokens = query.lower().split()
            scores = await asyncio.to_thread(self.bm25.get_scores, query_tokens)
            
            # Get top-k indices
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            retrieved = []
            for idx in ranked_indices:
                if scores[idx] > 0:
                    doc = self.all_documents[idx]
                    retrieved.append(RetrievedDoc(
                        doc_id=doc.get("doc_id", f"doc_{idx}"),
                        content=doc.get("content", ""),
                        source=doc.get("source", "unknown"),
                        score=float(scores[idx]),
                        retrieval_method="keyword",
                        metadata=doc.get("metadata", {}),
                    ))
            
            logger.debug(f"Keyword retrieval: {len(retrieved)} results")
            return retrieved
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    def _combine_results(
        self,
        semantic: List[RetrievedDoc],
        keyword: List[RetrievedDoc],
        alpha: float,
        top_k: int,
    ) -> List[RetrievedDoc]:
        """Combine semantic and keyword results with weighted scoring."""
        # Build score maps
        combined_map: Dict[str, RetrievedDoc] = {}
        
        # Add semantic results
        for i, doc in enumerate(semantic):
            normalized_score = (1.0 - i / len(semantic)) * alpha if semantic else 0
            combined_map[doc.doc_id] = doc
            combined_map[doc.doc_id].score = normalized_score
            combined_map[doc.doc_id].retrieval_method = "hybrid"
        
        # Add/update with keyword results
        for i, doc in enumerate(keyword):
            normalized_score = (1.0 - i / len(keyword)) * (1.0 - alpha) if keyword else 0
            if doc.doc_id in combined_map:
                # Average the scores
                combined_map[doc.doc_id].score = (
                    combined_map[doc.doc_id].score + normalized_score
                ) / 2
            else:
                doc.score = normalized_score
                doc.retrieval_method = "hybrid"
                combined_map[doc.doc_id] = doc
        
        # Sort by score
        combined = sorted(combined_map.values(), key=lambda x: x.score, reverse=True)
        return combined
    
    async def rerank_results(
        self,
        query: str,
        results: List[RetrievedDoc],
        top_k: int = 3,
    ) -> List[RetrievedDoc]:
        """Rerank results with an LLM-based relevance model, falling back to the existing score order when unavailable."""
        if not results:
            return []

        if not self.reranker:
            return sorted(results, key=lambda doc: doc.score, reverse=True)[:top_k]

        try:
            prompt_docs = []
            for idx, doc in enumerate(results[:top_k * 2]):
                snippet = doc.content.replace("\n", " ")
                if len(snippet) > 1000:
                    snippet = snippet[:997] + "..."
                prompt_docs.append(
                    f"{idx+1}. id={doc.doc_id} score={doc.score:.2f}\n{snippet}\n"
                )

            prompt_text = (
                "You are a relevance scorer. Rank the following documents by how useful they are to answer the question. "
                "Return a JSON list of objects with fields 'doc_id' and 'score' (0.0-1.0) only."
                f"\n\nQuestion: {query}\n\nDocuments:\n" + "\n".join(prompt_docs)
            )

            response = await self.reranker.ainvoke(prompt_text)
            if isinstance(response, dict):
                response_text = response.get("text", "")
            else:
                response_text = str(response)

            ranked = []
            parsed = None
            try:
                parsed = json.loads(response_text)
            except Exception:
                pass

            if isinstance(parsed, list):
                score_map = {}
                for item in parsed:
                    doc_id = item.get("doc_id")
                    score = float(item.get("score", 0.0))
                    score_map[doc_id] = score
                ranked = [doc for doc in results if doc.doc_id in score_map]
                ranked.sort(key=lambda d: score_map.get(d.doc_id, 0.0), reverse=True)
            else:
                # Fallback: parse lines like id=... score=...
                score_map = {}
                for line in response_text.splitlines():
                    if "doc_id" in line and "score" in line:
                        parts = line.replace(',', ' ').split()
                        doc_id = None
                        score = None
                        for part in parts:
                            if part.startswith("doc_id"):
                                doc_id = part.split("=")[-1]
                            if part.startswith("score"):
                                try:
                                    score = float(part.split("=")[-1])
                                except ValueError:
                                    score = None
                        if doc_id and score is not None:
                            score_map[doc_id] = score
                if score_map:
                    ranked = [doc for doc in results if doc.doc_id in score_map]
                    ranked.sort(key=lambda d: score_map.get(d.doc_id, 0.0), reverse=True)

            if ranked:
                for doc in ranked:
                    doc.score = round(doc.score, 4)
                return ranked[:top_k]
        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")

        return sorted(results, key=lambda doc: doc.score, reverse=True)[:top_k]
    
    def get_retrieval_diagnostics(self, query_id: str) -> Dict[str, Any]:
        """Get diagnostics for a retrieval operation."""
        if query_id not in self.retrieval_history:
            return {"error": "Query ID not found"}
        
        return self.retrieval_history[query_id]
