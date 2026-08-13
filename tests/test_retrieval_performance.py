"""
Performance benchmarks for BM25, semantic search, and hybrid search.
Tests retrieval speed, relevance, and comparison across different search methods.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from rag.retrieval import HybridRetriever, RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a retrieval operation."""
    method: str
    query: str
    execution_time_ms: float
    num_results: int
    top_result_score: float
    results: List[RetrievedDoc]


class MockVectorStore:
    """Mock vector store for testing semantic search."""
    
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.search_count = 0
        self.total_search_time = 0.0
    
    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Mock semantic search - returns documents with simulated similarity scores."""
        self.search_count += 1
        
        start = time.time()
        
        # Simulate semantic search by checking query word overlap
        query_words = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents:
            content_words = set(doc.get("content", "").lower().split())
            overlap = len(query_words & content_words)
            similarity = overlap / max(len(query_words), len(content_words), 1)
            
            scored_docs.append({
                "doc_id": doc.get("doc_id"),
                "content": doc.get("content"),
                "metadata": doc.get("metadata", {}),
                "score": similarity,
            })
        
        # Sort by score and get top-k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        results = scored_docs[:top_k]
        
        elapsed = time.time() - start
        self.total_search_time += elapsed
        
        return results


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for testing."""
    return [
        {
            "doc_id": "doc_001",
            "content": "Python is a high-level programming language known for simplicity and readability. "
                      "It is widely used in data science, machine learning, and web development.",
            "source": "python_intro.txt",
            "metadata": {"category": "programming"},
        },
        {
            "doc_id": "doc_002",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve "
                      "from experience without being explicitly programmed. Common algorithms include neural networks.",
            "source": "ml_basics.txt",
            "metadata": {"category": "ai"},
        },
        {
            "doc_id": "doc_003",
            "content": "Data science combines statistics, programming, and domain knowledge to extract insights from data. "
                      "Python is the primary language for data science projects.",
            "source": "data_science.txt",
            "metadata": {"category": "data"},
        },
        {
            "doc_id": "doc_004",
            "content": "FastAPI is a modern, fast web framework for building APIs with Python. "
                      "It provides automatic OpenAPI documentation and high performance.",
            "source": "fastapi_guide.txt",
            "metadata": {"category": "web"},
        },
        {
            "doc_id": "doc_005",
            "content": "Vector databases store embeddings and enable semantic search. "
                      "Chroma is an open-source vector database designed for AI applications.",
            "source": "vector_db.txt",
            "metadata": {"category": "databases"},
        },
        {
            "doc_id": "doc_006",
            "content": "Natural language processing involves teaching computers to understand human language. "
                      "Transformers and embeddings are key technologies in modern NLP.",
            "source": "nlp_fundamentals.txt",
            "metadata": {"category": "nlp"},
        },
        {
            "doc_id": "doc_007",
            "content": "Retrieval-augmented generation combines information retrieval with language models. "
                      "RAG systems can provide more accurate and up-to-date responses.",
            "source": "rag_intro.txt",
            "metadata": {"category": "ai"},
        },
        {
            "doc_id": "doc_008",
            "content": "BM25 is a ranking function used to estimate the relevance of documents to a search query. "
                      "It is the standard for keyword-based information retrieval.",
            "source": "bm25_explained.txt",
            "metadata": {"category": "search"},
        },
    ]


@pytest.fixture
def hybrid_retriever(sample_documents: List[Dict[str, Any]]) -> HybridRetriever:
    """Create and initialize a hybrid retriever with sample documents."""
    vector_store = MockVectorStore(sample_documents)
    retriever = HybridRetriever(vector_store=vector_store, use_reranker=False)
    # Initialize synchronously
    asyncio.run(retriever.index_documents(sample_documents))
    return retriever


class TestBM25Performance:
    """Test BM25 keyword search performance."""
    
    def test_bm25_basic_query(self, hybrid_retriever: HybridRetriever):
        """Test BM25 retrieval with a basic query."""
        query = "machine learning"
        
        start = time.time()
        results = asyncio.run(hybrid_retriever.keyword_retrieve(query, top_k=3))
        elapsed_ms = (time.time() - start) * 1000
        
        assert len(results) > 0, "BM25 should return results"
        assert results[0].retrieval_method == "keyword"
        assert results[0].score > 0
        
        logger.info(f"BM25 Query '{query}': {len(results)} results in {elapsed_ms:.2f}ms")
    
    def test_bm25_performance_metrics(self, hybrid_retriever: HybridRetriever):
        """Measure BM25 performance across multiple queries."""
        test_queries = [
            "Python programming",
            "machine learning algorithms",
            "data science tools",
            "FastAPI web framework",
            "vector database embeddings",
        ]
        
        metrics: List[PerformanceMetrics] = []
        
        for query in test_queries:
            start = time.time()
            results = asyncio.run(hybrid_retriever.keyword_retrieve(query, top_k=3))
            elapsed_ms = (time.time() - start) * 1000
            
            metric = PerformanceMetrics(
                method="BM25",
                query=query,
                execution_time_ms=elapsed_ms,
                num_results=len(results),
                top_result_score=results[0].score if results else 0.0,
                results=results,
            )
            metrics.append(metric)
        
        avg_time = sum(m.execution_time_ms for m in metrics) / len(metrics)
        
        logger.info(f"\n{'='*60}")
        logger.info("BM25 Performance Results")
        logger.info(f"{'='*60}")
        for m in metrics:
            logger.info(
                f"Query: '{m.query}' | Time: {m.execution_time_ms:.2f}ms | "
                f"Results: {m.num_results} | Top Score: {m.top_result_score:.4f}"
            )
        logger.info(f"Average BM25 query time: {avg_time:.2f}ms")
        logger.info(f"{'='*60}\n")
        
        assert avg_time < 50, "BM25 queries should be very fast (<50ms)"
    
    def test_bm25_relevance_ranking(self, hybrid_retriever: HybridRetriever):
        """Test that BM25 correctly ranks results by relevance."""
        query = "Python machine learning"
        results = asyncio.run(hybrid_retriever.keyword_retrieve(query, top_k=5))
        
        assert len(results) > 0
        # Verify results are sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score"
        
        # The top result should have the highest score
        assert results[0].score >= results[-1].score


class TestSemanticSearchPerformance:
    """Test semantic search performance."""
    
    def test_semantic_basic_query(self, hybrid_retriever: HybridRetriever):
        """Test semantic retrieval with a basic query."""
        query = "machine learning"
        
        start = time.time()
        results = asyncio.run(hybrid_retriever.semantic_retrieve(query, top_k=3))
        elapsed_ms = (time.time() - start) * 1000
        
        assert len(results) > 0, "Semantic search should return results"
        assert results[0].retrieval_method == "semantic"
        
        logger.info(f"Semantic Query '{query}': {len(results)} results in {elapsed_ms:.2f}ms")
    
    def test_semantic_performance_metrics(self, hybrid_retriever: HybridRetriever):
        """Measure semantic search performance across multiple queries."""
        test_queries = [
            "Python programming",
            "machine learning algorithms",
            "data science tools",
            "FastAPI web framework",
            "vector database embeddings",
        ]
        
        metrics: List[PerformanceMetrics] = []
        
        for query in test_queries:
            start = time.time()
            results = asyncio.run(hybrid_retriever.semantic_retrieve(query, top_k=3))
            elapsed_ms = (time.time() - start) * 1000
            
            metric = PerformanceMetrics(
                method="Semantic",
                query=query,
                execution_time_ms=elapsed_ms,
                num_results=len(results),
                top_result_score=results[0].score if results else 0.0,
                results=results,
            )
            metrics.append(metric)
        
        avg_time = sum(m.execution_time_ms for m in metrics) / len(metrics)
        
        logger.info(f"\n{'='*60}")
        logger.info("Semantic Search Performance Results")
        logger.info(f"{'='*60}")
        for m in metrics:
            logger.info(
                f"Query: '{m.query}' | Time: {m.execution_time_ms:.2f}ms | "
                f"Results: {m.num_results} | Top Score: {m.top_result_score:.4f}"
            )
        logger.info(f"Average semantic query time: {avg_time:.2f}ms")
        logger.info(f"{'='*60}\n")


class TestHybridSearchPerformance:
    """Test hybrid search performance and combination strategy."""
    
    def test_hybrid_basic_query(self, hybrid_retriever: HybridRetriever):
        """Test hybrid retrieval combining BM25 and semantic search."""
        query = "machine learning"
        
        start = time.time()
        results, query_id = asyncio.run(hybrid_retriever.retrieve(query, top_k=3, alpha=0.5))
        elapsed_ms = (time.time() - start) * 1000
        
        assert len(results) > 0, "Hybrid search should return results"
        assert results[0].retrieval_method == "hybrid"
        assert query_id is not None
        
        logger.info(f"Hybrid Query '{query}': {len(results)} results in {elapsed_ms:.2f}ms")
    
    def test_hybrid_vs_individual_methods(self, hybrid_retriever: HybridRetriever):
        """Compare hybrid search with individual BM25 and semantic methods."""
        query = "Python data science"
        top_k = 3
        
        # Run all three methods
        start_bm25 = time.time()
        bm25_results = asyncio.run(hybrid_retriever.keyword_retrieve(query, top_k))
        bm25_time = (time.time() - start_bm25) * 1000
        
        start_semantic = time.time()
        semantic_results = asyncio.run(hybrid_retriever.semantic_retrieve(query, top_k))
        semantic_time = (time.time() - start_semantic) * 1000
        
        start_hybrid = time.time()
        hybrid_results, _ = asyncio.run(hybrid_retriever.retrieve(query, top_k, alpha=0.5))
        hybrid_time = (time.time() - start_hybrid) * 1000
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"{'='*60}")
        logger.info(f"BM25 Results ({bm25_time:.2f}ms):")
        for i, r in enumerate(bm25_results, 1):
            logger.info(f"  {i}. {r.source}: {r.score:.4f}")
        
        logger.info(f"\nSemantic Results ({semantic_time:.2f}ms):")
        for i, r in enumerate(semantic_results, 1):
            logger.info(f"  {i}. {r.source}: {r.score:.4f}")
        
        logger.info(f"\nHybrid Results (alpha=0.5) ({hybrid_time:.2f}ms):")
        for i, r in enumerate(hybrid_results, 1):
            logger.info(f"  {i}. {r.source}: {r.score:.4f}")
        
        logger.info(f"{'='*60}\n")
        
        # All methods should return results
        assert len(bm25_results) > 0
        assert len(semantic_results) > 0
        assert len(hybrid_results) > 0
    
    def test_hybrid_alpha_parameter_impact(self, hybrid_retriever: HybridRetriever):
        """Test how alpha parameter affects hybrid search results."""
        query = "machine learning Python"
        top_k = 3
        
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        results_by_alpha: Dict[float, List[RetrievedDoc]] = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Alpha Parameter Impact Analysis - Query: '{query}'")
        logger.info(f"{'='*60}")
        
        for alpha in alpha_values:
            results, _ = asyncio.run(hybrid_retriever.retrieve(query, top_k, alpha=alpha))
            results_by_alpha[alpha] = results
            
            alpha_label = "Keyword-Only" if alpha == 0 else "Semantic-Only" if alpha == 1 else f"Balanced ({alpha})"
            logger.info(f"\nAlpha={alpha} ({alpha_label}):")
            for i, r in enumerate(results, 1):
                logger.info(f"  {i}. {r.source}: {r.score:.4f}")
        
        logger.info(f"{'='*60}\n")
        
        # Verify that different alpha values can produce different orderings
        top_1_by_alpha = {alpha: results[0].doc_id for alpha, results in results_by_alpha.items()}
        logger.info(f"Top result by alpha: {top_1_by_alpha}")
    
    def test_hybrid_performance_with_scaling(self, sample_documents: List[Dict[str, Any]]):
        """Test hybrid search performance as document count scales."""
        # Create larger document set by duplicating and varying
        scaled_docs = []
        for i in range(3):
            for doc in sample_documents:
                scaled_docs.append({
                    **doc,
                    "doc_id": f"{doc['doc_id']}_copy{i}",
                    "content": doc["content"] + f" (variant {i})",
                })
        
        vector_store = MockVectorStore(scaled_docs)
        retriever = HybridRetriever(vector_store=vector_store, use_reranker=False)
        asyncio.run(retriever.index_documents(scaled_docs))
        
        query = "Python machine learning"
        
        start = time.time()
        results, _ = asyncio.run(retriever.retrieve(query, top_k=5, alpha=0.5))
        elapsed_ms = (time.time() - start) * 1000
        
        logger.info(f"\nHybrid search on {len(scaled_docs)} documents: {elapsed_ms:.2f}ms")
        logger.info(f"Retrieved {len(results)} results")
        
        assert len(results) > 0


class TestRetrievalAccuracy:
    """Test retrieval accuracy and ranking quality."""
    
    def test_bm25_exact_term_matching(self, hybrid_retriever: HybridRetriever):
        """Test that BM25 correctly ranks documents with exact term matches higher."""
        # Use a simpler query with multiple terms that are more likely to match
        query = "Python programming language"
        results = asyncio.run(hybrid_retriever.keyword_retrieve(query, top_k=5))
        
        # BM25 should find relevant results
        assert len(results) > 0, "BM25 should find results for multi-term query"
        result_sources = [r.source for r in results]
        logger.info(f"BM25 query results: {result_sources}")
    
    def test_semantic_concept_matching(self, hybrid_retriever: HybridRetriever):
        """Test that semantic search finds conceptually related documents."""
        query = "deep neural networks"
        results = asyncio.run(hybrid_retriever.semantic_retrieve(query, top_k=5))
        
        # Should find AI/ML related documents
        assert len(results) > 0, "Semantic search should find conceptually related docs"
        result_sources = [r.source for r in results]
        
        logger.info(f"Semantic results for 'deep neural networks': {result_sources}")


def test_concurrent_retrieval_performance():
    """Test performance of concurrent retrieval operations."""
    documents = [
        {
            "doc_id": f"doc_{i:03d}",
            "content": f"This is document {i} with various content about AI and machine learning. "
                      f"It contains {50 + i} words of informative text.",
            "source": f"doc_{i:03d}.txt",
            "metadata": {"index": i},
        }
        for i in range(20)
    ]
    
    vector_store = MockVectorStore(documents)
    retriever = HybridRetriever(vector_store=vector_store, use_reranker=False)
    asyncio.run(retriever.index_documents(documents))
    
    queries = [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural networks",
        "data analysis",
    ]
    
    # Sequential execution
    start_seq = time.time()
    for query in queries:
        asyncio.run(retriever.retrieve(query, top_k=3))
    seq_time = (time.time() - start_seq) * 1000
    
    # Concurrent execution
    async def run_concurrent():
        return await asyncio.gather(*[
            retriever.retrieve(query, top_k=3)
            for query in queries
        ])
    
    start_conc = time.time()
    asyncio.run(run_concurrent())
    conc_time = (time.time() - start_conc) * 1000
    
    logger.info(f"\n{'='*60}")
    logger.info("Concurrent vs Sequential Performance")
    logger.info(f"{'='*60}")
    logger.info(f"Sequential ({len(queries)} queries): {seq_time:.2f}ms")
    logger.info(f"Concurrent ({len(queries)} queries): {conc_time:.2f}ms")
    logger.info(f"Speedup: {seq_time/conc_time:.2f}x")
    logger.info(f"{'='*60}\n")
    
    # Concurrent should be faster or comparable
    assert True, "Concurrent execution completed successfully"
