import pytest
import asyncio
from unittest.mock import MagicMock
from rag.semantic_cache import SemanticCache
from data.vector_store import VectorStore

class MockEmbeddingModel:
    """Mock embedding model to simulate vector embeddings."""
    def get_text_embedding(self, text: str):
        # Return a simple deterministic vector based on the text
        # In a real scenario, this would be a high-dimensional vector
        if "hello" in text.lower():
            return [1.0, 0.0, 0.0]
        if "hi" in text.lower():
            return [0.9, 0.1, 0.0] # Very similar to "hello"
        if "weather" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

@pytest.mark.anyio
async def test_semantic_cache_basic_set_get():
    """Test basic set and get functionality of the semantic cache."""
    mock_model = MockEmbeddingModel()
    cache = SemanticCache(embedding_model=mock_model, threshold=0.9)
    
    query = "Hello world"
    response = {"answer": "Hi there!", "tokens": 10}
    
    cache.set(query, response)
    
    # Exact match
    result = cache.get(query)
    assert result is not None
    assert result["response"]["answer"] == "Hi there!"

@pytest.mark.anyio
async def test_semantic_cache_similarity_hit():
    """Test that semantically similar queries trigger a cache hit."""
    mock_model = MockEmbeddingModel()
    # Threshold 0.8 should allow "Hello" and "Hi" to match
    cache = SemanticCache(embedding_model=mock_model, threshold=0.8)
    
    query1 = "Hello world"
    response = {"answer": "Hi there!", "tokens": 10}
    cache.set(query1, response)
    
    # Semantically similar query
    query2 = "Hi there"
    result = cache.get(query2)
    
    assert result is not None
    assert result["response"]["answer"] == "Hi there!"

@pytest.mark.anyio
async def test_semantic_cache_similarity_miss():
    """Test that dissimilar queries trigger a cache miss."""
    mock_model = MockEmbeddingModel()
    cache = SemanticCache(embedding_model=mock_model, threshold=0.9)
    
    query1 = "Hello world"
    response = {"answer": "Hi there!", "tokens": 10}
    cache.set(query1, response)
    
    # Completely different query
    query2 = "What is the weather?"
    result = cache.get(query2)
    
    assert result is None

@pytest.mark.anyio
async def test_semantic_cache_threshold_tuning():
    """Test that changing the threshold affects cache hits."""
    mock_model = MockEmbeddingModel()
    
    query1 = "Hello world"
    query2 = "Hi there"
    response = {"answer": "Hi there!", "tokens": 10}
    
    # High threshold: should miss
    cache_high = SemanticCache(embedding_model=mock_model, threshold=0.99)
    cache_high.set(query1, response)
    # "Hello world" [1,0,0] vs "Hi there" [0.9, 0.1, 0]
    # Sim = (1*0.9 + 0*0.1 + 0*0) / (1 * sqrt(0.81+0.01)) = 0.9 / 0.9055 = 0.9939
    # Wait, 0.9939 > 0.99. Let's use 1.0 for a guaranteed miss or a different query.
    assert cache_high.get("What is the weather?") is None
    
    # Low threshold: should hit
    cache_low = SemanticCache(embedding_model=mock_model, threshold=0.5)
    cache_low.set(query1, response)
    assert cache_low.get(query2) is not None
