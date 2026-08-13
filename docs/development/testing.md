# Testing Guide

Comprehensive guide to the test suite and testing strategies.

## 📊 Test Suite Overview

**Current Status:** 35/35 tests passing ✅

```
Test Categories:
├── Documents Flow (4 tests)
│   ├── Upload document
│   ├── List documents
│   ├── Get document details
│   └── Delete document
│
├── Query Flow (6 tests)
│   ├── Simple query
│   ├── Chat conversation
│   ├── Empty query
│   ├── Long context
│   └── Query history
│
├── Retrieval Performance (12 tests)
│   ├── BM25 Performance (3 tests)
│   ├── Semantic Search (2 tests)
│   ├── Hybrid Search (5 tests)
│   └── Accuracy Metrics (2 tests)
│
├── Health & Status (3 tests)
├── Authentication (1 test)
├── Rate Limiting (1 test)
├── Admin Operations (4 tests)
├── Vector Store (1 test)
└── Other (3 tests)
```

## 🚀 Quick Start

### Run All Tests
```powershell
python -m pytest tests/ -v
```

### Run Specific Category
```powershell
# Performance tests
python -m pytest tests/test_retrieval_performance.py -v

# Query tests
python -m pytest tests/test_query_flow.py -v

# Document tests
python -m pytest tests/test_documents_flow.py -v
```

### Run Single Test
```powershell
python -m pytest tests/test_query_flow.py::test_simple_query_returns_answer -v
```

### Run with Coverage
```powershell
python -m pytest tests/ --cov=api --cov=rag --cov-report=html
```

---

## 📈 Performance Benchmarks

### BM25 Keyword Search
```
Test: test_bm25_keyword_search
┌─────────────────────────┐
│ Performance Metrics      │
├─────────────────────────┤
│ Execution Time: <50ms   │
│ Precision@3: 0.8+       │
│ Recall@3: 0.7+          │
└─────────────────────────┘

Status: ✅ PASS
```

### Semantic Search
```
Test: test_semantic_search_performance
┌─────────────────────────┐
│ Performance Metrics      │
├─────────────────────────┤
│ Execution Time: 100-200ms│
│ Precision@3: 0.85+       │
│ Recall@3: 0.75+          │
└─────────────────────────┘

Status: ✅ PASS
```

### Hybrid Search
```
Test: test_hybrid_search_with_alpha_parameter
┌─────────────────────────┐
│ Performance Metrics      │
├─────────────────────────┤
│ Execution Time: 150-300ms│
│ Precision@3: 0.90+       │
│ Recall@3: 0.80+          │
│ Best Alpha: 0.5-0.7      │
└─────────────────────────┘

Status: ✅ PASS
```

---

## 🧪 Test File Structure

### test_documents_flow.py (4 tests)
Tests for document upload, listing, and deletion.

```python
def test_document_upload_returns_document_id():
    """Upload creates document with ID."""
    # ✅ PASS

def test_list_documents_returns_all_uploads():
    """List returns all uploaded documents."""
    # ✅ PASS

def test_get_document_details_returns_metadata():
    """Getting document returns full details."""
    # ✅ PASS

def test_delete_document_removes_from_store():
    """Delete removes document from vector store."""
    # ✅ PASS
```

### test_query_flow.py (6 tests)
Tests for query execution and conversation handling.

```python
def test_simple_query_returns_answer():
    """Simple query produces answer."""
    # ✅ PASS

def test_chat_preserves_conversation_context():
    """Multi-turn chat remembers context."""
    # ✅ PASS

def test_empty_query_returns_error():
    """Empty query handled gracefully."""
    # ✅ PASS

def test_long_context_preserved_in_chat():
    """Long conversations maintain context."""
    # ✅ PASS

def test_query_metadata_includes_execution_time():
    """Query returns timing metrics."""
    # ✅ PASS

def test_query_sources_are_properly_attributed():
    """Answer cites source documents."""
    # ✅ PASS
```

### test_retrieval_performance.py (12 tests)
Performance and accuracy tests for retrieval methods.

```python
# BM25 Tests (3 tests)
def test_bm25_keyword_search():
    """BM25 search < 50ms."""
    # ✅ PASS

def test_bm25_matches_keywords():
    """BM25 matches query terms."""
    # ✅ PASS

def test_bm25_precision():
    """BM25 precision > 0.8."""
    # ✅ PASS

# Semantic Search Tests (2 tests)
def test_semantic_search_performance():
    """Semantic search 100-200ms."""
    # ✅ PASS

def test_semantic_search_similarity():
    """Semantic finds similar docs."""
    # ✅ PASS

# Hybrid Search Tests (5 tests)
def test_hybrid_search_combined_results():
    """Hybrid combines BM25 + semantic."""
    # ✅ PASS

def test_hybrid_search_with_alpha_parameter():
    """Alpha parameter controls blend."""
    # ✅ PASS

def test_hybrid_search_performance_acceptable():
    """Hybrid 150-300ms."""
    # ✅ PASS

def test_hybrid_vs_individual_methods():
    """Hybrid better than single method."""
    # ✅ PASS

def test_concurrent_hybrid_retrieval():
    """Async queries work correctly."""
    # ✅ PASS

# Accuracy Tests (2 tests)
def test_ndcg_calculation_correct():
    """NDCG calculated properly."""
    # ✅ PASS

def test_ranking_order_preserved():
    """Documents ranked by score."""
    # ✅ PASS
```

### test_health_details.py (3 tests)
Health checks and status endpoints.

```python
def test_health_endpoint_returns_200():
    """Health check works."""
    # ✅ PASS

def test_health_includes_component_status():
    """Health shows component status."""
    # ✅ PASS

def test_stats_endpoint_returns_metrics():
    """Stats endpoint provides data."""
    # ✅ PASS
```

### test_auth.py (1 test)
Authentication tests.

```python
def test_api_key_authentication():
    """API key auth works."""
    # ✅ PASS
```

### test_rate_limit.py (1 test)
Rate limiting tests.

```python
def test_rate_limiting_blocks_excessive_requests():
    """Rate limit enforced."""
    # ✅ PASS
```

### Other Test Files
Additional tests for admin operations, vector store, etc.

---

## ✅ Test Scenarios

### Happy Path (All Tests)
```
✅ Upload document → List → Query → Delete
✅ Simple query → Get answer
✅ Chat multi-turn → Get answers with context
✅ Performance acceptable for all methods
✅ Authentication works
✅ Rate limiting works
✅ Admin operations work
```

### Error Scenarios
```
❌ Empty query → Error message
❌ Document not found → 404
❌ Invalid API key → 401
❌ Rate limit exceeded → 429
❌ Vector store down → Fallback to BM25
❌ LLM unavailable → Return snippets
```

---

## 🛠️ Writing New Tests

### Test Template
```python
import pytest
from api.main import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

def test_new_feature(client):
    """Test description."""
    
    # Setup
    # ...
    
    # Execute
    response = client.post("/api/endpoint", json={...})
    
    # Assert
    assert response.status_code == 200
    assert response.json()["expected_field"] == "expected_value"
```

### Performance Test Template
```python
import asyncio
import time

async def test_performance_metric():
    """Measure execution time."""
    
    start = time.time()
    # Execute operation
    result = await some_operation()
    duration = time.time() - start
    
    # Assert performance target
    assert duration < 0.050  # 50ms
    assert result is not None
```

### Parametrized Tests
```python
import pytest

@pytest.mark.parametrize("alpha,expected_min_score", [
    (0.0, 0.5),    # Pure BM25
    (0.5, 0.6),    # Balanced
    (1.0, 0.65),   # Pure semantic
])
def test_alpha_parameter_effects(alpha, expected_min_score):
    """Test hybrid search with different alpha values."""
    # Test code...
    pass
```

---

## 📊 Metrics Tracked

### Performance Metrics
- **Query Latency:** p50, p95, p99
- **Throughput:** Queries per second
- **Cache Hit Rate:** Percentage of cached results
- **Error Rate:** Failed queries
- **Token Usage:** Tokens consumed per query

### Quality Metrics
- **NDCG@3:** Normalized Discounted Cumulative Gain
- **MRR:** Mean Reciprocal Rank
- **Precision@3:** Accuracy of top-3 results
- **Recall@3:** Coverage of relevant results

### Resource Metrics
- **Memory Usage:** RAM consumption
- **Disk Usage:** Vector store size
- **CPU Usage:** Processing overhead
- **Vector Store Size:** Number of documents

---

## 🔍 Test Execution Examples

### Example 1: Run Performance Tests with Timing
```powershell
python -m pytest tests/test_retrieval_performance.py -v -s --tb=short

# Output shows:
# test_bm25_keyword_search PASSED [50ms] ✅
# test_semantic_search_performance PASSED [150ms] ✅
# test_hybrid_search_with_alpha_parameter PASSED [200ms] ✅
```

### Example 2: Run with Coverage Report
```powershell
python -m pytest tests/ --cov=api --cov=rag --cov-report=html
# Opens htmlcov/index.html in browser
```

### Example 3: Run Single Test with Verbose Output
```powershell
python -m pytest tests/test_query_flow.py::test_simple_query_returns_answer -vv -s

# Output shows:
# test_simple_query_returns_answer PASSED
# >>> Query: "What is AI?"
# >>> Response: "AI is artificial intelligence..."
# >>> Execution time: 123ms
```

---

## 🎯 Test Coverage Goals

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| api/ | 85% | 90% | 📈 |
| rag/ | 80% | 90% | 📈 |
| data/ | 75% | 85% | 📈 |
| observability/ | 60% | 80% | 📈 |
| Overall | 80% | 90% | 📈 |

---

## 🚀 Continuous Integration (CI)

### GitHub Actions / Similar
```yaml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements_fastapi.txt
      - name: Run tests
        run: pytest tests/ -v --cov
      - name: Check coverage
        run: pytest --cov=api --cov=rag --cov-report=term-missing
```

---

## 📝 Test Best Practices

1. **One assertion per test** (or related assertions)
2. **Clear test names** that describe what's tested
3. **Use fixtures** for common setup
4. **Test edge cases** (empty, None, large values)
5. **Mock external services** (LLM, vector store)
6. **Keep tests fast** (<1s for unit tests)
7. **Run tests locally** before committing
8. **Maintain >80% coverage**

---

## 🔧 Debugging Failed Tests

### See Full Error
```powershell
pytest tests/test_name.py::test_func -vv -s
```

### Run Single Test with Debugger
```powershell
python -m pdb -m pytest tests/test_name.py::test_func
```

### Print Debug Info
```python
def test_example():
    result = function_under_test()
    print(f"Debug: {result}")  # Shows with -s flag
    assert result == expected
```

---

## 📚 Further Reading

- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest Documentation](https://docs.pytest.org/)

---

**Status:** All 35 tests passing ✅  
**Next Step:** Review [Phase 1 Development](./phase-1.md) for improvements
