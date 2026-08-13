# Development Guide

Information for developers working on or testing the Hybrid RAG system.

## 📚 Quick Links

- **[Phase 1 Development](./phase-1.md)** - Implementation guide for Phase 1 improvements
- **[Testing Guide](./testing.md)** - Test suite, running tests, writing tests

## 🧪 Testing

### Quick Test Run
```powershell
# Run all tests
python -m pytest tests/ -v

# Expected: 35/35 tests passing ✅
```

### Performance Tests
```powershell
# Run performance benchmarks
python -m pytest tests/test_retrieval_performance.py -v

# See timing metrics for:
# - BM25 keyword search
# - Semantic search
# - Hybrid search
# - Concurrent queries
```

### Specific Test Categories
```powershell
# Documents workflow
python -m pytest tests/test_documents_flow.py -v

# Query workflow  
python -m pytest tests/test_query_flow.py -v

# Health and status
python -m pytest tests/test_health_details.py -v

# Security (auth, rate limiting)
python -m pytest tests/test_auth.py tests/test_rate_limit.py -v
```

---

## 🚀 Development Setup

### 1. Clone/Access Repository
```powershell
cd d:\projects\ai_ws\rag\multimodel_rag\hybrid_rag
```

### 2. Activate Virtual Environment
```powershell
.\genai\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements_fastapi.txt
```

### 4. Run Tests
```powershell
python -m pytest tests/ -v
```

### 5. Start Development Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📂 Project Structure

```
hybrid_rag/
├── api/                      # REST API implementation
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration & environment
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/              # Endpoint handlers
│       ├── documents.py
│       ├── query.py
│       ├── retrieval.py
│       └── admin.py
│
├── rag/                      # RAG system core
│   ├── hybrid_rag.py        # Main orchestrator
│   ├── retrieval.py         # Hybrid retrieval (BM25 + semantic)
│   ├── chains.py            # LLM chain definitions
│   ├── memory.py            # Conversation memory
│   └── indexing.py          # Document processing
│
├── data/                     # Data management
│   ├── vector_store.py      # ChromaDB abstraction
│   └── metadata.py          # Metadata storage
│
├── observability/            # Monitoring & metrics
│   ├── metrics.py           # Prometheus metrics
│   ├── tracing.py           # LangSmith integration
│   ├── logging_config.py    # Logging setup
│   └── __init__.py
│
├── tests/                    # Test suite (35 tests, 100% passing)
│   ├── test_documents_flow.py
│   ├── test_query_flow.py
│   ├── test_retrieval_performance.py
│   ├── test_auth.py
│   ├── test_rate_limit.py
│   ├── test_health_details.py
│   └── ...
│
├── docs/                     # Documentation (you are here)
│   ├── getting-started/
│   ├── architecture/
│   ├── development/          (this folder)
│   └── improvements/
│
├── requirements_fastapi.txt  # Python dependencies
├── .env.example             # Environment template
├── README.md                # Main project README
└── ...
```

---

## 🛠️ Common Development Tasks

### Running Tests
```powershell
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_query_flow.py -v

# Single test
pytest tests/test_query_flow.py::test_simple_query_returns_answer -v

# With coverage
pytest tests/ --cov=api --cov=rag
```

### Adding a New Endpoint
1. Create route handler in `api/routes/`
2. Add Pydantic models to `api/schemas.py`
3. Mount route in `api/main.py`
4. Write tests in `tests/`

### Adding a New Feature
1. Implement in appropriate module (`rag/`, `data/`, etc.)
2. Write tests for the feature
3. Update relevant schemas
4. Document in relevant files
5. Run full test suite to verify no breakage

### Debugging Issues
```powershell
# Enable debug logging
$env:DEBUG = "true"

# Run with verbose output
python -m pytest tests/ -vv -s

# Use Python debugger
python -m pdb -m pytest tests/test_query_flow.py
```

---

## 📊 Current Test Coverage

| Area | Tests | Status |
|------|-------|--------|
| Documents | 4 | ✅ Passing |
| Queries | 6 | ✅ Passing |
| Retrieval Performance | 12 | ✅ Passing |
| Health & Status | 3 | ✅ Passing |
| Authentication | 1 | ✅ Passing |
| Rate Limiting | 1 | ✅ Passing |
| Admin Operations | 4 | ✅ Passing |
| Vector Store | 1 | ✅ Passing |
| Other | 3 | ✅ Passing |
| **Total** | **35** | **✅ 100%** |

---

## 🎓 Learning Resources

### Understanding the System
1. Read [Architecture Overview](../architecture/overview.md)
2. Explore the FastAPI Swagger UI: http://localhost:8000/docs
3. Read source code in `rag/` and `api/`
4. Run and analyze performance tests

### Making Changes
1. Read [Implementation Roadmap](../../IMPLEMENTATION_ROADMAP.md) for improvement context
2. Create feature branch: `git checkout -b feature/my-feature`
3. Write tests first (TDD approach)
4. Implement feature
5. Ensure all tests pass
6. Submit for code review

### Understanding Performance
1. Run `test_retrieval_performance.py`
2. Review metrics in output
3. Check `observability/metrics.py` for available metrics
4. Use Prometheus UI if available

---

## 💡 Best Practices

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings for functions
- Keep functions focused and small

### Testing
- Write tests alongside features
- Aim for >80% code coverage
- Test both happy path and error cases
- Use fixtures for common setup

### Documentation
- Update docstrings when changing code
- Add comments for complex logic
- Keep README up to date
- Document API changes

### Performance
- Monitor query latency
- Check memory usage
- Profile slow operations
- Use caching appropriately

---

## 🔧 Troubleshooting Development Issues

### Tests Failing
```powershell
# Clear cache and reinstall
rm -r __pycache__ .pytest_cache
pip install --upgrade -r requirements_fastapi.txt

# Run single failing test with verbose output
pytest tests/test_name.py::test_func -vv -s
```

### Server Won't Start
```powershell
# Check port availability
netstat -ano | findstr :8000

# Try different port
python -m uvicorn api.main:app --port 8001
```

### Import Errors
```powershell
# Verify virtual environment is activated
python -c "import sys; print(sys.prefix)"

# Reinstall dependencies
pip install -r requirements_fastapi.txt
```

### Vector Store Issues
```powershell
# Reset ChromaDB (deletes all documents!)
rm -r data/chroma

# Or use in-memory store
$env:VECTOR_STORE_TYPE = "in-memory"
```

---

## 📈 Development Metrics

Monitor these metrics while developing:

- **Test Pass Rate:** Should be 100%
- **Code Coverage:** Aim for >80%
- **Query Latency:** Should stay <300ms for hybrid search
- **Error Rate:** Should be <1%
- **Cache Hit Rate:** Target >20% after Phase 2

---

## 🚀 Next Steps

### For New Developers
1. Read [Architecture Overview](../architecture/overview.md)
2. Follow [Getting Started](../getting-started/quick-start.md)
3. Run test suite
4. Explore Swagger UI
5. Read source code

### For Improving the System
1. Read [Improvements Overview](../improvements/)
2. Review [Implementation Roadmap](../../IMPLEMENTATION_ROADMAP.md)
3. Start Phase 1 improvements
4. Write tests for new features
5. Document changes

### For Contributing
1. Create feature branch
2. Write tests first
3. Implement feature
4. Run full test suite
5. Document changes
6. Submit for review

---

**See Also:**
- [Phase 1 Development](./phase-1.md) - Implementation guide
- [Testing Guide](./testing.md) - Comprehensive testing info
- [Architecture](../architecture/) - System design
