# Phase 1a: FastAPI Backend Core Setup

## Overview

Phase 1a establishes the FastAPI backend skeleton with core infrastructure:
- FastAPI application entry point with middleware configuration
- Request/response schemas using Pydantic
- Placeholder route modules for future implementation
- Configuration management with environment variables
- Error handling and logging

## Directory Structure Created

```
multimodel_rag/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration from environment
│   ├── schemas.py           # Pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       ├── documents.py     # Document CRUD endpoints (implemented)
│       ├── query.py         # Query/chat endpoints (implemented)
│       ├── retrieval.py     # Retrieval diagnostics (implemented)
│       └── admin.py         # Admin/control endpoints (implemented)
├── rag/
│   ├── __init__.py
│   ├── hybrid_rag.py        # Main RAG orchestrator (implemented)
│   ├── indexing.py          # LlamaIndex integration (implemented)
│   ├── retrieval.py         # Hybrid retrieval logic (implemented)
│   ├── memory.py            # Conversation memory
│   └── chains.py            # LangChain definitions (implemented)
├── observability/
│   ├── __init__.py
│   ├── logging_config.py    # Structured JSON logging
│   ├── metrics.py           # Prometheus metrics
│   └── tracing.py           # LangSmith integration
├── data/
│   ├── __init__.py
│   └── vector_store.py      # Vector store abstraction
├── requirements_fastapi.txt # Phase 1 dependencies
├── .env.example             # Configuration template
└── PHASE_1_SETUP.md        # This file
```

## Installation

### 1. Setup Python Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Or Command Prompt
venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
# Install Phase 1a minimal requirements
pip install -r requirements_fastapi.txt

# Verify installations
pip list
```

### 3. Configure Environment

```powershell
# Copy example configuration
Copy-Item .env.example .env

# Edit .env and set your values (especially NVIDIA_API_KEY)
notepad .env

# Or set NVIDIA_API_KEY directly in PowerShell session
$env:NVIDIA_API_KEY = 'nvapi-xxxxx'
```

### 4. Verify Setup

```bash
# Test FastAPI app can start (will show warning about routes not yet implemented)
python -m api.main

# In another terminal, test health endpoint
curl http://localhost:8000/health

# View API documentation
# Visit: http://localhost:8000/docs (Swagger UI)
# Visit: http://localhost:8000/redoc (ReDoc)
```

## Current Status: Phase 1a Foundation

### ✅ Completed
- [x] FastAPI application skeleton with CORS, error handlers, lifespan events
- [x] Pydantic schemas for all major operations (documents, queries, chat, retrieval, admin)
- [x] Configuration system with environment variables and `.env` support
- [x] Placeholder route modules with API endpoint signatures
- [x] Core RAG module structure (memory, chains, retrieval, indexing, orchestration)
- [x] Observability foundation (logging, metrics, tracing)
- [x] Vector store abstraction layer (Chroma, Pinecone support)

### 🔄 Next Steps: Phase 1b (Document Management)
After Phase 1a is verified working, proceed to:

1. **Implement Document Routes** (`api/routes/documents.py`)
   - POST `/documents/upload` - File upload with LlamaIndex ingestion
   - GET `/documents/list` - List indexed documents
   - DELETE `/documents/{doc_id}` - Remove from index

2. **Implement Document Indexing** (`rag/indexing.py`)
   - LlamaIndex SimpleDirectoryReader for multi-format support
   - Adaptive text splitting strategies
   - Metadata extraction

3. **Implement Vector Store** (`data/vector_store.py`)
   - Chroma database initialization and persistence
   - Document embedding and storage
   - Search functionality

## API Endpoints Overview (Planned)

### Health & Info
- `GET /health` - Health check with service status
- `GET /` - API info and links to documentation

### Documents (Phase 1b)
- `POST /api/documents/upload` - Upload document(s)
- `GET /api/documents/list` - List indexed documents
- `DELETE /api/documents/{doc_id}` - Delete document

### Query & Chat (Phase 1c)
- `POST /api/query` - Single RAG query
- `POST /api/query/stream` - Streaming query response
- `POST /api/chat` - Multi-turn conversation
- `POST /api/chat/stream` - Streaming chat response

### Retrieval Diagnostics (Phase 1c)
- `GET /api/retrieval/{query_id}` - Retrieval diagnostics
- `GET /api/retrieval` - Recent retrievals

### Admin (Phase 1d)
- `POST /api/admin/rebuild-index` - Rebuild vector index
- `POST /api/admin/clear-cache` - Clear cache
- `GET /api/admin/index-stats` - Index statistics

### Observability (Phase 1d)
- `GET /metrics` - Prometheus metrics (port 9090)
- Structured JSON logs to `logs/app.log`
- LangSmith tracing for all LLM calls

## Testing

### Manual Testing

```bash
# Start server
python -m api.main

# In another terminal:

# Test health check
curl http://localhost:8000/health

# Visit interactive API docs
# http://localhost:8000/docs
```

### Automated Testing (Phase 1b+)

```bash
# Run tests (when implemented)
pytest tests/ -v

# With coverage
pytest tests/ --cov=api --cov=rag
```

## Configuration Details

### Environment Variables

See `.env.example` for all configurable options:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NVIDIA_API_KEY` | Required | API key for NVIDIA AI Endpoints |
| `VECTOR_STORE_TYPE` | `chroma` | Vector database backend |
| `CHUNK_SIZE` | 512 | Document chunk size for indexing |
| `RETRIEVAL_K` | 3 | Number of documents to retrieve |
| `LANGSMITH_ENABLED` | False | Enable LangSmith tracing |
| `DEBUG` | False | Debug mode with verbose logging |

### CORS Configuration

By default, CORS allows requests from:
- `http://localhost:3000` (React dev server)
- `http://localhost:8080` (Alternative dev)
- `http://localhost:8501` (Streamlit)

Modify in `api/config.py` for production.

## Troubleshooting

### ImportError: ModuleNotFoundError
```bash
# Ensure you're in virtual environment
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Install requirements again
pip install -r requirements_fastapi.txt
```

### NVIDIA_API_KEY Not Available
```powershell
# Set in current session
$env:NVIDIA_API_KEY = 'nvapi-xxxxx'

# Or persistently (Windows)
setx NVIDIA_API_KEY "nvapi-xxxxx"
```

### Port 8000 Already in Use
```bash
# Use different port
set PORT=8001
python -m api.main

# Or kill process using port 8000
# Linux/macOS: lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill
# Windows: netstat -ano | findstr :8000
```

## Next Phase: Phase 1b - Document Management

Once Phase 1a is verified working, proceed to implement:

1. Full document upload pipeline with file validation
2. LlamaIndex SimpleDirectoryReader integration
3. Chroma vector store initialization and document storage
4. Document metadata extraction and tracking

See ARCHITECTURE.md for complete Phase 1b specifications.
