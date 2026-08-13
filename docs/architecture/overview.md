# Enhancement Architecture: Hybrid RAG System with FastAPI + LlamaIndex

## Overview

This document outlines the enhancement strategy for the existing `multimodel_rag` Streamlit application. The enhancement adds a FastAPI backend with LlamaIndex integration, advanced RAG capabilities, REST API endpoints, and observability features—while keeping the Streamlit UI intact for interactive exploration.

## Goals

1. **Dual-Framework Integration**: Combine LangChain (orchestration) + LlamaIndex (indexing & retrieval)
2. **REST API Layer**: Expose RAG operations via FastAPI for programmatic access
3. **Advanced Retrieval**: Hybrid search, multi-source ingestion, adaptive chunking
4. **Observability**: LangSmith tracing, Prometheus metrics, structured logging
5. **Data Management**: Feature stores, vector DB abstraction, metadata filtering
6. **Production-Ready**: Scalable architecture, error handling, monitoring dashboards

## Architecture

### Current State (Streamlit Only)
```
User (Browser)
    ↓
Streamlit UI (main.py)
    ↓
LangChain + In-Memory Vector Store
    ↓
NVIDIA AI Endpoints
```

### Enhanced State (FastAPI + Streamlit)
```
REST Clients / External Systems
    ↓
FastAPI Backend (api/main.py)
    ├── Document Ingestion (LlamaIndex)
    ├── Hybrid Retrieval (LangChain + LlamaIndex)
    ├── Query Orchestration
    └── Observability (LangSmith, Prometheus)
    ↓
Vector Store (Chroma / Pinecone)
    ↓
NVIDIA AI Endpoints

User (Browser)
    ↓
Enhanced Streamlit UI (main_app.py)
    ├── Call FastAPI backend
    ├── Display results
    └── Show observability dashboard
```

## Directory Structure

```
multimodel_rag/
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Configuration & env vars
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── documents.py           # POST/GET/DELETE documents
│   │   ├── query.py               # Query, chat, stream endpoints
│   │   ├── retrieval.py           # Retrieval diagnostics
│   │   └── admin.py               # Index operations
│   └── schemas.py                 # Pydantic models
│
├── rag/
│   ├── __init__.py
│   ├── hybrid_rag.py              # LangChain + LlamaIndex orchestrator
│   ├── indexing.py                # LlamaIndex document indexing
│   ├── retrieval.py               # Retrieval strategies (hybrid, reranking)
│   ├── memory.py                  # Conversation memory & state
│   └── chains.py                  # LangChain chain definitions
│
├── observability/
│   ├── __init__.py
│   ├── logging_config.py          # Structured logging setup
│   ├── metrics.py                 # Prometheus metrics & collectors
│   ├── tracing.py                 # LangSmith integration
│   └── dashboard.py               # Streamlit monitoring dashboard
│
├── data/
│   ├── __init__.py
│   ├── vector_store.py            # Vector DB abstraction
│   ├── metadata.py                # Metadata management
│   └── feature_store.py           # Cached embeddings & metadata
│
├── main.py                        # CLI/runner (for FastAPI server)
├── main_app.py                    # Enhanced Streamlit app (calls FastAPI)
├── requirements.txt               # Updated dependencies
├── README.md                      # Updated setup & usage
├── ARCHITECTURE.md                # This file
├── .gitignore                     # Updated ignore rules
└── docker-compose.yml             # Local development (optional)
```

## Key Components

### 1. FastAPI Backend (`api/main.py`)
- Async endpoints for document operations
- Streaming responses for long queries
- Request validation with Pydantic
- CORS for Streamlit cross-origin calls
- Health checks & liveness probes

**Endpoints:**
```
POST   /documents/upload           # Upload & ingest documents
GET    /documents/list             # List indexed documents
DELETE /documents/{doc_id}         # Remove document
POST   /query                      # Single query
POST   /chat                       # Multi-turn conversation
GET    /retrieval/{query_id}       # Diagnostics (sources, scores)
POST   /admin/rebuild-index        # Rebuild vector store
POST   /admin/clear-cache          # Clear embeddings cache
GET    /health                     # Health check
```

### 2. Hybrid RAG Orchestrator (`rag/hybrid_rag.py`)
- LlamaIndex `SimpleDirectoryReader` for multi-format ingestion
- LangChain `Document` objects for compatibility
- Dual-retriever: semantic (vector) + keyword (BM25)
- Reranking with LangChain `LLMChain`
- Source attribution & confidence scoring

**Workflow:**
```
Document → LlamaIndex Index → Chroma Vector Store
Query → Decomposer → Dual Retrieval → Reranking → Synthesis
```

### 3. LlamaIndex Indexing (`rag/indexing.py`)
- Adaptive text splitting (semantic, recursive, hierarchical)
- Multi-type document support (PDF, PPTX, CSV, JSON, web)
- Metadata extraction & preservation
- Index caching for performance

### 4. Conversation Memory (`rag/memory.py`)
- Multi-turn context window management
- Summary-based memory compression for long conversations
- User session isolation
- SQLite or in-memory storage options

### 5. Observability (`observability/`)
- **LangSmith**: Trace LLM calls, latency, token usage
- **Prometheus**: Metrics (request count, latency, cache hits)
- **Structured Logging**: JSON logs with context
- **Streamlit Dashboard**: Real-time monitoring, query analytics

### 6. Vector Store Abstraction (`data/vector_store.py`)
- Factory pattern for Chroma / Pinecone / other backends
- Metadata filtering support
- Batch operations for efficiency
- Connection pooling

## Integration Flow

### Document Ingestion
1. User uploads file via FastAPI or Streamlit
2. FastAPI stores file, calls `LlamaIndex SimpleDirectoryReader`
3. Documents split using adaptive chunking strategy
4. Embeddings generated via NVIDIA API
5. Metadata extracted and stored
6. Vectors + metadata saved to Chroma
7. Index state cached locally

### Query Execution
1. User submits query (REST API or Streamlit)
2. FastAPI decomposes query using LangChain
3. Hybrid retrieval: semantic search + keyword search
4. Reranking with LLM
5. Context + query sent to NVIDIA LLM
6. Response streamed back
7. Sources and confidence logged
8. Metrics recorded (latency, tokens, cache hit)

### Observability
1. LangSmith captures LLM call traces
2. Prometheus scrapes metrics from `/metrics` endpoint
3. Structured logs written to `logs/app.log`
4. Streamlit dashboard queries Prometheus + logs
5. Alert thresholds configured (e.g., latency > 5s)

## Dependencies (New)

```
# FastAPI & async
fastapi==0.104.0
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# LlamaIndex ecosystem
llama-index==0.9.0
llama-index-llms-nvidia==0.1.0
llama-index-embeddings-nvidia==0.1.0
llama-index-readers-web==0.1.0

# Vector database
chroma-db==0.4.0
chromadb==0.4.0

# LangChain (already present)
langchain>=1.2.17
langchain-core>=1.3.3

# Observability
langsmith==0.1.0
prometheus-client==0.19.0

# Data processing
pandas>=2.0.0
numpy>=2.0

# Utilities
python-dotenv==1.0.0
typing-extensions==4.8.0
```

## Development Workflow

### Phase 1: FastAPI Backend
1. Set up FastAPI with CORS
2. Implement document upload/management routes
3. Add basic query endpoint
4. Integrate with existing vector store

### Phase 2: LlamaIndex Integration
1. Replace manual document loading with LlamaIndex readers
2. Implement adaptive chunking
3. Add reranking logic
4. Support multi-source ingestion

### Phase 3: Observability
1. Add LangSmith tracing
2. Implement Prometheus metrics
3. Set up structured logging
4. Create Streamlit monitoring dashboard

### Phase 4: Production Hardening
1. Error handling & retries
2. Rate limiting
3. Input validation
4. Cache invalidation strategies
5. Docker containerization

## Running the Enhanced System

### Start FastAPI Backend
```bash
cd e:\ai_ws\rag\multimodel_rag
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Streamlit (calls FastAPI)
```bash
streamlit run main_app.py
```

### Access
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090 (if running)

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Indexing | Manual split + in-memory | LlamaIndex adaptive + persistent |
| Retrieval | Single semantic search | Hybrid + reranking + multi-source |
| API Access | Streamlit UI only | REST API + Streamlit |
| Monitoring | Logs only | LangSmith + Prometheus + dashboard |
| Scalability | Single in-memory store | Vector DB abstraction + caching |
| Error Handling | Basic try-catch | Comprehensive + retries |
| Production | Prototype | Ready for deployment |

## Next Steps

1. ✅ Approve architecture
2. Implement FastAPI backend
3. Integrate LlamaIndex
4. Add observability
5. Test with multi-source documents
6. Deploy and monitor

---

**Status**: Ready for implementation  
**Updated**: 2026-06-18  
**Owner**: Development Team
