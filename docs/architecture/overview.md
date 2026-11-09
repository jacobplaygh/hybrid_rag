# Architecture overview

## Runtime shape

The project is a FastAPI application with a `HybridRAG` orchestrator. The API exposes document ingestion, retrieval, query, chat, streaming, administration, analytics, a browser UI, and Prometheus metrics. NVIDIA model access is optional; without a configured key, the ordinary query path can return document-based fallback responses.

```text
Client / browser
  ├─ FastAPI routes (`api/`)
  ├─ Query and chat orchestration (`rag/hybrid_rag.py`)
  │    ├─ hybrid retrieval (`rag/retrieval.py`)
  │    ├─ query understanding and routing
  │    ├─ optional reranking and iterative retrieval
  │    ├─ GraphRAG, context selection, and response validation
  │    └─ cache, conversation memory, and analytics
  ├─ vector store (`data/vector_store.py`)
  └─ optional NVIDIA chat/embedding services
```

The vector-store setting selects the configured backend. Chroma is the normal persistent backend; an in-memory backend is available for tests and development. The knowledge graph is a separate lightweight in-process index, not a replacement for the document vector store.

## Main request flows

### Query

`POST /api/query/` sends the query to `HybridRAG`, retrieves and ranks document chunks, applies configured context selection, and generates a response when an LLM is available. The response includes retrieved documents, timing, model, and quality/confidence metadata.

The standard retrieval path is the default. The separate iterative agentic retrieval loop can retry retrieval based on context sufficiency, but `AGENTIC_LOOP_ENABLED` defaults to `false` because its recorded rollout evaluation regressed baseline success rate, context relevance, and groundedness. It remains an explicit configuration option.

### Chat and streaming

`POST /api/query/chat` accepts a message and session ID for conversational context. Query and chat streaming endpoints return server-sent events.

### Goal-directed agent

`POST /api/query/agent/run` is a distinct, bounded, read-only agent path. It can select document search, graph search, or aggregate analytics tools, and returns steps, retrieved evidence, citations, status, and optional validation. See [Agent Architecture](./agent.md) for limits and request/response details.

## Useful endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | API service information |
| `GET /health` | Health and dependency status |
| `GET /docs` | OpenAPI / Swagger UI |
| `GET /ui` | Browser UI |
| `POST /api/documents/upload` | Upload and ingest documents |
| `GET /api/documents/` | List indexed documents |
| `POST /api/query/` | Standard RAG query |
| `POST /api/query/workflow` | Bounded search, context selection, and validation |
| `POST /api/query/agent` | Constrained `grounded_validation` task |
| `POST /api/query/agent/run` | Goal-directed RAG agent |
| `POST /api/query/chat` | Multi-turn chat |
| `GET /analytics/summary` | Query-pattern summary |
| `GET /analytics/latency` | Stage latency analysis |
| `GET /analytics/failures` | Failed or low-confidence query records |
| `GET /metrics` | Prometheus metrics |

API-key enforcement is controlled by `AUTH_REQUIRED`; when enabled, the configured API key is required for the API routers.
