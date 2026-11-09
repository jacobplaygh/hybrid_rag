# API integration guide

The examples assume the API is listening at `http://localhost:8000`. See the [quick start](../getting-started/quick-start.md) to start it.

## Standard query

```python
import requests

base_url = "http://localhost:8000"
response = requests.post(
    f"{base_url}/api/query/",
    json={"query": "What does the document say?", "top_k": 3, "mode": "simple"},
    timeout=60,
)
response.raise_for_status()
result = response.json()

print(result["response"])
for document in result["retrieved_docs"]:
    print(document["source"], document["score"])
```

The response schema also includes `query_id`, `mode`, `model_used`, `tokens_used`, `processing_time_ms`, timestamp, confidence, and quality metrics.

## Multi-turn chat

Chat requests use `message` and `session_id` (not `query` and `conversation_id`):

```python
response = requests.post(
    f"{base_url}/api/query/chat",
    json={"message": "What is the main idea?", "session_id": "session-123"},
    timeout=60,
)
response.raise_for_status()
print(response.json()["response"])
```

Reuse the same session ID for follow-up messages. The request can also include an optional `model`.

## Goal-directed agent

The agent endpoint uses the existing model configuration and performs bounded, read-only tool selection:

```python
response = requests.post(
    f"{base_url}/api/query/agent/run",
    json={"query": "Find the document's explanation and supporting sources", "top_k": 5, "max_steps": 5},
    timeout=120,
)
response.raise_for_status()
result = response.json()
print(result["status"], result["response"])
print(result["citations"])
```

`top_k` must be 1–10 and `max_steps` 1–6. See [Agent Architecture](./agent.md) for the returned fields and behavior when no model is configured.

## Bounded workflow endpoints

`POST /api/query/workflow` accepts a `query`, a draft `response`, optional `history`, and `top_k`; it runs search, context selection, and validation. `POST /api/query/agent` accepts the same fields plus `task`, which currently must be `grounded_validation`. These endpoints are distinct from `/api/query/agent/run`.

## Document upload

Upload uses multipart form data with a `files` field. The response reports uploaded files and errors; it does not return the `doc_id` shape shown in some older examples.

```python
with open("notes.txt", "rb") as document:
    response = requests.post(
        f"{base_url}/api/documents/upload",
        files={"files": ("notes.txt", document, "text/plain")},
        timeout=120,
    )
response.raise_for_status()
print(response.json())
```

The accepted extensions and maximum file size are controlled by `ALLOWED_EXTENSIONS` and `MAX_UPLOAD_SIZE_MB` in `api/config.py`.

## Streaming

`POST /api/query/stream` and `POST /api/query/chat/stream` return `text/event-stream`. The query stream emits JSON `data:` events containing a `content` field; clients should parse SSE framing rather than treating each network chunk as a complete response.

## Other useful routes

- `GET /api/documents/` — indexed-document metadata.
- `GET /api/retrieval/` — retrieval diagnostics listing.
- `GET /api/retrieval/{query_id}` — diagnostics for a query.
- `GET /analytics/summary`, `/analytics/latency`, `/analytics/failures` — aggregate analytics.
- `GET /health` and `GET /metrics` — health and Prometheus metrics.
- `GET /docs` — interactive OpenAPI documentation.

When `AUTH_REQUIRED=true`, API routers require the configured API key. Check the FastAPI OpenAPI schema for exact request and response contracts.
