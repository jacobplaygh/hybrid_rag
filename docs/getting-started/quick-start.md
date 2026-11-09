# Quick start

Commands below assume dependencies are installed and the current directory is the workspace directory containing `hybrid_rag`.

## 1. Configure model access (optional)

For NVIDIA-hosted generation, set a key in the current PowerShell session or in a root `.env` file:

```powershell
$env:NVIDIA_API_KEY = "your-key"
```

Without a key, the API can still run and return document-based fallback responses; model-planned agent behavior is not available.

## 2. Start the server

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'hybrid_rag')
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://localhost:8000/docs` for Swagger UI, `http://localhost:8000/ui` for the browser UI, or `http://localhost:8000/health` for health status.

## 3. Upload a document

From another PowerShell terminal:

```powershell
curl.exe -X POST "http://localhost:8000/api/documents/upload" `
  -F "files=@.\notes.txt"
```

The upload endpoint accepts the file extensions configured in `api/config.py`; its response contains `uploaded`, `failed`, `files`, and `errors` fields.

## 4. Ask a question

```powershell
$body = @{
    query = "What does the document say?"
    top_k = 3
    mode = "simple"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/query/" `
  -ContentType "application/json" `
  -Body $body
```

The query response includes `response` and `retrieved_docs`, as well as query metadata and quality/confidence fields.

## 5. Try the goal-directed agent

`POST /api/query/agent/run` is a separate bounded, read-only agent path. It accepts a query, `top_k` from 1–10, and `max_steps` from 1–6. See [Agent Architecture](../architecture/agent.md) for a request example and limits.

The iterative retrieval loop used by the standard query flow is a different feature and is disabled by default. Set `AGENTIC_LOOP_ENABLED=true` to opt in; it remains under evaluation.

## Common endpoints

- `POST /api/documents/upload` — ingest one or more files.
- `GET /api/documents/` — list indexed documents.
- `POST /api/query/` — standard query.
- `POST /api/query/chat` — multi-turn chat (`message`, `session_id`).
- `POST /api/query/stream` — streaming query.
- `POST /api/query/agent/run` — goal-directed agent.
- `GET /analytics/summary` — aggregate query analytics.
- `GET /health` — service health.
