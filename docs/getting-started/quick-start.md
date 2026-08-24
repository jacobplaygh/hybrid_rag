# Quick Start Guide

Get the Hybrid RAG system running in 5 minutes.

## Prerequisites

- Python 3.10+ installed
- Virtual environment activated
- Dependencies installed (`pip install -r requirements_fastapi.txt`)

If you haven't set up yet, see [Installation Guide](./installation.md).

## Step 1: Start the Server (1 min)

```powershell
# Navigate to project root
cd d:\projects\ai_ws\rag

# Activate virtual environment
.\genai\Scripts\Activate.ps1

# Set PYTHONPATH to include the hybrid_rag directory
$env:PYTHONPATH = "d:\projects\ai_ws\rag\hybrid_rag"

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

✅ **Server is running!**

## Step 2: Access the UI (1 min)

Open in your web browser:

- **Interactive API Docs:** http://localhost:8000/docs
- **Web UI:** http://localhost:8000/ui
- **API Root:** http://localhost:8000

### Swagger UI Features
- Try different endpoints interactively
- See request/response schemas
- Test with real data

## Step 3: Upload a Document (1 min)

### Via Swagger UI
1. Open http://localhost:8000/docs
2. Find `POST /api/documents/upload`
3. Click "Try it out"
4. Click "Choose File" and select a text or PDF file
5. Click "Execute"

### Via Command Line
```powershell
# Create a sample document
@"
Machine Learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed.

Key concepts:
- Supervised Learning: Learning from labeled data
- Unsupervised Learning: Finding patterns in unlabeled data
- Reinforcement Learning: Learning through interaction and rewards
"@ | Out-File sample.txt

# Upload the document
curl -X POST "http://localhost:8000/api/documents/upload" `
  -F "files=@sample.txt"
```

Expected response:
```json
{
  "uploaded": 1,
  "documents": [
    {
      "doc_id": "doc_...",
      "filename": "sample.txt",
      "size_bytes": 256,
      "chunks": 1
    }
  ]
}
```

✅ **Document uploaded!**

## Step 4: Run Your First Query (1 min)

### Via Swagger UI
1. Open http://localhost:8000/docs
2. Find `POST /api/query`
3. Click "Try it out"
4. Enter query: `What is machine learning?`
5. Click "Execute"

### Via Command Line
```powershell
$query = @{
    query = "What is machine learning?"
    top_k = 3
    mode = "simple"
} | ConvertTo-Json

curl -X POST "http://localhost:8000/api/query" `
  -Header "Content-Type: application/json" `
  -Body $query
```

### Via Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "What is machine learning?",
        "top_k": 3,
        "mode": "simple"
    }
)

print(response.json())
```

Expected response:
```json
{
  "response": "Machine Learning is a subset of artificial intelligence...",
  "sources": [
    {
      "doc_id": "doc_...",
      "filename": "sample.txt",
      "relevance_score": 0.95,
      "snippet": "Machine Learning is a subset of..."
    }
  ],
  "execution_time_ms": 45.23,
  "query_id": "query_..."
}
```

✅ **Query successful!**

## Step 5: Try Different Query Types (1 min)

### Simple Query (Default)
```json
{
  "query": "What are the key concepts?",
  "top_k": 3,
  "mode": "simple"
}
```

### Chat (Multi-turn)
```json
{
  "query": "What is supervised learning?",
  "conversation_id": "conv_123"
}
```

### Different Retrieval
```json
{
  "query": "machine learning supervised",
  "retrieval_alpha": 0.5  # 0=keyword-only, 1=semantic-only
}
```

## API Endpoints Cheat Sheet

### Documents
```
POST   /api/documents/upload      - Upload documents
GET    /api/documents             - List all documents
GET    /api/documents/{doc_id}    - Get document details
DELETE /api/documents/{doc_id}    - Delete document
```

### Queries
```
POST   /api/query                 - Single query
POST   /api/query/stream          - Streaming response
POST   /api/query/chat            - Multi-turn chat
POST   /api/query/workflow        - Bounded search, context selection, and validation
POST   /api/query/agent           - Explicitly supported constrained agent task
GET    /api/query/{query_id}      - Get previous query
```

### Admin
```
GET    /health                    - System health
POST   /api/admin/rebuild-index   - Rebuild vector index
GET    /api/admin/stats           - System statistics
POST   /api/admin/clear-cache     - Clear cache
```

## Common Workflows

### Upload and Query a Document
```powershell
# 1. Upload
$fileContent = "Your document content here"
$fileContent | Out-File test.txt

curl -X POST "http://localhost:8000/api/documents/upload" `
  -F "files=@test.txt"

# 2. Query
$query = @{ query = "Your question here?" } | ConvertTo-Json

curl -X POST "http://localhost:8000/api/query" `
  -Header "Content-Type: application/json" `
  -Body $query
```

### Multi-turn Conversation
```powershell
# Start conversation
$query1 = @{ 
    query = "What is machine learning?"
    conversation_id = "conv_001"
} | ConvertTo-Json

$response1 = curl -X POST "http://localhost:8000/api/query/chat" `
  -Header "Content-Type: application/json" `
  -Body $query1

# Continue conversation
$query2 = @{ 
    query = "Tell me more about supervised learning"
    conversation_id = "conv_001"  # Same ID
} | ConvertTo-Json

$response2 = curl -X POST "http://localhost:8000/api/query/chat" `
  -Header "Content-Type: application/json" `
  -Body $query2
```

## Testing the Full System

Run the test suite:
```powershell
# All tests
python -m pytest tests/ -v

# Specific tests
python -m pytest tests/test_documents_flow.py -v
python -m pytest tests/test_query_flow.py -v
python -m pytest tests/test_retrieval_performance.py -v
```

Expected: **35/35 tests passing** ✅

## Stopping the Server

```powershell
# Press CTRL+C in the terminal where server is running
```

## Next Steps

- **Explore the API:** Visit http://localhost:8000/docs
- **Upload real documents:** Try PDFs, EPUBs, or PowerPoints
- **Run tests:** See [Testing Guide](../development/testing.md)
- **Understand architecture:** See [Architecture](../architecture/)
- **Make improvements:** See [Improvements](../improvements/)

## Troubleshooting

### Server won't start
```
Address already in use
```
→ Change port: `--port 8001` or kill process using port 8000

### ModuleNotFoundError
```
No module named 'fastapi'
```
→ Run: `pip install -r requirements_fastapi.txt`

### NVIDIA API errors
→ That's OK! The system works without it.  
→ Set `NVIDIA_API_KEY` in `.env` if you have credentials.

### ChromaDB errors
```
Cannot load vector store
```
→ Delete `data/chroma` directory and restart (fresh database)

## Performance Tips

- **First query might be slow** (vector model loading) - subsequent queries are faster
- **Use `top_k=3`** for most queries (good speed/quality balance)
- **Cache is automatic** - repeated queries are faster
- **Use `alpha=0.5`** for best hybrid search results

---

**Congratulations! 🎉** You now have a working Hybrid RAG system!

**Next:** Explore the [API Documentation](http://localhost:8000/docs) or read [Architecture Guide](../architecture/)
