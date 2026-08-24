# Integration Guide

How to integrate the Hybrid RAG system with other applications and platforms.

## REST API Integration

### Basic Integration

```python
import requests

# Your RAG system endpoint
RAG_URL = "http://localhost:8000"

# Example: Query the system
response = requests.post(
    f"{RAG_URL}/api/query",
    json={
        "query": "What is machine learning?",
        "top_k": 3,
        "mode": "simple"
    }
)

result = response.json()
print(f"Answer: {result['response']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {[s['filename'] for s in result['sources']]}")
```

### Constrained Workflow

Use the bounded workflow when a caller has a draft response and needs retrieval, context selection, and validation under the configured tool policy:

```python
response = requests.post(
    f"{RAG_URL}/api/query/workflow",
    json={
        "query": "What framework does the backend use?",
        "response": "The backend uses FastAPI.",
        "top_k": 3,
        "history": None,
    },
)

result = response.json()
print(result["validation"]["is_valid"])
print(result["selection"]["report"])
```

The endpoint returns `400` with a structured `detail` object when a tool policy rejects the request. The object contains `code`, `tool_name`, and `message`.

### Multi-Turn Conversation

```python
import requests

RAG_URL = "http://localhost:8000"
conversation_id = "user_123_session_1"

# First question
response1 = requests.post(
    f"{RAG_URL}/api/query/chat",
    json={
        "query": "What is neural networks?",
        "conversation_id": conversation_id
    }
)

# Follow-up question (context preserved)
response2 = requests.post(
    f"{RAG_URL}/api/query/chat",
    json={
        "query": "How do they learn?",
        "conversation_id": conversation_id
    }
)
```

### Document Upload

```python
import requests

RAG_URL = "http://localhost:8000"

# Upload a PDF
with open("research_paper.pdf", "rb") as f:
    response = requests.post(
        f"{RAG_URL}/api/documents/upload",
        files={"files": f}
    )

result = response.json()
doc_id = result['documents'][0]['doc_id']
print(f"Uploaded with ID: {doc_id}")
```

## Streaming Responses

For long responses, use streaming:

```python
import requests

RAG_URL = "http://localhost:8000"

response = requests.post(
    f"{RAG_URL}/api/query/stream",
    json={"query": "Tell me about machine learning"},
    stream=True
)

# Print tokens as they arrive
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'), end='', flush=True)
```

## Webhook Integration

Call the RAG system and notify another service when done:

```python
import requests
import asyncio

async def query_with_webhook(query: str, webhook_url: str):
    """Query RAG and send result to webhook."""
    
    # Query
    response = requests.post(
        "http://localhost:8000/api/query",
        json={"query": query}
    )
    
    result = response.json()
    
    # Notify webhook
    requests.post(
        webhook_url,
        json={
            "query": query,
            "response": result['response'],
            "confidence": result['confidence'],
            "sources": result['sources']
        }
    )
```

## Slack Integration

```python
from slack_sdk import WebClient
import requests

slack_client = WebClient(token="xoxb-your-token")
RAG_URL = "http://localhost:8000"

def handle_slack_message(query: str, channel: str):
    """Handle queries from Slack."""
    
    # Query RAG
    response = requests.post(
        f"{RAG_URL}/api/query",
        json={"query": query}
    )
    
    result = response.json()
    
    # Format and send to Slack
    message = f"""
*Question:* {query}

*Answer:* {result['response']}

*Confidence:* {result['confidence']:.0%}

*Sources:* {', '.join([s['filename'] for s in result['sources']])}
    """
    
    slack_client.chat_postMessage(
        channel=channel,
        text=message
    )
```

## Discord Bot Integration

```python
import discord
import requests

RAG_URL = "http://localhost:8000"

class RAGBot(discord.Client):
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        if message.content.startswith("!ask "):
            query = message.content[5:]  # Remove "!ask "
            
            # Show typing indicator
            async with message.channel.typing():
                # Query RAG
                response = requests.post(
                    f"{RAG_URL}/api/query",
                    json={"query": query}
                )
                
                result = response.json()
                
                # Send response
                embed = discord.Embed(
                    title=query,
                    description=result['response'],
                    color=discord.Color.blue()
                )
                embed.add_field(
                    name="Confidence",
                    value=f"{result['confidence']:.0%}"
                )
                
                await message.reply(embed=embed)
```

## API Gateway Integration (AWS, Kong, etc.)

### AWS API Gateway

```yaml
# SAM template
Resources:
  RAGApiGateway:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: hybrid-rag-api
      Target: http://rag-server:8000
      
  QueryResource:
    Type: AWS::ApiGateway::Resource
    Properties:
      ParentId: !GetAtt RAGApiGateway.RootResourceId
      PathPart: query
  
  QueryMethod:
    Type: AWS::ApiGateway::Method
    Properties:
      ResourceId: !Ref QueryResource
      HttpMethod: POST
      Integration:
        Type: HTTP
        Uri: http://rag-server:8000/api/query
```

### Kong Integration

```yaml
# Kong service configuration
services:
- name: hybrid-rag
  host: rag-server
  port: 8000
  routes:
  - name: rag-query
    paths:
    - /api/query
  - name: rag-documents
    paths:
    - /api/documents
  plugins:
  - name: rate-limiting
    config:
      minute: 100
      hour: 1000
  - name: auth
    config:
      key_names: 
      - "X-API-Key"
```

## Search Engine Integration

### Embed in Search Results

```python
# Example: Custom search engine augmentation
def augmented_search(query: str):
    """Search with RAG enhancement."""
    
    # Get basic search results
    search_results = search_engine.search(query)
    
    # Get RAG answer
    rag_response = requests.post(
        "http://localhost:8000/api/query",
        json={"query": query}
    )
    
    rag_answer = rag_response.json()
    
    # Combine results
    return {
        "rag_answer": rag_answer['response'],
        "rag_confidence": rag_answer['confidence'],
        "search_results": search_results,
        "sources": rag_answer['sources']
    }
```

## Database Integration

### Store Query Results

```python
import sqlite3
import requests

RAG_URL = "http://localhost:8000"
DB_PATH = "rag_queries.db"

def query_and_store(query: str):
    """Query RAG and store result in database."""
    
    # Query RAG
    response = requests.post(
        f"{RAG_URL}/api/query",
        json={"query": query}
    )
    
    result = response.json()
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO rag_queries 
        (query, response, confidence, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    """, (query, result['response'], result['confidence']))
    
    conn.commit()
    conn.close()
    
    return result
```

## Analytics Integration

### Send Metrics to Analytics Platform

```python
import requests
from analytics import analytics

RAG_URL = "http://localhost:8000"

def track_query(query: str, user_id: str):
    """Query RAG and track analytics."""
    
    # Query
    response = requests.post(
        f"{RAG_URL}/api/query",
        json={"query": query}
    )
    
    result = response.json()
    
    # Track analytics
    analytics.track(user_id, "RAG Query", {
        "query": query,
        "response_quality": result['confidence'],
        "num_sources": len(result['sources']),
        "execution_time_ms": result['execution_time_ms'],
        "response_length": len(result['response'])
    })
    
    return result
```

## Data Pipeline Integration

### ETL Integration

```python
import requests

RAG_URL = "http://localhost:8000"

class RAGEnricher:
    """Enrich data with RAG answers."""
    
    async def enrich_records(self, records):
        """Add RAG answers to records."""
        
        enriched = []
        
        for record in records:
            # Query RAG
            response = requests.post(
                f"{RAG_URL}/api/query",
                json={"query": record['description']}
            )
            
            result = response.json()
            
            # Enrich record
            enriched_record = {
                **record,
                "rag_answer": result['response'],
                "rag_confidence": result['confidence'],
                "rag_sources": result['sources']
            }
            
            enriched.append(enriched_record)
        
        return enriched
```

## Error Handling & Resilience

### Retry Logic

```python
import requests
import time

def query_with_retry(query: str, max_retries: int = 3):
    """Query with automatic retry."""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/api/query",
                json={"query": query},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Health Checks

```python
import requests

def check_rag_health():
    """Check if RAG system is healthy."""
    
    try:
        response = requests.get(
            "http://localhost:8000/health",
            timeout=5
        )
        
        if response.status_code == 200:
            health = response.json()
            return {
                "healthy": all([
                    health.get("vector_store"),
                    health.get("file_system")
                ]),
                "details": health
            }
        else:
            return {"healthy": False, "error": "HTTP error"}
    
    except requests.exceptions.RequestException:
        return {"healthy": False, "error": "Connection failed"}
```

## Security Considerations

### API Key Management

```python
import requests
import os

RAG_API_KEY = os.getenv("RAG_API_KEY")

response = requests.post(
    "http://localhost:8000/api/query",
    json={"query": "..."},
    headers={"X-API-Key": RAG_API_KEY}
)
```

### Input Validation

```python
def safe_query(query: str) -> dict:
    """Validate and sanitize query before sending."""
    
    # Validate length
    if len(query) > 10000:
        raise ValueError("Query too long")
    
    # Remove potentially harmful characters
    sanitized = query.replace("\x00", "")
    
    # Send to RAG
    return requests.post(
        "http://localhost:8000/api/query",
        json={"query": sanitized}
    ).json()
```

---

**See also:**
- [Quick Start](../getting-started/quick-start.md) - Basic usage
- [Architecture](./overview.md) - System design
- [API Docs](http://localhost:8000/docs) - Interactive API reference
