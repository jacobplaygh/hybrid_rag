# Advanced Architecture Deep-Dive

Technical details and implementation specifics for the Hybrid RAG system.

## Hybrid Retrieval Algorithm

### Architecture
```
Query
    ↓
[Semantic Retrieval]  [Keyword Retrieval]
    ↓                       ↓
  Vector              BM25 Search
  Database            (top_k docs)
  (top_k docs)            ↓
    ↓                 Rank by BM25
Rank by              score
Similarity               ↓
    ↓            Return results
Return results         ↓
    ↓          [Combination Phase]
[Combination Phase]     ↓
    ↓          Merge results
    └──────────→ Combine scores
               (alpha blending)
                      ↓
                   Rerank
              (optional LLM/cross-encoder)
                      ↓
              Return top_k results
```

### Score Combination Algorithm

```python
# For each document in combined set:
combined_score = (
    semantic_score * alpha +          # 0 to 1
    keyword_score * (1 - alpha) +     # 0 to 1
    confidence_boost * 0.1            # Small boost for appearing in both
)

# Where alpha ∈ [0, 1]:
# alpha = 0.0 → pure keyword search
# alpha = 0.5 → balanced hybrid
# alpha = 1.0 → pure semantic search
```

## Vector Store Integration

### ChromaDB Implementation

```python
class ChromaVectorStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="hybrid-rag",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def add_documents(self, documents: List[Dict]):
        """Add embeddings to vector store."""
        # Generate embeddings
        # Store with metadata
        # Index automatically
        
    async def search(self, query_embedding: List[float], top_k: int):
        """Search vector store by embedding."""
        # Query similar vectors
        # Return top_k matches
```

### Embedding Strategy

**Model:** `all-MiniLM-L6-v2`
- 384-dimensional vectors
- Fast embedding generation
- Good semantic understanding
- Lightweight (22MB)

**Chunking Strategy:**
- Default chunk size: 256 tokens
- Overlap: 200 tokens (for context)
- Adaptive sizing based on document type

## BM25 Keyword Search

### Implementation

```python
class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def index_documents(self, docs: List[Dict]):
        """Index documents for keyword search."""
        # Tokenize each document
        corpus = [self._tokenize(doc["content"]) for doc in docs]
        # Create BM25 index
        self.bm25 = BM25Okapi(corpus)
    
    def search(self, query: str, top_k: int) -> List[Dict]:
        """Search by keywords."""
        # Tokenize query
        query_tokens = self._tokenize(query)
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        # Return top-k docs
```

### BM25 Parameters
- **k1 = 1.5** - Term frequency saturation
- **b = 0.75** - Length normalization
- **epsilon = 0.25** - Smoothing

## Conversation Memory Management

### In-Memory State

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 50):
        self.conversations = {}  # session_id → list of messages
        self.max_messages = max_messages
    
    async def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation."""
        # Store message
        # Enforce max length
        # Update metadata
    
    async def get_context(self, session_id: str) -> str:
        """Get conversation context for LLM."""
        # Format messages
        # Respect token budget
        # Return as string
```

### Message Format
```json
{
  "session_id": "conv_123",
  "timestamp": "2026-07-20T10:30:00Z",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."},
    ...
  ]
}
```

## LLM Chain Architecture

### Multi-Chain System

```
Input Query
    ↓
[Simple Mode]      [Chat Mode]         [Reasoning Mode]
Query → Retrieve   Chat History        Query → Decompose
    ↓              Query History           ↓
Direct Answer      Retrieve + Context  Multiple Queries
                        ↓              (Parallel)
                   Contextual Answer       ↓
                                      Synthesize Answer
```

### Chain Execution

```python
class HybridRAG:
    async def query(self, query: str, mode: str = "simple"):
        """Execute query based on mode."""
        
        if mode == "simple":
            # 1. Retrieve documents
            docs = await self.retrieve(query)
            # 2. Generate answer
            answer = await self.generate(query, docs)
            return answer
        
        elif mode == "chat":
            # 1. Get conversation context
            context = await self.memory.get_context(session_id)
            # 2. Retrieve documents
            docs = await self.retrieve(query, context)
            # 3. Generate contextual answer
            answer = await self.generate_chat(query, docs, context)
            return answer
        
        elif mode == "reasoning":
            # 1. Decompose query
            sub_questions = await self.decompose(query)
            # 2. Answer sub-questions in parallel
            answers = await asyncio.gather(*[
                self.query(sq, mode="simple")
                for sq in sub_questions
            ])
            # 3. Synthesize final answer
            answer = await self.synthesize(query, answers)
            return answer
```

## Document Processing Pipeline

### Indexing Workflow

```
Raw Document
    ↓
[Format Detection]
    ├─ PDF?    → PDFPlumber
    ├─ EPUB?   → EbookLib
    ├─ Text?   → Raw read
    └─ Other?  → Fallback
    ↓
[Cleaning & Normalization]
    ├─ Remove metadata
    ├─ Fix encoding
    └─ Normalize whitespace
    ↓
[Chunking]
    ├─ Split by size (256 tokens)
    ├─ Overlap (200 tokens)
    └─ Preserve structure
    ↓
[Metadata Extraction]
    ├─ Document title
    ├─ Section headers
    ├─ File info
    └─ Chunk info
    ↓
[Embedding Generation]
    ├─ Generate embeddings
    ├─ Create vectors
    └─ Store metadata
    ↓
[Index Creation]
    ├─ Add to vector store
    ├─ Build BM25 index
    └─ Cache embeddings
    ↓
Ready for Retrieval
```

## Error Handling & Resilience

### Fallback Strategies

```
[Hybrid Search]
    ↓ (fails)
[Semantic Search]
    ↓ (fails)
[Keyword Search]
    ↓ (fails)
"No results found"
    ↓ (return error)
```

### Failure Modes

| Failure | Symptom | Handling |
|---------|---------|----------|
| Vector DB down | Can't query vectors | Fall back to BM25 |
| LLM unavailable | Can't generate | Return doc snippets |
| Embeddings unavailable | Can't embed | Use fallback model |
| Rate limit | Too many requests | Return 429, suggest retry |
| Context too large | Model input exceeded | Summarize or truncate |

## Performance Optimization

### Caching Strategy

```python
# Query result cache
Cache key = hash(query + mode + top_k)
Cache TTL = 3600 seconds  # 1 hour

# If cache hit:
    return cached_result
else:
    result = execute_query()
    cache_result(key, result, ttl)
    return result
```

### Query Optimization

- Pre-compute embeddings for documents
- Cache tokenized queries
- Batch similar queries
- Use async/await throughout
- Parallel retrieval methods

## Token Management

### Token Counting

```python
def count_tokens(text: str) -> int:
    """Count tokens in text."""
    # Use tiktoken for accurate counting
    # Or estimate: ~4 chars per token
    pass

# Model limits:
- NVIDIA GLM: 8000 tokens
- Llama 2 70B: 4096 tokens
```

### Context Budget

```
Total budget = Model limit (8000)
Reserved for response = 1000
Available for context = 7000

Query tokens = estimate_tokens(query)
Context tokens = 7000 - query_tokens
```

## Monitoring & Metrics

### Key Metrics

```
- Queries per second (QPS)
- Query latency (p50, p95, p99)
- Cache hit rate
- Error rate by type
- Token usage per query
- Document retrieval time
- LLM generation time
```

### Prometheus Metrics

```python
# Counters
queries_total = Counter('rag_queries_total', 'Total queries')
errors_total = Counter('rag_errors_total', 'Total errors')

# Histograms
query_duration = Histogram('rag_query_duration_seconds', 'Query time')
retrieval_duration = Histogram('rag_retrieval_duration_seconds', 'Retrieval time')

# Gauges
active_sessions = Gauge('rag_active_sessions', 'Active conversations')
cache_size = Gauge('rag_cache_size_bytes', 'Cache memory usage')
```

---

**See also:**
- [Overview](./overview.md) - System design
- [Integration](./integration.md) - External systems
- [Performance Tests](../development/testing.md) - Benchmarks
