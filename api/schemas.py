"""Pydantic request/response schemas for FastAPI."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from rag.quality_metrics import QualityMetrics
from rag.confidence_scorer import ConfidenceScore


# Document Schemas
class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    filename: str = Field(..., description="Name of the uploaded file")
    file_type: str = Field(..., description="File type (pdf, txt, etc.)")


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    doc_id: str
    filename: str
    file_type: str
    uploaded_at: datetime
    size_bytes: int
    num_chunks: int
    source: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    total: int
    documents: List[DocumentMetadata]


# Query/Chat Schemas
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., description="User query", min_length=1, max_length=10000)
    mode: str = Field(default="simple", description="RAG mode: simple, multi-turn, decompose, structured")
    model: Optional[str] = Field(None, description="Override default model")
    top_k: int = Field(default=3, description="Number of retrieved docs")
    stream: bool = Field(default=False, description="Stream response chunks")


class RetrievedDocument(BaseModel):
    """A retrieved document chunk."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = {}


class SourceAttribution(BaseModel):
    """Detailed source attribution for a response."""
    doc_id: str
    filename: str
    relevance_score: float
    snippet: str
    method: str  # "semantic", "keyword", or "hybrid"


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    query_id: str
    query: str
    mode: str
    response: str
    retrieved_docs: List[RetrievedDocument] = []
    sources: List[SourceAttribution] = []
    model_used: str
    tokens_used: int
    processing_time_ms: float
    timestamp: datetime
    confidence_score: Optional[ConfidenceScore] = None
    quality_metrics: Optional[QualityMetrics] = None


class ChatMessage(BaseModel):
    """Single message in conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request model for multi-turn chat."""
    message: str = Field(..., description="User message", min_length=1)
    session_id: str = Field(..., description="Session/conversation ID")
    model: Optional[str] = Field(None, description="Override default model")
    stream: bool = Field(default=False, description="Stream response chunks")


class ChatResponse(BaseModel):
    """Response model for multi-turn chat."""
    session_id: str
    message_id: str
    response: str
    retrieved_docs: List[RetrievedDocument] = []
    sources: List[SourceAttribution] = []
    model_used: str
    tokens_used: int
    processing_time_ms: float
    timestamp: datetime
    confidence_score: Optional[ConfidenceScore] = None
    quality_metrics: Optional[QualityMetrics] = None


# Retrieval Diagnostics
class RetrieverDiagnostics(BaseModel):
    """Diagnostics for a query's retrieval."""
    query_id: str
    query: str
    retrieved_count: int
    documents: List[RetrievedDocument]
    scores_distribution: Dict[str, float] = Field(default_factory=dict)
    retrieval_time_ms: float


# Admin/Control
class RebuildIndexRequest(BaseModel):
    """Request to rebuild index."""
    include_new_docs: bool = Field(default=True, description="Include newly uploaded docs")
    clear_existing: bool = Field(default=False, description="Clear existing index")


class IndexStats(BaseModel):
    """Statistics about the index."""
    total_documents: int
    total_chunks: int
    vector_store_size_mb: float
    last_updated: datetime
    model_embedding: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    nvidia_api_available: bool
    vector_store_available: bool
    dependencies: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


# Streaming Response
class StreamChunk(BaseModel):
    """Single chunk in a streamed response."""
    chunk_id: int
    content: str
    is_final: bool = False


# Error Response
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime
