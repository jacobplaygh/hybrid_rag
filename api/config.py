"""Configuration for FastAPI RAG backend."""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env file if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # API Configuration
    API_TITLE: str = "Hybrid RAG API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "FastAPI backend for advanced RAG with LangChain + LlamaIndex"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # NVIDIA API
    NVIDIA_API_KEY: Optional[str] = os.getenv("NVIDIA_API_KEY")
    NVIDIA_EMBEDDING_MODEL: str = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    NVIDIA_CHAT_MODELS: list = [
        "meta/llama3-8b-instruct",
    ]
    # Allow overriding the default chat model via environment.
    # Prefer DEFAULT_CHAT_MODEL, but fall back to NVIDIA_MODEL for compatibility.
    DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL") or os.getenv("NVIDIA_MODEL") or "meta/llama3-8b-instruct"

    # Separate model for lightweight reranking. Defaults to the main chat model if unset.
    RERANK_MODEL: str = os.getenv("RERANK_MODEL") or os.getenv("DEFAULT_CHAT_MODEL") or os.getenv("NVIDIA_MODEL") or "meta/llama3-8b-instruct"

    # NVIDIA_MODEL is retained for compatibility and used as a fallback.
    NVIDIA_MODEL: Optional[str] = os.getenv("NVIDIA_MODEL")

    def get_default_chat_model(self) -> str:
        """Return the configured default chat model, falling back if needed."""
        configured_model = self.DEFAULT_CHAT_MODEL or self.NVIDIA_MODEL
        if configured_model:
            return configured_model
        return self.NVIDIA_CHAT_MODELS[0]
    
    # Vector Store
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma, pinecone, etc
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "hybrid_rag/data/chroma")
    
    # Document Management
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "hybrid_rag/uploaded_docs")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 100))
    ALLOWED_EXTENSIONS: set = {".txt", ".md", ".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".csv", ".json", ".epub"}
    
    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", 80000))
    
    # Retrieval
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 10))
    RETRIEVAL_RERANK: bool = os.getenv("RETRIEVAL_RERANK", "True").lower() == "true"
    DYNAMIC_ROUTING_ENABLED: bool = os.getenv("DYNAMIC_ROUTING_ENABLED", "True").lower() == "true"
    
    # LangSmith (Observability)
    LANGSMITH_SMITH_API_URL: str = os.getenv("LANGSMITH_API_URL", "https://apac.api.smith.langchain.com")
    LANGSMITH_ENABLED: bool = os.getenv("LANGSMITH_ENABLED", "False").lower() == "true"
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "hybrid-rag")
    
    # Semantic Cache
    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "True").lower() == "true"
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.9"))

    # Agentic Loop
    AGENTIC_LOOP_ENABLED: bool = os.getenv("AGENTIC_LOOP_ENABLED", "True").lower() == "true"
    AGENTIC_MAX_RETRIES: int = int(os.getenv("AGENTIC_MAX_RETRIES", 3))
    AGENTIC_CONFIDENCE_THRESHOLD: float = float(os.getenv("AGENTIC_CONFIDENCE_THRESHOLD", "0.55"))
    AGENTIC_REFORMULATION_STRATEGY: str = os.getenv("AGENTIC_REFORMULATION_STRATEGY", "auto")
    AGENTIC_TIMEOUT_SECONDS: int = int(os.getenv("AGENTIC_TIMEOUT_SECONDS", 30))

    # Corrective RAG (CRAG)
    CRAG_ENABLED: bool = os.getenv("CRAG_ENABLED", "True").lower() == "true"
    CRAG_CONFIDENCE_THRESHOLD: float = float(os.getenv("CRAG_CONFIDENCE_THRESHOLD", "0.55"))
    CRAG_MAX_FALLBACKS: int = int(os.getenv("CRAG_MAX_FALLBACKS", 2))
    CRAG_EXPAND_QUERY: bool = os.getenv("CRAG_EXPAND_QUERY", "True").lower() == "true"
    
    # Auth & Security
    AUTH_REQUIRED: bool = os.getenv("AUTH_REQUIRED", "False").lower() == "true"
    API_KEY: Optional[str] = os.getenv("API_KEY")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))
    
    # CORS Origins - will be set in model_post_init
    CORS_ORIGINS: list = ["*"]
    
    # Prometheus
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", 9090))
    
    def model_post_init(self, __context):
        """Parse CORS_ORIGINS after Pydantic initialization."""
        cors_str = os.getenv("CORS_ORIGINS", "*")
        if cors_str and cors_str != "*":
            self.CORS_ORIGINS = [origin.strip() for origin in cors_str.split(",")]
        else:
            self.CORS_ORIGINS = ["*"]
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
