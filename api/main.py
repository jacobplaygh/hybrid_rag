"""
FastAPI application for Hybrid RAG system.
Combines LangChain orchestration with LlamaIndex advanced indexing.
"""

import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
import sys

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.security import APIKeyHeader
    from prometheus_client import make_asgi_app
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    HTTPException = None
    Request = None
    status = None
    CORSMiddleware = None
    JSONResponse = None
    uvicorn = None

from api.config import get_settings
from api.schemas import HealthResponse, ErrorResponse
from api.exceptions import RAGException
from data.vector_store import get_vector_store
from rag.hybrid_rag import HybridRAG
from observability.tracing import LangSmithTracer
from observability.logging_config import setup_logging
import os
import uuid

# Configure logging
setup_logging(debug=get_settings().DEBUG)
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_request_history = defaultdict(deque)


async def require_api_key(request: Request, api_key: str | None = Depends(api_key_header)):
    """Require an API key for protected routes when authentication is enabled."""
    settings = get_settings()
    if not settings.AUTH_REQUIRED:
        return None

    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
        )

    return None


# ===================== Lifespan Context Manager =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Hybrid RAG API...")
    settings = get_settings()

# Reset ephemeral runtime state on each app startup
    app.state.request_log = []
    _request_history.clear()
    
    if not settings.NVIDIA_API_KEY:
        logger.warning("⚠️ NVIDIA_API_KEY not set in environment")
    
    # Create upload directory if not exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Initialize LangSmith Tracer
    app.state.tracer = LangSmithTracer(
        enabled=settings.LANGSMITH_ENABLED,
        project_name=settings.LANGSMITH_PROJECT
    )
    logger.info(f"LangSmith tracer initialized (enabled={app.state.tracer.enabled})")
    
    logger.info("✅ Startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Hybrid RAG API...")
    logger.info("✅ Shutdown complete")


# ===================== FastAPI App Setup =====================

settings = get_settings()

if not HAS_FASTAPI:
    raise ImportError("FastAPI is required to run the backend. Install the dependencies from requirements_fastapi.txt")

app = FastAPI(
    title="Hybrid RAG API",
    description="Advanced Hybrid RAG system with semantic caching and observability",
    version="1.0.0",
    lifespan=lifespan,
)

# Initialize state attributes that must be present even before lifespan starts
app.state.request_log = []
app.state.rag = None # Will be initialized in lifespan or routes if needed

# Initialize Observability
setup_logging()
tracer = LangSmithTracer()

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware to log request metadata to app.state.request_log."""
    start_time = datetime.now(timezone.utc)
    response = await call_next(request)
    process_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    app.state.request_log.append({
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "process_time": process_time,
        "timestamp": start_time.isoformat(),
    })
    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce request rate limits based on client IP."""
    settings = get_settings()
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc)
    
    # Clean up old requests
    window_start = now - timedelta(seconds=settings.RATE_LIMIT_WINDOW_SECONDS)
    while _request_history[client_ip] and _request_history[client_ip][0] < window_start:
        _request_history[client_ip].popleft()
    
    if len(_request_history[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum of {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS} seconds allowed.",
                "status_code": 429,
                "timestamp": now.isoformat(),
            }
        )
    
    _request_history[client_ip].append(now)
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Global Exception Handler =====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None,
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Custom RAG exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "status": "error"
        }
    )


# ===================== Health & Info Endpoints =====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with dependency status details."""
    settings = get_settings()

    try:
        vector_store = get_vector_store(
            store_type=settings.VECTOR_STORE_TYPE,
            persist_dir=settings.CHROMA_PERSIST_DIR,
        )
        stats = await vector_store.get_stats()
        vector_store_available = True
    except Exception as exc:
        logger.warning(f"Vector store health check failed: {exc}")
        stats = {"error": str(exc)}
        vector_store_available = False

    return HealthResponse(
        status="healthy" if vector_store_available else "degraded",
        version=settings.API_VERSION,
        nvidia_api_available=bool(settings.NVIDIA_API_KEY),
        vector_store_available=vector_store_available,
        dependencies={
            "vector_store": {
                "available": vector_store_available,
                "type": settings.VECTOR_STORE_TYPE,
                "stats": stats,
            },
            "nvidia_api": {
                "available": bool(settings.NVIDIA_API_KEY),
                "configured": bool(settings.NVIDIA_API_KEY),
            },
        },
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/", tags=["Info"])
async def root():
    """API root endpoint with basic runtime status information."""
    return {
        "message": "Hybrid RAG API",
        "status": "running",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "ui": "/ui",
        "openapi_schema": "/openapi.json",
        "health": "/health",
    }


@app.get("/health")
async def health_check_simple():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Retrieve a summary of query patterns and system performance."""
    try:
        # Access the rag instance from the app state
        rag = app.state.rag
        summary = rag.analytics.analyze_patterns()
        return summary
    except Exception as e:
        logger.error(f"Failed to retrieve analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving analytics")

@app.get("/analytics/failures")
async def get_analytics_failures(limit: int = 10):
    """Retrieve a list of failed or low-confidence queries."""
    try:
        rag = app.state.rag
        failures = rag.analytics.get_failed_queries(limit=limit)
        return {"failures": failures}
    except Exception as e:
        logger.error(f"Failed to retrieve analytics failures: {e}")
        raise HTTPException(status_code=500, detail="Internal server error retrieving failures")


# ===================== Import Routes =====================

# Routes will be imported here and included
from api.routes import documents, query, retrieval, admin
from api.ui import router as ui_router

# Include UI router before API prefixes so /ui is available directly
app.include_router(ui_router)

# Include API routers with prefixes
app.include_router(
    documents.router,
    prefix="/api/documents",
    tags=["Documents"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    query.router,
    prefix="/api/query",
    tags=["Query"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    retrieval.router,
    prefix="/api/retrieval",
    tags=["Retrieval"],
    dependencies=[Depends(require_api_key)],
)
app.include_router(
    admin.router,
    prefix="/api/admin",
    tags=["Admin"],
    dependencies=[Depends(require_api_key)],
)


# ===================== Startup Message =====================

logger.info(f"""
╔═══════════════════════════════════════════════════════════════╗
║         Hybrid RAG API - FastAPI Backend                      ║
║         Version: {settings.API_VERSION}                       ║
║         Debug: {settings.DEBUG}                               ║
║         Upload Dir: {settings.UPLOAD_DIR}                     ║
║         Vector Store: {settings.VECTOR_STORE_TYPE}            ║
╚═══════════════════════════════════════════════════════════════╝
""")


# ===================== Main =====================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
