"""Query and chat routes."""

import logging
import os
import sys
import time
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from datetime import datetime, timezone

from api.config import get_settings
from api.schemas import (
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    RetrievedDocument,
    ConstrainedWorkflowRequest,
    ConstrainedAgentRequest,
)
from rag.quality_metrics import QualityMetricsCalculator
from rag.confidence_scorer import ConfidenceScorer
from observability.tracing import get_tracer
from rag.hybrid_rag import HybridRAG
from data.vector_store import get_vector_store
from observability.metrics import stream_duration, stream_first_chunk_latency
from api.services.constrained_workflow import (
    ConstrainedWorkflowAgent,
    ConstrainedWorkflowService,
)
from rag.constrained_tools import ToolExecutionError

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Global RAG instance
_rag_system: HybridRAG = None


def _running_pytest() -> bool:
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None


def get_rag_system() -> HybridRAG:
    """Get or initialize RAG system."""
    global _rag_system
    if _rag_system is None:
        settings = get_settings()
        
        # Initialize vector store
        vector_store = get_vector_store(
            store_type=settings.VECTOR_STORE_TYPE,
            persist_dir=settings.CHROMA_PERSIST_DIR,
        )
        
        # Initialize LLMs only when available and not under pytest
        chat_llm = None
        reasoning_llm = None
        structured_llm = None
        if not _running_pytest() and HAS_LANGCHAIN and settings.NVIDIA_API_KEY:
            chat_model = settings.get_default_chat_model()
            if chat_model != settings.DEFAULT_CHAT_MODEL:
                logger.warning(
                    "Configured DEFAULT_CHAT_MODEL is unsupported for chat; "
                    f"falling back to {chat_model}"
                )
            chat_llm = ChatNVIDIA(model=chat_model)
            reasoning_llm = ChatNVIDIA(model=chat_model)
            structured_llm = ChatNVIDIA(model=chat_model)
        elif _running_pytest():
            logger.info("Detected pytest runtime; using document-only fallback responses")

        tracer = get_tracer(
            enabled=settings.LANGSMITH_ENABLED and bool(settings.LANGSMITH_API_KEY),
            project=settings.LANGSMITH_PROJECT,
        )
        
        # Initialize RAG
        _rag_system = HybridRAG(
            vector_store=vector_store,
            chat_llm=chat_llm,
            reasoning_llm=reasoning_llm,
            structured_llm=structured_llm,
            tracer=tracer,
        )
    
    return _rag_system


def get_constrained_workflow_service() -> ConstrainedWorkflowService:
    """Create a fresh bounded workflow for one caller request."""
    return ConstrainedWorkflowService.from_rag(get_rag_system())


def get_constrained_workflow_agent() -> ConstrainedWorkflowAgent:
    """Create a fresh constrained agent for one caller request."""
    return ConstrainedWorkflowAgent.from_rag(get_rag_system())


@router.post("/", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Execute a RAG query."""
    try:
        rag = get_rag_system()
        
        result = await rag.query(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            model=request.model,
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        
        # Calculate Phase 1 Metrics & Confidence
        docs = result["retrieved_docs"]
        quality = QualityMetricsCalculator.calculate_all(docs)
        scorer = ConfidenceScorer()
        confidence = scorer.score_response(docs, result["response"])
        
        # Convert to response format
        return QueryResponse(
            query_id=result["query_id"],
            query=result["query"],
            mode=result["mode"],
            response=result["response"],
            retrieved_docs=[
                RetrievedDocument(
                    content=doc["content"],
                    source=doc["source"],
                    score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in result["retrieved_docs"]
            ],
            model_used=result["model_used"],
            tokens_used=result.get("tokens_used", 0),
            processing_time_ms=result["query_time_ms"],
            timestamp=datetime.fromisoformat(result["timestamp"]),
            confidence_score=confidence,
            quality_metrics=quality,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/workflow", tags=["Query"])
async def constrained_workflow(request: ConstrainedWorkflowRequest):
    """Run the bounded workflow used by constrained callers and future agents."""
    try:
        service = get_constrained_workflow_service()
        return await service.run(
            query=request.query,
            response=request.response,
            history=request.history,
            top_k=request.top_k,
        )
    except ToolExecutionError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error.as_dict())
    except Exception as error:
        logger.error(f"Constrained workflow endpoint error: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        )


@router.post("/agent", tags=["Query"])
async def constrained_agent(request: ConstrainedAgentRequest):
    """Run an explicitly supported multi-step constrained agent task."""
    try:
        agent = get_constrained_workflow_agent()
        return await agent.run(
            task=request.task,
            query=request.query,
            response=request.response,
            history=request.history,
            top_k=request.top_k,
        )
    except ToolExecutionError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error.as_dict())
    except Exception as error:
        logger.error(f"Constrained agent endpoint error: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        )


@router.post("/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """Stream RAG query response."""
    try:
        rag = get_rag_system()
        generator = await rag.query_stream(request)
        
        async def stream_wrapper():
            stream_start = time.perf_counter()
            first_chunk_recorded = False
            try:
                previous_content = ""
                async for chunk in generator:
                    # Extract content if chunk is an object (e.g. AIMessageChunk)
                    content = chunk
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    
                    if content is None:
                        continue

                    content = str(content)
                    if (
                        previous_content
                        and previous_content[-1].isalnum()
                        and content
                        and content[0].isalnum()
                    ):
                        content = " " + content
                    previous_content += content

                    # Encode content so newlines and SSE delimiters remain part of one event.
                    data = f"data: {json.dumps({'content': content})}\n\n"
                    if not first_chunk_recorded:
                        if stream_first_chunk_latency:
                            stream_first_chunk_latency.observe(time.perf_counter() - stream_start)
                        first_chunk_recorded = True
                    yield data.encode("utf-8")
            except Exception as e:
                logger.error(f"Error during stream generation: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8")
            finally:
                if stream_duration:
                    stream_duration.observe(time.perf_counter() - stream_start)
        
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Stream query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Multi-turn conversation endpoint."""
    try:
        rag = get_rag_system()
        
        result = await rag.chat(
            message=request.message,
            session_id=request.session_id,
            model=request.model,
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        
        # Calculate Phase 1 Metrics & Confidence
        docs = result["retrieved_docs"]
        quality = QualityMetricsCalculator.calculate_all(docs)
        scorer = ConfidenceScorer()
        confidence = scorer.score_response(docs, result["response"])
        
        return ChatResponse(
            session_id=result["session_id"],
            message_id=result["message_id"],
            response=result["response"],
            retrieved_docs=[
                RetrievedDocument(
                    content=doc["content"],
                    source=doc["source"],
                    score=doc["score"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in result["retrieved_docs"]
            ],
            model_used=result["model_used"],
            tokens_used=result.get("tokens_used", 0),
            processing_time_ms=result["chat_time_ms"],
            timestamp=datetime.fromisoformat(result["timestamp"]),
            confidence_score=confidence,
            quality_metrics=quality,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Stream chat response."""
    try:
        rag = get_rag_system()
        
        # Call HybridRAG with stream=True to get the async generator
        generator = rag.chat(
            message=request.message,
            session_id=request.session_id,
            model=request.model,
            stream=True,
        )
        
        async def stream_wrapper():
            async for chunk in generator:
                # If chunk is a string, yield it directly. 
                # If it's a dict/object, yield as JSON.
                if isinstance(chunk, str):
                    content = chunk
                else:
                    content = json.dumps(chunk)
                
                # SSE format: "data: <content>\n\n"
                data = f"data: {content}\n\n"
                yield data.encode("utf-8")
        
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
