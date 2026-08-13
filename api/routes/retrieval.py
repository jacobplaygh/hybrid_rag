"""Retrieval diagnostics routes."""

import logging
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from api.routes.query import get_rag_system
from api.schemas import RetrieverDiagnostics, RetrievedDocument
from rag.hybrid_rag import HybridRAG
from data.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


def get_rag_instance() -> HybridRAG:
    """Get RAG instance."""
    rag = get_rag_system()
    if rag is None:
        raise ValueError("RAG system not initialized")
    return rag


@router.get("/{query_id}", response_model=RetrieverDiagnostics, tags=["Retrieval"])
async def get_retrieval_diagnostics(query_id: str):
    """Get retrieval diagnostics for a query."""
    try:
        rag = get_rag_instance()
        
        diagnostics = rag.get_query_diagnostics(query_id)
        
        if "error" in diagnostics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=diagnostics["error"],
            )
        
        docs = [
            RetrievedDocument(
                content=doc["content"],
                source=doc["source"],
                score=doc["score"],
                metadata=doc.get("metadata", {}),
            )
            for doc in diagnostics.get("retrieved_docs", [])
        ]

        score_values = [doc.score for doc in docs] if docs else []
        distribution = {
            "min": min(score_values) if score_values else 0,
            "max": max(score_values) if score_values else 0,
            "avg": sum(score_values) / len(score_values) if score_values else 0,
        }

        return RetrieverDiagnostics(
            query_id=query_id,
            query=diagnostics["query"],
            retrieved_count=diagnostics["documents_retrieved"],
            documents=docs,
            scores_distribution=distribution,
            retrieval_time_ms=diagnostics.get("retrieval_time_ms", 0),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval diagnostics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/", tags=["Retrieval"])
async def list_retrievals(limit: int = 10, offset: int = 0):
    """List recent retrieval operations."""
    try:
        rag = get_rag_instance()
        
        # Get query history
        all_queries = list(rag.query_history.values())
        
        # Apply pagination
        retrievals = all_queries[offset:offset + limit]
        
        return {
            "total": len(all_queries),
            "limit": limit,
            "offset": offset,
            "retrievals": retrievals,
        }
    
    except Exception as e:
        logger.error(f"List retrievals error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/test", tags=["Retrieval"])
async def test_retrieval(query: str, top_k: int = 3):
    """Test retrieval for a query."""
    try:
        rag = get_rag_instance()
        
        docs, query_id = await rag.retrieve(query, top_k=top_k)
        
        return {
            "query_id": query_id,
            "query": query,
            "retrieved_count": len(docs),
            "documents": docs,
            "scores_distribution": {
                "min": min([d["score"] for d in docs]) if docs else 0,
                "max": max([d["score"] for d in docs]) if docs else 0,
                "avg": sum([d["score"] for d in docs]) / len(docs) if docs else 0,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Test retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
