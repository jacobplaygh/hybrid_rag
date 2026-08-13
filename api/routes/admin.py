"""Admin and control routes."""

from fastapi import APIRouter, HTTPException, status
import logging
from datetime import datetime

from api.config import get_settings
from api.schemas import RebuildIndexRequest, IndexStats
from rag.indexing import DocumentIndexer
from data.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


def get_indexer() -> DocumentIndexer:
    """Create or return the shared document indexer."""
    settings = get_settings()
    return DocumentIndexer(
        settings.UPLOAD_DIR,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )


def get_vector_store_instance():
    """Create or return the shared vector store."""
    settings = get_settings()
    return get_vector_store(
        store_type=settings.VECTOR_STORE_TYPE,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )


@router.post("/rebuild-index", tags=["Admin"])
async def rebuild_index(request: RebuildIndexRequest):
    """Rebuild the vector index from the stored documents."""
    try:
        indexer = get_indexer()
        vector_store = get_vector_store_instance()

        if request.clear_existing:
            indexer.documents_metadata.clear()
            indexer._save_metadata()
            if hasattr(vector_store, "clear"):
                await vector_store.clear()

        documents = indexer.load_documents(indexer.storage_path)
        if documents:
            await vector_store.add_documents(documents)

        stats = await vector_store.get_stats()
        return {
            "status": "success",
            "documents_loaded": len(documents),
            "total_tracked": indexer.get_document_count(),
            "vector_store": stats,
            "include_new_docs": request.include_new_docs,
            "clear_existing": request.clear_existing,
        }
    except Exception as exc:
        logger.error(f"Index rebuild failed: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/clear-cache", tags=["Admin"])
async def clear_cache():
    """Clear the current vector-store cache."""
    try:
        vector_store = get_vector_store_instance()
        cleared = await vector_store.clear() if hasattr(vector_store, "clear") else True
        return {
            "status": "success",
            "cleared": cleared,
            "message": "Vector store cache cleared",
        }
    except Exception as exc:
        logger.error(f"Cache clear failed: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/index-stats", response_model=IndexStats, tags=["Admin"])
async def get_index_stats():
    """Get index statistics."""
    try:
        indexer = get_indexer()
        vector_store = get_vector_store_instance()
        store_stats = await vector_store.get_stats()
        return IndexStats(
            total_documents=indexer.get_document_count(),
            total_chunks=store_stats.get("document_count", 0),
            vector_store_size_mb=0.0,
            last_updated=datetime.now(timezone.utc),
            model_embedding=get_settings().NVIDIA_EMBEDDING_MODEL,
        )
    except Exception as exc:
        logger.error(f"Index stats failed: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
