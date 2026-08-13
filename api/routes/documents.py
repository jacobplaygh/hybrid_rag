"""Document management routes."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Optional, Union
from datetime import datetime, timezone
from pathlib import Path

from api.config import get_settings
from api.schemas import DocumentMetadata, DocumentListResponse
from rag.indexing import DocumentIndexer
from data.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state (in production, use proper DI or service layer)
_indexer: DocumentIndexer = None
_vector_store = None


def get_indexer() -> DocumentIndexer:
    """Get or initialize document indexer."""
    global _indexer
    if _indexer is None:
        settings = get_settings()
        _indexer = DocumentIndexer(
            settings.UPLOAD_DIR,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
    return _indexer


def get_vector_store_instance():
    """Get or initialize vector store."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = get_vector_store(
            store_type=settings.VECTOR_STORE_TYPE,
            persist_dir=settings.CHROMA_PERSIST_DIR,
        )
    return _vector_store


async def ingest_documents_background(file_paths: list):
    """Background task to ingest uploaded documents into vector store."""
    try:
        indexer = get_indexer()
        vector_store = get_vector_store_instance()
        
        # Load only the specified uploaded files for indexing
        documents = indexer.load_documents_from_paths(file_paths)
        
        if documents:
            # Add to vector store
            await vector_store.add_documents(documents)
            logger.info(f"✅ Ingested {len(documents)} documents to vector store")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")


@router.post("/upload", tags=["Documents"])
async def upload_document(files: Optional[Union[UploadFile, list[UploadFile]]] = File(default=None)):
    """Upload and ingest documents."""
    settings = get_settings()

    if files is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )

    file_list = files if isinstance(files, list) else [files]
    
    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = []
    failed_files = []
    
    try:
        for file in file_list:
            try:
                if not file.filename:
                    failed_files.append({"filename": "<unknown>", "error": "Filename is missing"})
                    continue

                # Validate file extension
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in settings.ALLOWED_EXTENSIONS:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"File type {file_ext} not allowed"
                    })
                    continue
                
                # Check file size
                content = await file.read()
                file_size_mb = len(content) / (1024 * 1024)
                if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
                    failed_files.append({
                        "filename": file.filename,
                        "error": f"File size {file_size_mb:.2f}MB exceeds limit of {settings.MAX_UPLOAD_SIZE_MB}MB"
                    })
                    continue
                
                # Save file
                safe_name = Path(file.filename).name
                file_path = upload_dir / safe_name
                if file_path.exists():
                    file_path = upload_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{safe_name}"

                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Add to indexer metadata
                indexer = get_indexer()
                indexer.add_documents([str(file_path)])
                
                uploaded_files.append({
                    "filename": file.filename,
                    "stored_path": str(file_path),
                    "size_bytes": len(content),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                })
                
                logger.info(f"✅ Uploaded file: {file.filename}")
            
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        # Trigger ingestion for the uploaded files
        if uploaded_files:
            file_paths = [Path(f["stored_path"]) for f in uploaded_files]
            await ingest_documents_background([str(path) for path in file_paths])
        
        return {
            "status": "success",
            "uploaded": len(uploaded_files),
            "failed": len(failed_files),
            "files": uploaded_files,
            "errors": failed_files if failed_files else None,
        }
    
    except Exception as e:
        logger.error(f"Upload operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("", response_model=DocumentListResponse, tags=["Documents"])
@router.get("/", response_model=DocumentListResponse, tags=["Documents"])
@router.get("/list", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all indexed documents."""
    try:
        indexer = get_indexer()
        documents = indexer.get_documents()
        
        # Convert to response format
        doc_list = [
            DocumentMetadata(
                doc_id=doc["doc_id"],
                filename=doc["file_name"],
                file_type=Path(doc["file_name"]).suffix.lower(),
                uploaded_at=datetime.fromisoformat(doc["uploaded_at"]),
                size_bytes=doc.get("file_size", 0),
                num_chunks=doc.get("num_chunks", 0),
                source=doc.get("file_path"),
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            total=len(doc_list),
            documents=doc_list,
        )
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document from the index."""
    try:
        indexer = get_indexer()
        vector_store = get_vector_store_instance()
        
        # Remove from indexer metadata
        success = indexer.remove_document(doc_id)
        
        if success:
            # Remove from vector store
            await vector_store.delete(doc_id)
            
            logger.info(f"✅ Deleted document: {doc_id}")
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": f"Document {doc_id} deleted"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found",
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/stats", tags=["Documents"])
async def get_document_stats():
    """Get document statistics."""
    try:
        indexer = get_indexer()
        vector_store = get_vector_store_instance()
        
        doc_count = indexer.get_document_count()
        store_stats = await vector_store.get_stats()
        
        return {
            "total_documents": doc_count,
            "vector_store": store_stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
