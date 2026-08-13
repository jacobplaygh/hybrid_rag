"""LlamaIndex document indexing and management."""

import logging
import json
import zipfile
import re
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone

try:
    from ebooklib import epub
    from html2text import html2text
    HAS_EPUB_LIB = True
except ImportError:
    HAS_EPUB_LIB = False

try:
    from llama_index.core import SimpleDirectoryReader, Document
    from llama_index.core.text_splitter import SentenceSplitter
    HAS_LLAMA_INDEX = True
except ImportError:
    HAS_LLAMA_INDEX = False
    logger = logging.getLogger(__name__)
    logger.warning("LlamaIndex not installed; using fallback document loading")

logger = logging.getLogger(__name__)

# Metadata storage for documents
METADATA_FILE = "document_metadata.json"


class DocumentIndexer:
    """Manages document indexing with LlamaIndex."""
    
    def __init__(self, storage_path: str, chunk_size: int = 512, chunk_overlap: int = 200):
        """Initialize document indexer."""
        self.storage_path = Path(storage_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata store
        self.metadata_path = self.storage_path / METADATA_FILE
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()
        
        # Initialize LlamaIndex if available
        if HAS_LLAMA_INDEX:
            self.text_splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            logger.info(f"✅ LlamaIndex initialized at {storage_path}")
        else:
            logger.warning("⚠️ LlamaIndex not available; using fallback")
        
        logger.info(f"📚 Initializing DocumentIndexer at {storage_path}")
    
    def _load_metadata(self):
        """Load document metadata from storage."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    self.documents_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.documents_metadata = {}
    
    def _save_metadata(self):
        """Save document metadata to storage."""
        try:
            with open(self.metadata_path, "w") as f:
                json.dump(self.documents_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_documents(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load documents from directory using LlamaIndex SimpleDirectoryReader.
        Supports: .pdf, .txt, .md, .pptx, .csv, .json, .png, .jpg, .jpeg, .epub
        """
        docs_dir = Path(directory or self.storage_path)
        logger.info(f"Loading documents from {docs_dir}")
        
        documents = []
        
        if HAS_LLAMA_INDEX:
            try:
                reader = SimpleDirectoryReader(
                    input_dir=str(docs_dir),
                    recursive=True,
                    required_exts=[".pdf", ".txt", ".md", ".pptx", ".csv", ".json", ".png", ".jpg", ".jpeg", ".epub"],
                )
                loaded_docs = reader.load_data()
                
                chunk_counts: Dict[str, int] = {}
                for doc in loaded_docs:
                    source = doc.metadata.get("file_path", "unknown") if doc.metadata else "unknown"
                    if source.endswith(METADATA_FILE):
                        continue

                    content = doc.get_content()
                    if self.text_splitter and content:
                        chunks = self.text_splitter.split_text(content)
                    else:
                        chunks = [content] if content else []

                    original_id = Path(source).stem if source and source != "unknown" else (doc.doc_id or "unknown")
                    chunk_counts[original_id] = chunk_counts.get(original_id, 0) + len(chunks)

                    for idx, chunk in enumerate(chunks, start=1):
                        doc_dict = {
                            "doc_id": f"{doc.doc_id or str(doc)}_chunk_{idx}",
                            "content": chunk,
                            "metadata": dict(doc.metadata) if doc.metadata else {},
                            "source": source,
                        }
                        documents.append(doc_dict)

                for original_id, count in chunk_counts.items():
                    if original_id in self.documents_metadata:
                        self.documents_metadata[original_id]["num_chunks"] = count
                if chunk_counts:
                    self._save_metadata()

                logger.info(f"✅ Loaded {len(documents)} documents with LlamaIndex and chunked into {len(documents)} pieces")
            except Exception as e:
                logger.error(f"Error loading with LlamaIndex: {e}")
                documents = self._load_documents_fallback(docs_dir)
        else:
            documents = self._load_documents_fallback(docs_dir)
        
        return documents

    def load_documents_from_paths(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load and chunk only the specified document files."""
        documents = []
        chunk_counts: Dict[str, int] = {}
        supported_exts = {".pdf", ".txt", ".md", ".pptx", ".csv", ".json", ".png", ".jpg", ".jpeg", ".epub"}

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists() or path.suffix.lower() not in supported_exts:
                logger.warning(f"Skipping unsupported or missing file for ingestion: {file_path}")
                continue

            try:
                if path.suffix.lower() == ".epub":
                    content = self._parse_epub(path)
                else:
                    content = path.read_text(encoding="utf-8", errors="ignore")

                if not content:
                    continue

                chunks = self.text_splitter.split_text(content) if self.text_splitter and content else [content]
                for idx, chunk in enumerate(chunks, start=1):
                    documents.append({
                        "doc_id": f"{path.stem}_chunk_{idx}",
                        "content": chunk,
                        "metadata": {
                            "file_name": path.name,
                            "file_path": str(path),
                        },
                        "source": str(path),
                    })

                chunk_counts[path.stem] = len(chunks)

            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        for original_id, count in chunk_counts.items():
            if original_id in self.documents_metadata:
                self.documents_metadata[original_id]["num_chunks"] = count
        if chunk_counts:
            self._save_metadata()
            logger.info(f"✅ Loaded {len(documents)} uploaded documents and chunked into {len(documents)} pieces")

        return documents

    class _HTMLTextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._texts: List[str] = []

        def handle_data(self, data: str) -> None:
            self._texts.append(data)

        def get_text(self) -> str:
            return " ".join(self._texts).strip()

    def _parse_epub(self, file_path: Path) -> str:
        """Extract plain text from EPUB files with a robust fallback path."""
        contents: List[str] = []

        def _to_plain_text(raw_html: str) -> str:
            if not raw_html:
                return ""
            if hasattr(html2text, "__call__"):
                try:
                    result = html2text(raw_html)
                    return result or ""
                except Exception as html_exc:
                    logger.warning(f"html2text conversion failed for EPUB content: {html_exc}")
            return re.sub(r"<[^>]+>", " ", raw_html)

        try:
            if HAS_EPUB_LIB:
                try:
                    book = epub.read_epub(file_path)
                    for item in book.get_items():
                        if item.get_type() == epub.ITEM_DOCUMENT:
                            try:
                                text = item.get_content().decode("utf-8", errors="ignore")
                                plain_text = _to_plain_text(text)
                                cleaned = re.sub(r"\s+", " ", plain_text).strip()
                                if cleaned:
                                    contents.append(cleaned)
                            except Exception as item_exc:
                                logger.warning(f"Failed to parse EPUB item {item}: {item_exc}")
                                continue
                except Exception as epub_exc:
                    logger.warning(f"Primary EPUB parser failed for {file_path}: {epub_exc}")

            if not contents:
                with zipfile.ZipFile(file_path, "r") as zf:
                    for name in zf.namelist():
                        if name.lower().endswith((".xhtml", ".html", ".htm", ".xml")):
                            try:
                                raw = zf.read(name)
                                text = raw.decode("utf-8", errors="ignore")
                                cleaned = _to_plain_text(text)
                                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                                if cleaned:
                                    contents.append(cleaned)
                            except Exception as zip_exc:
                                logger.warning(f"Failed to extract EPUB zipped resource {name}: {zip_exc}")
                                continue
        except Exception as e:
            logger.warning(f"Failed to parse EPUB {file_path}: {e}")

        return "\n\n".join(contents).strip()

    def _load_documents_fallback(self, directory: Path) -> List[Dict[str, Any]]:
        """Fallback document loading without LlamaIndex."""
        documents = []
        supported_exts = {".pdf", ".txt", ".md", ".pptx", ".csv", ".json", ".png", ".jpg", ".jpeg", ".epub"}
        
        chunk_counts: Dict[str, int] = {}
        for file_path in directory.rglob("*"):
            if file_path.name == METADATA_FILE:
                continue
            if file_path.suffix.lower() in supported_exts:
                try:
                    if file_path.suffix.lower() == ".epub":
                        content = self._parse_epub(file_path)
                    else:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")

                    if not content:
                        continue

                    if self.text_splitter and content:
                        chunks = self.text_splitter.split_text(content)
                    else:
                        chunks = [content]

                    for idx, chunk in enumerate(chunks, start=1):
                        doc_dict = {
                            "doc_id": f"{file_path.stem}_chunk_{idx}",
                            "content": chunk,
                            "metadata": {"file_name": file_path.name},
                            "source": str(file_path),
                        }
                        documents.append(doc_dict)
                        chunk_counts[file_path.stem] = chunk_counts.get(file_path.stem, 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        for original_id, count in chunk_counts.items():
            if original_id in self.documents_metadata:
                self.documents_metadata[original_id]["num_chunks"] = count
        if chunk_counts:
            self._save_metadata()
        
        logger.info(f"✅ Loaded {len(documents)} documents with fallback reader")
        return documents
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to index and track metadata."""
        added = 0
        failed = 0
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    logger.warning(f"File not found: {file_path}")
                    failed += 1
                    continue
                
                doc_id = path.stem
                self.documents_metadata[doc_id] = {
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": str(path),
                    "doc_id": doc_id,
                    "num_chunks": 0,
                }
                added += 1
                logger.info(f"✅ Added document: {path.name}")
            except Exception as e:
                logger.error(f"Failed to add document {file_path}: {e}")
                failed += 1
        
        self._save_metadata()
        return {"added": added, "failed": failed, "total": len(self.documents_metadata)}
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from metadata tracking."""
        if doc_id in self.documents_metadata:
            del self.documents_metadata[doc_id]
            self._save_metadata()
            logger.info(f"✅ Removed document: {doc_id}")
            return True
        logger.warning(f"Document not found: {doc_id}")
        return False
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """List indexed documents with metadata."""
        docs_list = []
        for doc_id, metadata in self.documents_metadata.items():
            docs_list.append({
                "doc_id": doc_id,
                "file_name": metadata.get("file_name"),
                "file_size": metadata.get("file_size"),
                "uploaded_at": metadata.get("uploaded_at"),
                "file_path": metadata.get("file_path"),
                "num_chunks": metadata.get("num_chunks", 0),
            })
        return docs_list
    
    def get_document_count(self) -> int:
        """Get total document count."""
        return len(self.documents_metadata)
    
    def rebuild_index(self, clear_existing: bool = False) -> Dict[str, Any]:
        """Rebuild index with new chunking strategy."""
        if clear_existing:
            self.documents_metadata.clear()
            self._save_metadata()
            logger.info("🗑️ Cleared all document metadata")
        
        docs = self.load_documents()
        return {
            "status": "success",
            "documents_loaded": len(docs),
            "total_tracked": self.get_document_count(),
        }
