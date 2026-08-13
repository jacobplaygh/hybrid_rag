import asyncio
from pathlib import Path

from rag.indexing import DocumentIndexer


def test_document_indexing_splits_text_into_chunks(tmp_path):
    # Create a test document with enough content to generate multiple chunks.
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is sentence one. " * 100, encoding="utf-8")

    indexer = DocumentIndexer(
        storage_path=str(tmp_path),
        chunk_size=50,
        chunk_overlap=10,
    )

    documents = indexer.load_documents(str(tmp_path))

    assert len(documents) > 1
    assert all("_chunk_" in doc["doc_id"] for doc in documents)
    assert all(len(doc["content"]) > 0 for doc in documents)
    assert any("_chunk_2" in doc["doc_id"] for doc in documents)


def test_document_metadata_records_chunk_counts(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("This is sentence one. " * 100, encoding="utf-8")

    indexer = DocumentIndexer(
        storage_path=str(tmp_path),
        chunk_size=50,
        chunk_overlap=10,
    )

    indexer.add_documents([str(test_file)])
    documents = indexer.load_documents(str(tmp_path))

    metadata = indexer.get_documents()
    assert len(metadata) == 1
    assert metadata[0]["num_chunks"] == len(documents)
    assert metadata[0]["num_chunks"] > 1
