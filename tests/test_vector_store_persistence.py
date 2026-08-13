import asyncio

from data.vector_store import InMemoryVectorStore


def test_in_memory_store_persists_across_instances(tmp_path):
    store_path = tmp_path / "persisted-store"
    store_path.mkdir(parents=True, exist_ok=True)

    store = InMemoryVectorStore(storage_path=str(store_path))
    asyncio.run(store.add_documents([
        {
            "doc_id": "doc-1",
            "content": "Persistent content",
            "metadata": {"source": "test.txt"},
        }
    ]))

    reloaded_store = InMemoryVectorStore(storage_path=str(store_path))
    docs = reloaded_store.documents

    assert "doc-1" in docs
    assert docs["doc-1"]["content"] == "Persistent content"
