from rag.graph_rag import KnowledgeGraph


def test_knowledge_graph_extracts_entities_and_edges():
    graph = KnowledgeGraph()
    graph.index_documents([
        {
            "id": "doc-1",
            "content": "FastAPI integrates with LlamaIndex and uses a vector store for retrieval.",
        }
    ])

    assert "FastAPI" in graph.nodes
    assert "LlamaIndex" in graph.nodes
    assert "vector store" in graph.nodes or "vector" in graph.nodes
    assert "FastAPI" in graph.nodes["LlamaIndex"].neighbors or "LlamaIndex" in graph.nodes["FastAPI"].neighbors


def test_graph_rag_returns_entity_neighbors_for_relational_queries():
    graph = KnowledgeGraph()
    graph.index_documents([
        {
            "id": "doc-1",
            "content": "FastAPI uses a vector store and integrates with LlamaIndex for indexing.",
        },
        {
            "id": "doc-2",
            "content": "The semantic cache helps optimize repeated queries and works with the reranker.",
        },
    ])

    results = graph.retrieve("How does FastAPI relate to LlamaIndex?", top_k=5)

    assert results
    assert any(item["entity"] == "FastAPI" for item in results) or any(item["entity"] == "LlamaIndex" for item in results)
    assert any("FastAPI" in item["content"] or "LlamaIndex" in item["content"] for item in results)
