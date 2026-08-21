import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from rag.hybrid_rag import HybridRAG
from rag.retrieval import HybridRetriever, RetrievedDoc


def test_build_context_text_truncates_to_token_budget():
    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    docs = [
        SimpleNamespace(source="doc1.txt", content="alpha " * 400),
        SimpleNamespace(source="doc2.txt", content="beta " * 400),
    ]

    context_text = rag._build_context_text(docs, max_tokens=400)

    assert context_text
    assert rag._estimate_tokens(context_text, "", None) <= 400


def test_rerank_falls_back_to_lexical_scoring_when_model_unavailable():
    retriever = HybridRetriever(vector_store=object(), use_reranker=False, reranker=None)
    results = [
        RetrievedDoc(doc_id="doc-b", content="gamma delta epsilon", source="b.txt", score=0.9),
        RetrievedDoc(doc_id="doc-a", content="alpha beta", source="a.txt", score=0.1),
    ]

    ranked = asyncio.run(retriever.rerank_results("alpha", results, top_k=2))

    assert [doc.doc_id for doc in ranked] == ["doc-b", "doc-a"]


def test_simple_rag_retries_once_on_transient_gateway_timeout():
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def __call__(self, payload):
            self.calls += 1
            if self.calls == 1:
                raise Exception("[504] Gateway Timeout")
            return "recovered response"

    from rag.chains import SimpleRAGChain

    chain = SimpleRAGChain(FlakyLLM())
    response = asyncio.run(chain.invoke("what happened", "context"))

    assert response == "recovered response"


def test_stream_response_flattens_nested_async_generators():
    async def nested_stream():
        yield "hello"
        yield " world"

    async def response_stream():
        yield nested_stream()
        yield "!"

    async def collect_response():
        rag = HybridRAG.__new__(HybridRAG)
        metadata = {}
        chunks = [chunk async for chunk in rag._stream_response(response_stream(), metadata)]
        return chunks, metadata

    chunks, metadata = asyncio.run(collect_response())

    assert chunks == ["hello", " world", "!"]
    assert metadata["response"] == "hello world!"


def test_chat_falls_back_to_document_response_when_llm_raises():
    class FailingChain:
        async def invoke(self, **kwargs):
            raise RuntimeError("boom")

    async def fake_retrieve(*args, **kwargs):
        doc = SimpleNamespace(content="document content", source="doc.txt")
        doc.to_dict = lambda: {"content": doc.content, "source": doc.source}
        return [doc], "retrieval-id"

    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    rag.chains = {"multi_turn": FailingChain()}
    rag.retriever = SimpleNamespace(
        bm25=None,
        retrieve=fake_retrieve,
        get_retrieval_diagnostics=lambda _id: {"retrieval_time_ms": 0},
    )
    rag._select_llm = lambda *args, **kwargs: object()

    result = asyncio.run(rag.chat("hello", "session-1"))

    assert result["response"].startswith("I found relevant information")


def test_chat_stream_passes_history_and_updates_memory():
    class StreamingChain:
        def __init__(self):
            self.arguments = None

        async def astream(self, **kwargs):
            self.arguments = kwargs
            yield "streamed"
            yield " response"

    async def fake_retrieve(*args, **kwargs):
        doc = SimpleNamespace(content="document context", source="doc.txt", score=1.0)
        doc.to_dict = lambda: {
            "content": doc.content,
            "source": doc.source,
            "score": doc.score,
        }
        return [doc], "retrieval-id"

    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    chain = StreamingChain()
    rag.chains = {"multi_turn": chain}
    rag.retriever = SimpleNamespace(
        bm25=object(),
        retrieve=fake_retrieve,
        get_retrieval_diagnostics=lambda _id: {"retrieval_time_ms": 0},
    )
    rag._select_llm = lambda *args, **kwargs: object()
    rag.memory.add_message("session-1", "user", "Earlier question")
    rag.memory.add_message("session-1", "assistant", "Earlier answer")

    async def collect_response():
        generator = await rag.chat("Follow-up question", "session-1", stream=True)
        return [chunk async for chunk in generator]

    chunks = asyncio.run(collect_response())

    assert chunks == ["streamed", " response"]
    assert chain.arguments == {
        "query": "Follow-up question",
        "history": "USER: Earlier question\nASSISTANT: Earlier answer\nUSER: Follow-up question",
        "context": "[doc.txt]\ndocument context",
    }
    assert "ASSISTANT: streamed response" in rag.memory.get_context("session-1")


def test_multi_turn_chat_uses_session_history():
    class InvokingChain:
        def __init__(self):
            self.arguments = None

        async def invoke(self, **kwargs):
            self.arguments = kwargs
            return "new answer"

    rag = HybridRAG.__new__(HybridRAG)
    rag.memory = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None).memory
    chain = InvokingChain()
    rag.chains = {"multi_turn": chain}
    rag.memory.add_message("session-2", "user", "Previous question")
    rag.memory.add_message("session-2", "assistant", "Previous answer")

    response = asyncio.run(rag._multi_turn_chat("New question", "context", object(), "session-2"))

    assert response == "new answer"
    assert chain.arguments == {
        "query": "New question",
        "history": "USER: Previous question\nASSISTANT: Previous answer",
        "context": "context",
    }
    assert "ASSISTANT: new answer" in rag.memory.get_context("session-2")


def test_memory_selects_relevant_older_turns_and_recent_turns():
    from rag.memory import ConversationMemory

    memory = ConversationMemory(max_context_tokens=100)
    memory.add_message("session-3", "user", "We discussed engine performance")
    memory.add_message("session-3", "assistant", "The engine reaches 90 percent efficiency")
    memory.add_message("session-3", "user", "Unrelated deployment question")
    memory.add_message("session-3", "assistant", "The service runs on FastAPI")

    context = memory.get_relevant_context("session-3", query="How is engine efficiency measured?")

    assert "engine performance" in context
    assert "90 percent efficiency" in context
    assert "Unrelated deployment question" in context
    assert "service runs on FastAPI" in context


def test_memory_relevant_context_respects_word_budget():
    from rag.memory import ConversationMemory

    memory = ConversationMemory(max_context_tokens=5)
    memory.add_message("session-4", "user", "first old message about engines")
    memory.add_message("session-4", "assistant", "first old answer")
    memory.add_message("session-4", "user", "latest question")

    context = memory.get_relevant_context("session-4", query="engines")

    assert len(context.split()) <= 5
    assert "latest question" in context


def test_memory_compresses_older_turns_and_preserves_recent_turns():
    from rag.memory import ConversationMemory

    memory = ConversationMemory(max_context_tokens=16)
    memory.add_message("session-5", "user", "The engine uses a turbo cooling system")
    memory.add_message("session-5", "assistant", "The cooling system reduces heat")
    memory.add_message("session-5", "user", "What is the current status?")
    memory.add_message("session-5", "assistant", "The current status is stable")

    context = memory.get_relevant_context("session-5", query="engine cooling")

    assert "Earlier conversation summary:" in context
    assert "USER: What is the current status?" in context
    assert "ASSISTANT: The current status is stable" in context
    assert len(context.split()) <= 16


def test_query_returns_answer_from_uploaded_documents():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)

    upload_response = client.post(
        "/api/documents/upload",
        files={"files": ("sample.txt", b"The backend uses FastAPI for the API layer.", "text/plain")},
        headers={"X-API-Key": "test-key"},
    )
    assert upload_response.status_code == 200

    query_response = client.post(
        "/api/query/",
        json={"query": "What framework does the backend use?", "mode": "simple", "top_k": 3},
        headers={"X-API-Key": "test-key"},
    )

    assert query_response.status_code == 200
    body = query_response.json()
    assert body["query"] == "What framework does the backend use?"
    assert body["response"].strip() != ""
    assert any(keyword in body["response"] for keyword in ["FastAPI", "backend", "document"])


def test_simple_query_cache_hits_on_repeat():
    upload_dir = Path("./uploaded_docs")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)

    upload_response = client.post(
        "/api/documents/upload",
        files={"files": ("sample.txt", b"FastAPI is used for the backend.", "text/plain")},
        headers={"X-API-Key": "test-key"},
    )
    assert upload_response.status_code == 200

    first_response = client.post(
        "/api/query/",
        json={"query": "What framework powers the backend?", "mode": "simple", "top_k": 3},
        headers={"X-API-Key": "test-key"},
    )
    second_response = client.post(
        "/api/query/",
        json={"query": "What framework powers the backend?", "mode": "simple", "top_k": 3},
        headers={"X-API-Key": "test-key"},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_body = first_response.json()
    second_body = second_response.json()

    assert first_body["response"] == second_body["response"]
    assert first_body["processing_time_ms"] == second_body["processing_time_ms"]
    assert first_body["tokens_used"] == second_body["tokens_used"]
    assert first_body["model_used"] == second_body["model_used"]
