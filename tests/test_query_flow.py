import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.config import get_settings
from rag.hybrid_rag import HybridRAG
from rag.retrieval import HybridRetriever, RetrievedDoc
from rag.retrieval_router import RetrievalRouter
from api.routes import query as query_routes
from api.schemas import ConstrainedWorkflowRequest
from rag.constrained_tools import ToolExecutionError


def test_dynamic_routing_selects_keyword_strategy_for_identifier_queries():
    router = RetrievalRouter(enabled=True)

    route = router.route("Find part number A-100 and the configuration guide", {"intent": "FACTUAL", "entities": ["A-100"]})

    assert route.strategy == "keyword"
    assert route.alpha == 0.0


def test_dynamic_routing_selects_hybrid_strategy_for_comparisons():
    router = RetrievalRouter(enabled=True)

    route = router.route("Compare the H100 versus A100 for training performance", {"intent": "COMPARATIVE", "entities": ["H100", "A100"]})

    assert route.strategy == "hybrid"
    assert route.alpha > 0.5


def test_dynamic_routing_uses_hybrid_for_dependency_queries_even_with_entities():
    router = RetrievalRouter(enabled=True)

    route = router.route(
        "How does FastAPI depend on Pydantic for request validation?",
        {"intent": "FACTUAL", "entities": ["FastAPI", "Pydantic"]},
    )

    assert route.strategy == "hybrid"
    assert route.alpha >= 0.75


def test_build_context_text_truncates_to_token_budget():
    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    docs = [
        SimpleNamespace(source="doc1.txt", content="alpha " * 400),
        SimpleNamespace(source="doc2.txt", content="beta " * 400),
    ]

    context_text = rag._build_context_text(docs, max_tokens=400)

    assert context_text
    assert rag._estimate_tokens(context_text, "", None) <= 400


def test_query_uses_agentic_documents_and_exposes_loop_metadata():
    doc = {
        "content": "Agentic retrieval found the relevant document.",
        "source": "agentic.txt",
        "score": 0.95,
        "relevance_score": 0.95,
    }

    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    rag.agentic_loop = SimpleNamespace(
        retrieve_with_retry=AsyncMock(return_value={
            "documents": [doc],
            "confidence": 0.88,
            "iterations": 2,
            "final_status": "success",
            "queries_tried": ["original question", "refined question"],
            "reformulation_reasons": ["Added missing terminology"],
            "missing_aspects": [],
        })
    )
    rag.retriever = SimpleNamespace(
        bm25=object(),
        get_retrieval_diagnostics=lambda _retrieval_id: {"retrieval_time_ms": 1},
    )

    result = asyncio.run(rag.query("original question"))

    assert result["retrieved_docs"] == [doc]
    assert result["agentic_loop"] == {
        "enabled": True,
        "status": "success",
        "confidence": 0.88,
        "iterations": 2,
        "queries_tried": ["original question", "refined question"],
        "reformulation_reasons": ["Added missing terminology"],
        "missing_aspects": [],
    }


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
    assert metadata["first_token_time_ms"] >= 0
    assert metadata["stream_duration_ms"] >= metadata["first_token_time_ms"]


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


def test_constrained_workflow_route_returns_structured_policy_error(monkeypatch):
    class FailingService:
        async def run(self, **kwargs):
            raise ToolExecutionError("token_limit", "select", "context is too large")

    monkeypatch.setattr(
        query_routes,
        "get_constrained_workflow_service",
        lambda: FailingService(),
    )

    try:
        asyncio.run(
            query_routes.constrained_workflow(
                ConstrainedWorkflowRequest(query="question", response="answer")
            )
        )
    except Exception as error:
        assert error.status_code == 400
        assert error.detail == {
            "code": "token_limit",
            "tool_name": "select",
            "message": "context is too large",
        }
    else:
        raise AssertionError("workflow policy errors should be returned as HTTP errors")


def test_constrained_agent_route_rejects_unsupported_task(monkeypatch):
    class Agent:
        async def run(self, **kwargs):
            raise ToolExecutionError("unsupported_task", "agent", "unsupported")

    monkeypatch.setattr(query_routes, "get_constrained_workflow_agent", lambda: Agent())

    try:
        asyncio.run(
            query_routes.constrained_agent(
                query_routes.ConstrainedAgentRequest(
                    task="arbitrary_task", query="question", response="answer"
                )
            )
        )
    except Exception as error:
        assert error.status_code == 400
        assert error.detail["code"] == "unsupported_task"
    else:
        raise AssertionError("unsupported agent tasks should be rejected")


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


def test_corrective_rag_retries_when_initial_context_is_low_confidence():
    rag = HybridRAG(vector_store=object(), chat_llm=None, reasoning_llm=None, structured_llm=None, tracer=None)
    settings = get_settings()
    settings.CRAG_ENABLED = True
    settings.CRAG_CONFIDENCE_THRESHOLD = 0.55
    settings.CRAG_MAX_FALLBACKS = 2

    low_doc = SimpleNamespace(doc_id="low-doc", content="legacy context", source="low.txt", score=0.2)
    low_doc.to_dict = lambda: {"doc_id": "low-doc", "content": "legacy context", "source": "low.txt", "score": 0.2}
    high_doc = SimpleNamespace(doc_id="high-doc", content="relevant engineered fix detail", source="high.txt", score=0.9)
    high_doc.to_dict = lambda: {"doc_id": "high-doc", "content": "relevant engineered fix detail", "source": "high.txt", "score": 0.9}

    async def fake_retrieve(query, top_k=None, alpha=None):
        if "details" in query.lower() or "overview" in query.lower():
            return [high_doc], "high-retrieval"
        return [low_doc], "low-retrieval"

    rag.retriever = SimpleNamespace(retrieve=fake_retrieve, get_retrieval_diagnostics=lambda _id: {"retrieval_time_ms": 0})
    rag.reranker = SimpleNamespace(rerank=lambda query, docs, top_k=None: docs)

    docs, metadata = asyncio.run(
        rag._apply_corrective_retrieval(
            "engine failure after startup",
            [low_doc],
            analysis={"entities": ["engine"]},
            top_k=3,
            retrieval_alpha=0.7,
        )
    )

    assert metadata["triggered"] is True
    assert metadata["queries_tried"]
    assert docs[0].doc_id == "high-doc"
