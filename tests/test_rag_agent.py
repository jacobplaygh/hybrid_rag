import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import query as query_routes
from rag.agent_runtime import AgentConfig, RAGToolAgent
from rag.constrained_tools import ToolExecutionError


class FakeContextManager:
    def truncate_context(self, documents, query, history=None):
        return documents


class FakeValidator:
    async def validate(self, query, response, context_docs):
        return SimpleNamespace(is_valid=True, score=0.95, issues=[], sanitized_response=None)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=next(self.responses))


def test_agent_plans_tools_and_returns_source_citations():
    llm = FakeLLM([
        '{"action":"search","args":{"query":"FastAPI API layer","top_k":2}}',
        '{"action":"graph_search","args":{"query":"FastAPI","top_k":2}}',
        '{"action":"finish","args":{}}',
        "The API layer uses FastAPI [guide.md].",
    ])
    graph_calls = []

    async def search(query, top_k, alpha):
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "The API layer uses FastAPI.",
                "source": "guide.md",
                "score": 0.9,
                "metadata": {},
            }]
        }

    def graph_search(query, top_k):
        graph_calls.append((query, top_k))
        return [{
            "doc_id": "graph-1",
            "content": "FastAPI is related to the API layer.",
            "source": "graph",
            "score": 0.8,
            "metadata": {"graph_rag": True},
        }]

    agent = RAGToolAgent(
        tools={"search": search, "graph_search": graph_search},
        context_manager=FakeContextManager(),
        llm=llm,
        validator=FakeValidator(),
        config=AgentConfig(max_steps=4),
    )

    result = asyncio.run(agent.run("What framework is used for the API?", top_k=3))

    assert [step.action for step in result.steps] == ["search", "graph_search"]
    assert graph_calls == [("FastAPI", 2)]
    assert result.tools_used == ["search", "graph_search"]
    assert result.citations[0]["source"] == "guide.md"
    assert result.validation["is_valid"] is True
    assert "[guide.md]" in result.response


def test_agent_without_llm_uses_bounded_search_and_grounded_fallback():
    calls = []

    async def search(query, top_k, alpha):
        calls.append((query, top_k, alpha))
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "FastAPI serves the backend API.",
                "source": "backend.md",
                "score": 0.8,
                "metadata": {},
            }]
        }

    agent = RAGToolAgent(
        tools={"search": search},
        context_manager=FakeContextManager(),
    )

    result = asyncio.run(agent.run("What serves the API?", top_k=50))

    assert calls == [("What serves the API?", 10, 0.5)]
    assert result.status == "completed"
    assert "backend.md" in result.response
    assert result.citations[0]["source"] == "backend.md"


def test_agent_reports_when_step_budget_is_reached():
    llm = FakeLLM([
        '{"action":"search","args":{"query":"system detail"}}',
        "A system detail is available [system.md].",
    ])

    async def search(query, top_k, alpha):
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "System detail.",
                "source": "system.md",
                "score": 0.7,
                "metadata": {},
            }]
        }

    agent = RAGToolAgent(
        tools={"search": search},
        context_manager=FakeContextManager(),
        llm=llm,
        config=AgentConfig(max_steps=1),
    )

    result = asyncio.run(agent.run("Find a system detail"))

    assert result.status == "max_steps"
    assert len(result.steps) == 1
    assert "configured limit of 1" in result.response


def test_agent_rejects_planned_tools_outside_the_request_policy():
    llm = FakeLLM(['{"action":"analytics","args":{"metric":"latency"}}'])

    async def search(query, top_k, alpha):
        return {"documents": []}

    agent = RAGToolAgent(
        tools={"search": search},
        context_manager=FakeContextManager(),
        llm=llm,
    )

    with pytest.raises(ToolExecutionError) as error:
        asyncio.run(agent.run("What is the system latency?"))

    assert error.value.code == "tool_not_allowed"


def test_agent_retries_a_malformed_plan_once():
    llm = FakeLLM([
        "not JSON",
        '{"action":"search","args":{"query":"fact"}}',
        "Grounded answer [source.md].",
    ])

    async def search(query, top_k, alpha):
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "A fact.",
                "source": "source.md",
                "score": 0.7,
                "metadata": {},
            }]
        }

    agent = RAGToolAgent(
        tools={"search": search},
        context_manager=FakeContextManager(),
        llm=llm,
        config=AgentConfig(max_steps=1),
    )

    result = asyncio.run(agent.run("Find a fact"))

    assert result.steps[0].action == "search"
    assert len(llm.prompts) == 3
    assert "previous response was invalid" in llm.prompts[1]


def test_agent_allows_analytics_only_completion_for_operational_request():
    llm = FakeLLM([
        '{"action":"analytics","args":{"metric":"latency"}}',
        '{"action":"finish","args":{}}',
        "System analytics report an average latency of 200 ms.",
    ])

    async def search(query, top_k, alpha):
        raise AssertionError("an operational analytics query should not require document search")

    def analytics(metric, limit):
        return {"metric": metric, "result": {"average_latency_ms": 200}}

    agent = RAGToolAgent(
        tools={"search": search, "analytics": analytics},
        context_manager=FakeContextManager(),
        llm=llm,
    )

    result = asyncio.run(agent.run("What is the system latency?"))

    assert result.status == "completed"
    assert result.tools_used == ["analytics"]
    assert "200 ms" in result.response


def test_agent_requires_document_evidence_for_non_analytics_request():
    llm = FakeLLM([
        '{"action":"analytics","args":{"metric":"latency"}}',
        '{"action":"finish","args":{}}',
        "The indexed document says the system uses FastAPI [api.md].",
    ])
    calls = []

    async def search(query, top_k, alpha):
        calls.append(query)
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "The system uses FastAPI.",
                "source": "api.md",
                "score": 0.9,
                "metadata": {},
            }]
        }

    def analytics(metric, limit):
        return {"metric": metric, "result": {"average_latency_ms": 200}}

    agent = RAGToolAgent(
        tools={"search": search, "analytics": analytics},
        context_manager=FakeContextManager(),
        llm=llm,
    )

    result = asyncio.run(agent.run("Which framework does the system use?"))

    assert calls == ["Which framework does the system use?"]
    assert result.tools_used == ["search"]
    assert result.citations[0]["source"] == "api.md"


def test_agent_searches_documents_before_relying_on_graph_results():
    llm = FakeLLM([
        '{"action":"graph_search","args":{"query":"FastAPI"}}',
        '{"action":"graph_search","args":{"query":"FastAPI"}}',
        '{"action":"finish","args":{}}',
        "The system uses FastAPI [api.md].",
    ])
    calls = []

    async def search(query, top_k, alpha):
        calls.append("search")
        return {
            "documents": [{
                "doc_id": "doc-1",
                "content": "The system uses FastAPI.",
                "source": "api.md",
                "score": 0.9,
                "metadata": {},
            }]
        }

    def graph_search(query, top_k):
        calls.append("graph_search")
        return [{
            "doc_id": "graph-1",
            "content": "FastAPI is an API framework.",
            "source": "graph",
            "score": 0.8,
            "metadata": {},
        }]

    agent = RAGToolAgent(
        tools={"search": search, "graph_search": graph_search},
        context_manager=FakeContextManager(),
        llm=llm,
    )

    result = asyncio.run(agent.run("Which framework does the system use?"))

    assert calls == ["search", "graph_search"]
    assert result.tools_used == ["search", "graph_search"]


def test_agent_run_api_uses_requested_step_budget(monkeypatch):
    class FakeAgent:
        def as_dict(self, result):
            return {
                "query": result.query,
                "response": result.response,
                "status": result.status,
                "steps": [],
                "tools_used": [],
                "retrieved_docs": [],
                "citations": [],
                "validation": None,
                "elapsed_ms": 1.0,
            }

    class FakeService:
        agent = FakeAgent()

        async def run(self, query, top_k):
            return SimpleNamespace(query=query, response="Done.", status="completed")

    requested = []

    def get_service(max_steps):
        requested.append(max_steps)
        return FakeService()

    monkeypatch.setattr(query_routes, "get_rag_agent_service", get_service)
    client = TestClient(app)
    response = client.post(
        "/api/query/agent/run",
        json={"query": "Find a fact", "top_k": 2, "max_steps": 3},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Done."
    assert requested == [3]
