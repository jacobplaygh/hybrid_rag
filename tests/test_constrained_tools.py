import asyncio
from types import SimpleNamespace

from rag.constrained_tools import (
    ContextSelectionTool,
    ConstrainedToolExecutor,
    ConstrainedToolWorkflow,
    DocumentSearchTool,
    ResponseValidationTool,
    ToolExecutionError,
    WorkflowPolicy,
)
from api.services.constrained_workflow import ConstrainedWorkflowService


def test_document_search_tool_bounds_top_k():
    calls = []

    class Retriever:
        async def retrieve(self, query, top_k, alpha):
            calls.append((query, top_k, alpha))
            return ["doc"], "retrieval-id"

    result = asyncio.run(DocumentSearchTool(Retriever(), max_top_k=5).search("query", top_k=99))

    assert calls == [("query", 5, 0.5)]
    assert result["retrieval_id"] == "retrieval-id"


def test_context_selection_tool_returns_selection_report():
    class Manager:
        last_selection_report = {"selected": [{"id": "one"}]}

        def truncate_context(self, documents, query, history):
            return documents[:1]

    result = ContextSelectionTool(Manager()).select(["one", "two"], query="query")

    assert result == {
        "documents": ["one"],
        "report": {"selected": [{"id": "one"}]},
    }


def test_response_validation_tool_returns_stable_contract():
    class Validator:
        async def validate(self, query, response, documents):
            return SimpleNamespace(
                is_valid=False,
                score=0.2,
                issues=("unsupported claim",),
                sanitized_response="revised",
            )

    result = asyncio.run(
        ResponseValidationTool(Validator()).validate("query", "answer", [])
    )

    assert result == {
        "is_valid": False,
        "score": 0.2,
        "issues": ["unsupported claim"],
        "sanitized_response": "revised",
    }


def test_constrained_executor_enforces_allowlist_and_sources():
    class Search:
        async def search(self, query):
            return {"documents": [{"source": "approved.txt"}], "query": query}

    executor = ConstrainedToolExecutor(
        {"search": Search()},
        allowed_tools=["search"],
        required_sources=["approved.txt"],
    )

    result = asyncio.run(executor.execute("search", query="question"))

    assert result["documents"][0]["source"] == "approved.txt"
    try:
        asyncio.run(executor.execute("other"))
    except ToolExecutionError as error:
        assert error.code == "tool_not_allowed"
    else:
        raise AssertionError("disallowed tool should fail")


def test_constrained_executor_stops_after_failure_limit():
    async def failing_tool():
        raise RuntimeError("boom")

    executor = ConstrainedToolExecutor({"failing_tool": failing_tool}, max_failures=1)

    try:
        asyncio.run(executor.execute("failing_tool"))
    except ToolExecutionError as error:
        assert error.code == "tool_failed"
    else:
        raise AssertionError("tool should fail")

    try:
        asyncio.run(executor.execute("failing_tool"))
    except ToolExecutionError as error:
        assert error.code == "failure_limit"
    else:
        raise AssertionError("failure limit should stop execution")


def test_constrained_executor_enforces_call_and_token_budgets():
    async def tool():
        return {"documents": [{"content": "one two three four"}]}

    executor = ConstrainedToolExecutor(
        {"tool": tool}, max_calls=1, max_tokens=3
    )

    try:
        asyncio.run(executor.execute("tool"))
    except ToolExecutionError as error:
        assert error.code == "token_limit"
        assert error.as_dict()["tool_name"] == "tool"
    else:
        raise AssertionError("token budget should stop execution")

    try:
        asyncio.run(executor.execute("tool"))
    except ToolExecutionError as error:
        assert error.code == "call_limit"
    else:
        raise AssertionError("call budget should stop execution")


def test_constrained_workflow_composes_search_selection_and_validation():
    class Retriever:
        async def retrieve(self, query, top_k, alpha):
            return [{"source": "approved.txt", "content": "grounded context"}], "rid"

    class Manager:
        last_selection_report = {"selected": [{"id": "one"}]}

        def truncate_context(self, documents, query, history):
            return documents

    class Validator:
        async def validate(self, query, response, documents):
            return SimpleNamespace(
                is_valid=True, score=1.0, issues=[], sanitized_response=None
            )

    executor = ConstrainedToolExecutor(
        {
            "search": DocumentSearchTool(Retriever()),
            "select": ContextSelectionTool(Manager()),
            "validate": ResponseValidationTool(Validator()),
        },
        required_sources=["approved.txt"],
        max_calls=3,
    )
    result = asyncio.run(
        ConstrainedToolWorkflow(executor).run("question", "grounded answer")
    )

    assert result["search"]["retrieval_id"] == "rid"
    assert result["selection"]["documents"]
    assert result["validation"]["is_valid"] is True


def test_workflow_policy_builds_executor_with_explicit_limits():
    policy = WorkflowPolicy(
        allowed_tools=("search",),
        required_sources=("approved.txt",),
        max_failures=2,
        max_calls=4,
        max_tokens=50,
    )

    executor = ConstrainedToolExecutor.from_policy({}, policy)

    assert executor.allowed_tools == {"search"}
    assert executor.required_sources == {"approved.txt"}
    assert executor.max_failures == 2
    assert executor.max_calls == 4
    assert executor.max_tokens == 50


def test_service_factory_binds_hybrid_rag_components():
    rag = SimpleNamespace(retriever=object(), context_manager=object(), validator=object())

    service = ConstrainedWorkflowService.from_rag(rag)

    tools = service.workflow.executor.tools
    assert isinstance(tools["search"], DocumentSearchTool)
    assert isinstance(tools["select"], ContextSelectionTool)
    assert isinstance(tools["validate"], ResponseValidationTool)