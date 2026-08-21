"""Narrow, budget-aware tools for future constrained agent workflows."""

import inspect
import json
from typing import Any, Dict, List, Optional


class ToolExecutionError(RuntimeError):
    """Stable error contract for constrained tool callers."""

    def __init__(self, code: str, tool_name: str, message: str):
        super().__init__(message)
        self.code = code
        self.tool_name = tool_name
        self.message = message

    def as_dict(self) -> Dict[str, str]:
        return {
            "code": self.code,
            "tool_name": self.tool_name,
            "message": self.message,
        }


class DocumentSearchTool:
    """Expose bounded document retrieval without exposing the retriever internals."""

    def __init__(self, retriever: Any, max_top_k: int = 10):
        self.retriever = retriever
        self.max_top_k = max_top_k

    async def search(self, query: str, top_k: int = 3, alpha: float = 0.5) -> Dict[str, Any]:
        bounded_top_k = min(max(int(top_k), 1), self.max_top_k)
        documents, retrieval_id = await self.retriever.retrieve(
            query, top_k=bounded_top_k, alpha=alpha
        )
        return {
            "query": query,
            "retrieval_id": retrieval_id,
            "documents": documents,
            "top_k": bounded_top_k,
        }


class ContextSelectionTool:
    """Expose context assembly while preserving the manager's token budget."""

    def __init__(self, context_manager: Any):
        self.context_manager = context_manager

    def select(
        self,
        documents: List[Any],
        query: Optional[str] = None,
        history: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected = self.context_manager.truncate_context(documents, query, history)
        return {
            "documents": selected,
            "report": self.context_manager.last_selection_report,
        }


class ResponseValidationTool:
    """Normalize response-validator output for a future tool-calling layer."""

    def __init__(self, validator: Any):
        self.validator = validator

    async def validate(
        self, query: str, response: str, documents: List[Any]
    ) -> Dict[str, Any]:
        result = await self.validator.validate(query, response, documents)
        return {
            "is_valid": result.is_valid,
            "score": result.score,
            "issues": list(result.issues),
            "sanitized_response": result.sanitized_response,
        }


class ConstrainedToolExecutor:
    """Apply allowlists, source requirements, and failure limits to tool calls."""

    def __init__(
        self,
        tools: Dict[str, Any],
        allowed_tools: Optional[List[str]] = None,
        required_sources: Optional[List[str]] = None,
        max_failures: int = 3,
        max_calls: int = 10,
        max_tokens: Optional[int] = None,
    ):
        if max_failures < 1:
            raise ValueError("max_failures must be at least 1")
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.tools = tools
        self.allowed_tools = set(allowed_tools or tools)
        self.required_sources = set(required_sources or [])
        self.max_failures = max_failures
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.call_count = 0
        self.failure_count = 0

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if self.failure_count >= self.max_failures:
            raise ToolExecutionError("failure_limit", tool_name, "tool failure limit exceeded")
        if self.call_count >= self.max_calls:
            raise ToolExecutionError("call_limit", tool_name, "tool call limit exceeded")
        if tool_name not in self.allowed_tools:
            raise ToolExecutionError("tool_not_allowed", tool_name, f"tool is not allowed: {tool_name}")
        if tool_name not in self.tools:
            raise ToolExecutionError("unknown_tool", tool_name, f"unknown tool: {tool_name}")

        self.call_count += 1
        try:
            tool = self.tools[tool_name]
            method = getattr(tool, tool_name, tool)
            result = method(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            self._check_sources(result, kwargs.get("documents", []))
            self._check_tokens(result, tool_name)
            return result
        except ToolExecutionError:
            self.failure_count += 1
            raise
        except Exception as error:
            self.failure_count += 1
            raise ToolExecutionError("tool_failed", tool_name, str(error)) from error

    def _check_sources(self, result: Any, input_documents: Any = None) -> None:
        if not self.required_sources:
            return
        documents = result.get("documents", []) if isinstance(result, dict) else []
        documents = list(documents) + list(input_documents or [])
        sources = {
            doc.get("source") if isinstance(doc, dict) else getattr(doc, "source", None)
            for doc in documents
        }
        missing = self.required_sources - sources
        if missing:
            raise ValueError(f"required sources missing: {sorted(missing)}")

    def _check_tokens(self, result: Any, tool_name: str) -> None:
        if self.max_tokens is None:
            return
        token_count = len(json.dumps(result, default=str).split())
        if token_count > self.max_tokens:
            raise ToolExecutionError(
                "token_limit",
                tool_name,
                f"tool result exceeded token budget: {token_count} > {self.max_tokens}",
            )


class ConstrainedToolWorkflow:
    """Compose retrieval, context selection, and validation under one executor."""

    def __init__(self, executor: ConstrainedToolExecutor):
        self.executor = executor

    async def run(
        self,
        query: str,
        response: str,
        history: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        search = await self.executor.execute("search", query=query, top_k=top_k)
        selection = await self.executor.execute(
            "select",
            documents=search["documents"],
            query=query,
            history=history,
        )
        validation = await self.executor.execute(
            "validate",
            query=query,
            response=response,
            documents=selection["documents"],
        )
        return {
            "search": search,
            "selection": selection,
            "validation": validation,
        }