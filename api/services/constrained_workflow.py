"""Caller-facing boundary for constrained RAG workflows."""

from typing import Any, Optional

from rag.constrained_tools import (
    ContextSelectionTool,
    ConstrainedToolExecutor,
    ConstrainedToolWorkflow,
    DocumentSearchTool,
    ResponseValidationTool,
    WorkflowPolicy,
    ToolExecutionError,
)


class ConstrainedWorkflowService:
    """Run the bounded search, selection, and validation workflow."""

    def __init__(self, workflow: ConstrainedToolWorkflow):
        self.workflow = workflow

    @classmethod
    def from_rag(cls, rag: Any, policy: Optional[WorkflowPolicy] = None) -> "ConstrainedWorkflowService":
        """Build the service from the stable components exposed by HybridRAG."""
        workflow_policy = policy or WorkflowPolicy()
        tools = {
            "search": DocumentSearchTool(rag.retriever),
            "select": ContextSelectionTool(rag.context_manager),
            "validate": ResponseValidationTool(rag.validator),
        }
        executor = ConstrainedToolExecutor.from_policy(tools, workflow_policy)
        return cls(ConstrainedToolWorkflow(executor))

    async def run(
        self,
        query: str,
        response: str,
        history: Optional[str] = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Execute the workflow using the configured policy and budgets."""
        return await self.workflow.run(
            query=query,
            response=response,
            history=history,
            top_k=top_k,
        )


class ConstrainedWorkflowAgent:
    """Select from explicitly supported bounded workflows."""

    GROUNDED_VALIDATION_TASK = "grounded_validation"

    def __init__(self, service: ConstrainedWorkflowService):
        self.service = service

    @classmethod
    def from_rag(cls, rag: Any, policy: Optional[WorkflowPolicy] = None) -> "ConstrainedWorkflowAgent":
        """Build an agent using the existing constrained service boundary."""
        return cls(ConstrainedWorkflowService.from_rag(rag, policy))

    async def run(
        self,
        task: str,
        query: str,
        response: str,
        history: Optional[str] = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Run a supported multi-step task without arbitrary tool selection."""
        if task != self.GROUNDED_VALIDATION_TASK:
            raise ToolExecutionError(
                "unsupported_task",
                "agent",
                f"unsupported constrained task: {task}",
            )
        return await self.service.run(query, response, history, top_k)
