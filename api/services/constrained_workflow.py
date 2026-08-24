"""Caller-facing boundary for constrained RAG workflows."""

from typing import Any, Optional

from rag.constrained_tools import (
    ContextSelectionTool,
    ConstrainedToolExecutor,
    ConstrainedToolWorkflow,
    DocumentSearchTool,
    ResponseValidationTool,
    WorkflowPolicy,
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
