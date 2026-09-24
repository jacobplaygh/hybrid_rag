"""Caller-facing boundary for constrained RAG workflows."""

from typing import Any, Optional

from rag.constrained_tools import (
    AnalyticsLookupTool,
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
            "analytics": AnalyticsLookupTool(rag.analytics),
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
    ANALYTICS_DIAGNOSIS_TASK = "analytics_diagnosis"

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
        if task not in (self.GROUNDED_VALIDATION_TASK, self.ANALYTICS_DIAGNOSIS_TASK):
            raise ToolExecutionError(
                "unsupported_task",
                "agent",
                f"unsupported constrained task: {task}",
            )
        if task == self.GROUNDED_VALIDATION_TASK:
            return await self.service.run(query, response, history, top_k)
        
        if task == self.ANALYTICS_DIAGNOSIS_TASK:
            # Analytics diagnosis uses a specific policy: 
            # Allowed: {"lookup", "search", "select", "validate"}, Calls: 5, Tokens: 2000
            # We override the service's executor policy for this specific task.
            from rag.constrained_tools import WorkflowPolicy
            policy = WorkflowPolicy(
                allowed_tools=("analytics", "search", "select", "validate"),
                max_calls=5,
                max_tokens=2000
            )
            # Re-initialize service with the task-specific policy
            # Note: In a production system, we'd likely have a policy registry.
            from rag.constrained_tools import ConstrainedToolExecutor
            self.service.workflow.executor = ConstrainedToolExecutor.from_policy(
                self.service.workflow.executor.tools, policy
            )
            return await self.service.run(query, response, history, top_k)
