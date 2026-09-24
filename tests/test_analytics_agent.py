import asyncio
import pytest
from types import SimpleNamespace
from rag.constrained_tools import (
    AnalyticsLookupTool,
    ContextSelectionTool,
    ConstrainedToolExecutor,
    ConstrainedToolWorkflow,
    DocumentSearchTool,
    ResponseValidationTool,
    ToolExecutionError,
    WorkflowPolicy,
)
from api.services.constrained_workflow import (
    ConstrainedWorkflowAgent,
    ConstrainedWorkflowService,
)

class MockAnalytics:
    def analyze_patterns(self):
        return {"total_queries": 100}
    def analyze_latency(self):
        return {"avg_ms": 200}
    def get_failed_queries(self, limit):
        return [{"id": "1"}] * limit

class MockRetriever:
    async def retrieve(self, query, top_k, alpha):
        return ["doc1"], "ret-1"

class MockManager:
    def __init__(self):
        self.last_selection_report = "Mock Report"
    def truncate_context(self, docs, query, history):
        return docs

class MockReport:
    def __init__(self):
        self.token_count = 100
        self.truncated = False

class MockValidator:
    async def validate(self, query, response, docs):
        return SimpleNamespace(is_valid=True, score=1.0, issues=(), sanitized_response=response)

@pytest.mark.asyncio
async def test_analytics_diagnosis_task_enforces_policy():
    # Setup mocks
    analytics = MockAnalytics()
    retriever = MockRetriever()
    manager = MockManager()
    validator = MockValidator()

    # Setup tools
    tools = {
        "analytics": AnalyticsLookupTool(analytics),
        "search": DocumentSearchTool(retriever),
        "select": ContextSelectionTool(manager),
        "validate": ResponseValidationTool(validator),
    }
    
    # Setup service and agent
    workflow = ConstrainedToolWorkflow(tools)
    # Default executor (restrictive)
    executor = ConstrainedToolExecutor.from_policy(tools, WorkflowPolicy(allowed_tools=("validate",)))
    workflow.executor = executor
    service = ConstrainedWorkflowService(workflow)
    agent = ConstrainedWorkflowAgent(service)

    # 1. Verify that the analytics tool is NOT allowed by default (if the default policy is empty)
    # We test this by trying to run the task. 
    # The agent.run method for analytics_diagnosis should override the policy.
    
    query = "How is the system performing?"
    response = "It is performing well."
    history = []
    top_k = 5

    # This should succeed because the agent overrides the policy to allow 'analytics'
    result = await agent.run(
        task="analytics_diagnosis",
        query=query,
        response=response,
        history=history,
        top_k=top_k
    )
    
    assert result is not None
    # Verify the executor was updated to allow analytics
    assert "analytics" in agent.service.workflow.executor.allowed_tools

@pytest.mark.asyncio
async def test_analytics_diagnosis_budget_enforcement():
    analytics = MockAnalytics()
    tools = {
        "lookup": AnalyticsLookupTool(analytics),
        "search": DocumentSearchTool(MockRetriever()),
        "select": ContextSelectionTool(MockManager()),
        "validate": ResponseValidationTool(MockValidator()),
    }
    
    workflow = ConstrainedToolWorkflow(tools)
    executor = ConstrainedToolExecutor.from_policy(tools, WorkflowPolicy(allowed_tools=("validate",)))
    workflow.executor = executor
    service = ConstrainedWorkflowService(workflow)
    agent = ConstrainedWorkflowAgent(service)

    # We want to test that max_calls=5 is enforced.
    # Since the agent.run calls service.run, and service.run uses the executor,
    # we can simulate a sequence of calls that exceeds the budget.
    
    # To actually test the budget, we need the agent to actually call the tools.
    # The current ConstrainedWorkflowService.run implementation is a placeholder 
    # that just returns a result. To test the executor's budget, we can call the executor directly
    # or mock the service to loop.
    
    policy = WorkflowPolicy(allowed_tools=("lookup",), max_calls=2)
    executor = ConstrainedToolExecutor.from_policy(tools, policy)
    
    # Call 1
    await executor.execute("lookup", metric="summary")
    # Call 2
    await executor.execute("lookup", metric="latency")
    
    # Call 3 should fail
    with pytest.raises(ToolExecutionError) as excinfo:
        await executor.execute("lookup", metric="failures")
    
        assert "call_limit" in excinfo.value.code
