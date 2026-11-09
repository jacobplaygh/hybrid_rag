"""Factory and request boundary for the bounded RAG tool agent."""

from typing import Any, Dict, Optional

from rag.agent_runtime import AgentConfig, AgentResult, RAGToolAgent
from rag.constrained_tools import AnalyticsLookupTool, DocumentSearchTool


class AgentAnalyticsTool:
    """Expose the existing read-only analytics lookup under the agent tool name."""

    def __init__(self, analytics: Any):
        self.lookup = AnalyticsLookupTool(analytics)

    def analytics(self, metric: str, limit: int = 10) -> Dict[str, Any]:
        return self.lookup.lookup(metric, limit)


class RAGAgentService:
    """Compose existing HybridRAG components into an isolated agent per request."""

    def __init__(self, agent: RAGToolAgent):
        self.agent = agent

    @classmethod
    def from_rag(
        cls, rag: Any, max_steps: int = 5, config: Optional[AgentConfig] = None
    ) -> "RAGAgentService":
        agent_config = config or AgentConfig(max_steps=max_steps)
        search_tool = DocumentSearchTool(rag.retriever, max_top_k=agent_config.max_top_k)
        tools: Dict[str, Any] = {"search": search_tool.search}

        if getattr(rag, "graph_rag", None) is not None:
            def graph_search(query: str, top_k: int = 5) -> Dict[str, Any]:
                return {
                    "query": query,
                    "documents": rag.graph_rag.retrieve(query, top_k=top_k),
                }

            tools["graph_search"] = graph_search

        if getattr(rag, "analytics", None) is not None:
            tools["analytics"] = AgentAnalyticsTool(rag.analytics).analytics

        return cls(
            RAGToolAgent(
                tools=tools,
                context_manager=rag.context_manager,
                llm=getattr(rag, "chat_llm", None) or getattr(rag, "reasoning_llm", None),
                validator=getattr(rag, "validator", None),
                config=agent_config,
            )
        )

    async def run(self, query: str, top_k: int = 5) -> AgentResult:
        return await self.agent.run(query=query, top_k=top_k)
