"""LangSmith tracing integration for LLM observability."""

import logging
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from langsmith.client import Client
    HAS_LANGSMITH = True
except ImportError:
    HAS_LANGSMITH = False


class LangSmithTracer:
    """Wrapper for LangSmith tracing."""
    
    def __init__(self, enabled: bool = False, project_name: str = "hybrid-rag"):
        """
        Initialize LangSmith tracer.
        
        Requires LANGSMITH_API_KEY environment variable if enabled.
        """
        self.enabled = enabled
        self.project_name = project_name
        self.client = None
        
        if self.enabled and not HAS_LANGSMITH:
            logger.warning("LangSmith tracing requested but package is not installed")
            self.enabled = False

        if self.enabled:
            api_key = os.getenv("LANGSMITH_API_KEY")
            api_url = os.getenv("LANGSMITH_API_URL", "https://apac.api.smith.langchain.com")
            if not api_key:
                logger.warning("LangSmith enabled but LANGSMITH_API_KEY not set")
                self.enabled = False
            else:
                try:
                    self.client = Client(api_key=api_key, api_url=api_url)
                    logger.info(f"✅ LangSmith tracing enabled for project: {project_name} (URL: {api_url})")
                except Exception as exc:
                    logger.error(f"Failed to initialize LangSmith client: {exc}")
                    self.enabled = False
        else:
            logger.info("LangSmith tracing disabled")

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
    
    def trace_query(self, query_id: str, query: str, metadata: Dict[str, Any] = None):
        """Record a query trace run."""
        if not self.enabled or not self.client:
            return

        try:
            self.client.create_run(
                name=f"query-{query_id}",
                inputs={"query": query, **(metadata or {})},
                outputs={"status": "started"},
                run_type="tool",
                project_name=self.project_name,
                start_time=self._now(),
                end_time=self._now(),
            )
        except Exception as exc:
            logger.warning(f"LangSmith trace_query failed: {exc}")
    
    def trace_query_result(self, query_id: str, result: Dict[str, Any]):
        """Record the final result of a query, including quality metrics."""
        if not self.enabled or not self.client:
            return

        try:
            # Extract quality metrics and confidence for the trace
            confidence = result.get("confidence_score")
            quality = result.get("quality_metrics")
            
            outputs = {
                "response": result.get("response"),
                "confidence_score": confidence.overall_score if hasattr(confidence, 'overall_score') else confidence,
                "ndcg": quality if isinstance(quality, (int, float)) else None,
                "status": "completed"
            }
            
            # Update the existing run if possible, or create a result run
            self.client.create_run(
                name=f"result-{query_id}",
                inputs={"query_id": query_id},
                outputs=outputs,
                run_type="tool",
                project_name=self.project_name,
                start_time=self._now(),
                end_time=self._now(),
            )
        except Exception as exc:
            logger.warning(f"LangSmith trace_query_result failed: {exc}")
    
    def trace_llm_call(self, model: str, prompt: str, response: str, tokens: int, confidence: Any = None):
        """Record an LLM call trace."""
        if not self.enabled or not self.client:
            return

        try:
            outputs = {"response": response, "tokens_used": tokens}
            if confidence:
                outputs["confidence_score"] = confidence.overall_score if hasattr(confidence, 'overall_score') else confidence

            self.client.create_run(
                name=f"llm-{model}-{datetime.now(timezone.utc).isoformat()}",
                inputs={"model": model, "prompt": prompt},
                outputs=outputs,
                run_type="llm",
                project_name=self.project_name,
                start_time=self._now(),
                end_time=self._now(),
            )
        except Exception as exc:
            logger.warning(f"LangSmith trace_llm_call failed: {exc}")
    
    def trace_retrieval(self, query: str, results_count: int, latency_ms: float, ndcg: float = None):
        """Record a retrieval trace."""
        if not self.enabled or not self.client:
            return

        try:
            outputs = {"results_count": results_count, "latency_ms": latency_ms}
            if ndcg is not None:
                outputs["ndcg"] = ndcg

            self.client.create_run(
                name=f"retrieval-{datetime.now(timezone.utc).isoformat()}",
                inputs={"query": query},
                outputs=outputs,
                run_type="retriever",
                project_name=self.project_name,
                start_time=self._now(),
                end_time=self._now(),
            )
        except Exception as exc:
            logger.warning(f"LangSmith trace_retrieval failed: {exc}")
    
    def end_trace(self, query_id: str):
        """End trace placeholder."""
        if not self.enabled:
            return
        logger.debug(f"Ending trace for query {query_id}")


def get_tracer(enabled: bool = False, project: str = "hybrid-rag") -> LangSmithTracer:
    """Get LangSmith tracer instance."""
    return LangSmithTracer(enabled=enabled, project_name=project)
