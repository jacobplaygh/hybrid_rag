import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from api.config import get_settings
from rag.constrained_tools import WorkflowPolicy, ConstrainedToolExecutor
from rag.hybrid_rag import HybridRAG
from data.vector_store import ChromaVectorStore
from rag.context_manager import ContextManager
from rag.response_validator import ResponseValidator

@dataclass
class EvalResult:
    query: str
    policy_name: str
    response: str
    latency: float
    tokens_used: int
    tool_calls: int
    is_valid: bool
    score: float
    error: Optional[str] = None

class PolicyEvalRunner:
    """Runs a set of queries against different WorkflowPolicy configurations to optimize budgets."""

    def __init__(self, rag_system: HybridRAG):
        self.rag = rag_system
        self.settings = get_settings()

    async def run_query(self, query: str, policy: WorkflowPolicy, policy_name: str) -> EvalResult:
        start_time = time.perf_counter()
        
        # We assume the rag_system has a method to execute a constrained workflow
        # If not, we'll need to implement a wrapper that uses ConstrainedToolExecutor
        try:
            # This is a conceptual call; we will align it with the actual ConstrainedWorkflowAgent implementation
            # For now, we simulate the execution flow
            result = await self.rag.execute_constrained_query(query, policy)
            
            latency = time.perf_counter() - start_time
            return EvalResult(
                query=query,
                policy_name=policy_name,
                response=result.get("response", ""),
                latency=latency,
                tokens_used=result.get("tokens_used", 0),
                tool_calls=result.get("tool_calls", 0),
                is_valid=result.get("is_valid", False),
                score=result.get("score", 0.0)
            )
        except Exception as e:
            return EvalResult(
                query=query,
                policy_name=policy_name,
                response="",
                latency=time.perf_counter() - start_time,
                tokens_used=0,
                tool_calls=0,
                is_valid=False,
                score=0.0,
                error=str(e)
            )

    async def evaluate_policies(self, queries_file: str, policies: Dict[str, WorkflowPolicy]):
        results = []
        with open(queries_file, 'r') as f:
            queries = [json.loads(line) for line in f]

        for policy_name, policy in policies.items():
            print(f"Evaluating policy: {policy_name}...")
            for q_data in queries:
                query = q_data["query"]
                res = await self.run_query(query, policy, policy_name)
                results.append(asdict(res))
        
        return results

async def main():
    # Setup RAG system
    settings = get_settings()
    vector_store = ChromaVectorStore()
    rag = HybridRAG(vector_store)

    runner = PolicyEvalRunner(rag)

    # Define policies to test
    test_policies = {
        "strict": WorkflowPolicy(max_calls=3, max_tokens=1000),
        "balanced": WorkflowPolicy(max_calls=10, max_tokens=4000),
        "generous": WorkflowPolicy(max_calls=20, max_tokens=8000),
    }

    queries_path = "hybrid_rag/tools/eval_queries.jsonl"
    results = await runner.evaluate_policies(queries_path, test_policies)

    with open("hybrid_rag/tools/policy_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation complete. Results saved to hybrid_rag/tools/policy_eval_results.json")

if __name__ == "__main__":
    asyncio.run(main())
