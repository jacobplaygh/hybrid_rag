# Goal-Directed RAG Agent

The agent extends the existing HybridRAG system with bounded, model-directed tool selection. It is an opt-in API path; the existing query and chat flows are unchanged.

## Request

`POST /api/query/agent/run`

```json
{
  "query": "How does the API use FastAPI?",
  "top_k": 5,
  "max_steps": 5
}
```

`top_k` is limited to 1–10. `max_steps` is limited to 1–6.

## Runtime

`RAGToolAgent` builds a loop from the configured chat/reasoning LLM and request-scoped tool registry:

1. Ask the model for one JSON action.
2. Validate the action against the registered tools and argument schema.
3. Execute it with a per-run call budget, result-size budget, and timeout.
4. Add a bounded observation to the next planning turn.
5. Select document context, synthesize a source-attributed answer, then run the existing response validator.

Malformed planner JSON is retried once. Unknown tools, invalid arguments, exhausted budgets, and tool failures are surfaced as errors. If the step limit is reached, the partial completion is labeled `max_steps` in the response.

## Available Actions

| Action | Purpose |
|--------|---------|
| `search` | Hybrid document retrieval; the first search is required before finishing a document question. |
| `graph_search` | Entity and relationship retrieval from the local GraphRAG index. |
| `analytics` | Read-only aggregate analytics for summary, latency, or failures. |
| `finish` | Stop tool selection when sufficient evidence has been gathered. |

The tool registry does not expose arbitrary Python, filesystem writes, external APIs, or document mutation.

## Response

The response contains the answer, run status, tool steps and bounded observations, tools used, selected documents, source citations, optional validation details, and elapsed time. With no configured LLM, the runtime performs one bounded document search and returns a cited extract; it does not pretend that the model planner ran.

## Evaluation Status

The agent runtime is implemented, but broader rollout is not yet justified. Expand the evaluation dataset to cover tool selection, multi-step retrieval, graph search, analytics, invalid plans, citation accuracy, and latency. Agentic retrieval remains configurable because the existing small rollout evaluation regressed baseline success rate, context relevance, and groundedness.
