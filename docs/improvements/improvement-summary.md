# Improvement Summary (Quick Reference)

Current implementation status for the Hybrid RAG system.

## Current Capabilities

- **Hybrid retrieval** combines semantic search with BM25 keyword search and optional reranking.
- **Quality and reliability** include confidence scoring, response validation, source attribution, and structured API errors.
- **Context engineering** provides bounded document/history selection, diversity limits, and selection reports.
- **Advanced retrieval** includes query understanding, corrective retrieval, configurable agentic retries, dynamic routing, and GraphRAG.
- **Observability** includes analytics, Prometheus metrics, request tracing, and context-quality evaluation.
- **Goal-directed RAG agent** is available at `POST /api/query/agent/run`. It plans bounded, read-only calls to document search, graph search, and aggregate analytics, then synthesizes a source-attributed answer.

## Agent Safety and Limits

- Only registered read-only tools can be called; arbitrary code, writes, and external actions are not exposed.
- Each run has a configurable maximum of 1–6 tool steps, a maximum retrieval size of 10, per-call result-size limits, and tool timeouts.
- Document requests require a hybrid `search` call before the agent can finish; GraphRAG is an additional tool, not a substitute for that required search.
- The response includes tool steps, retrieved documents, citations, and validation status; reaching the step limit is reported explicitly.
- The existing `/api/query/` and `/api/query/agent` endpoints remain available and unchanged.

See [Agent Architecture](../architecture/agent.md) for the runtime and API contract.

## Progress and Next Focus

| Area | Status | Current position |
|------|--------|------------------|
| Foundation, retrieval, context, and observability | Implemented | Covered by the existing RAG pipeline |
| Goal-directed agent runtime | Implemented | Bounded tool planning and grounded synthesis |
| Agent evaluation and rollout | In progress | Expand representative agent evaluations and compare quality/latency against baseline |

Agentic retrieval remains configurable. Its recorded 15-case evaluation showed lower success rate, context relevance, and groundedness than baseline; do not make that retrieval path the unconditional default until it passes the release gates in [Phase 2 rollout evaluation](../development/phase-2-rollout-evaluation.md).
