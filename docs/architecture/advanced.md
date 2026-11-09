# Retrieval and context details

This page describes the implemented pipeline rather than a proposed vector-store or reranking design.

## Hybrid retrieval

`HybridRetriever` combines semantic vector-store results and BM25 keyword results. Each result list is rank-normalized and weighted by the retrieval `alpha`; query-term overlap adds a lexical relevance adjustment before results are sorted. Optional reranking is applied by the RAG orchestration layer when configured. If no reranker is available, retrieval score order is retained.

The current implementation does not use a fixed, documented `alpha=0.5` default across every request. Retrieval callers can provide an alpha, and query routing can select a retrieval strategy. See `rag/retrieval.py` and `rag/retrieval_router.py` for the live behavior.

## Query and context stages

The standard query flow may use query understanding and dynamic routing, then retrieval, context selection, answer generation, confidence/quality scoring, and response validation. `ContextManager` applies a token budget, ranks usable candidates by score when available, removes duplicate chunks, and can limit the number of documents retained per source. Its selection report records selected and dropped items and reasons.

Conversation context is managed separately by `ConversationMemory`. The goal-directed agent has its own per-run step, call, result-size, and timeout limits; these limits are not the same as the standard query context budget.

## Optional retrieval behaviors

- **Reranking:** enabled through `RETRIEVAL_RERANK`; model availability and returned scores affect whether a reranked order is used.
- **Dynamic routing:** controlled by `DYNAMIC_ROUTING_ENABLED`.
- **Iterative agentic retrieval:** controlled by `AGENTIC_LOOP_ENABLED`, which defaults to `false` pending the evaluation gates.
- **Corrective retrieval and semantic caching:** configured separately through `CRAG_*` and `SEMANTIC_CACHE_*` settings.
- **GraphRAG:** lightweight local entity/relationship retrieval, triggered by query analysis or the goal-directed agent.

These are separate mechanisms. In particular, the goal-directed tool-using agent at `/api/query/agent/run` is not the same feature as the iterative agentic retrieval loop used by the normal query path.

## Source of truth

Configuration defaults are defined in `api/config.py`. Retrieval, routing, context selection, caching, and graph behavior live in `rag/`; API contracts and route paths are defined in `api/schemas.py` and `api/routes/`.
