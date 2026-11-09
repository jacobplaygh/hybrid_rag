# Hybrid RAG documentation

This documentation describes the implementation in this repository. For repository conventions and contribution workflow, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Start here

- [Getting started](./getting-started/README.md)
- [Installation](./getting-started/installation.md)
- [Quick start](./getting-started/quick-start.md)

## Architecture and API

- [Architecture overview](./architecture/overview.md)
- [Retrieval and context details](./architecture/advanced.md)
- [API integration examples](./architecture/integration.md)
- [Goal-directed agent](./architecture/agent.md)

## Development and evaluation

- [Development guide](./development/README.md)
- [Testing guide](./development/testing.md)
- [Improvement summary](./improvements/improvement-summary.md)
- [Implementation roadmap](./improvements/implementation-roadmap.md)
- [Strategic improvement plan](./improvements/improvement-plan.md)
- [Context engineering plan](./improvements/context-engineering-plan.md)
- [Agentic retrieval rollout evaluation](./development/phase-2-rollout-evaluation.md)

## Current implementation status

| Area | Status |
|---|---|
| FastAPI endpoints, document ingestion, query/chat, and streaming | Implemented |
| Hybrid retrieval, reranking, context selection, validation, and analytics | Implemented |
| Optional iterative agentic retrieval | Implemented; disabled by default pending evaluation gates |
| Goal-directed, bounded RAG agent | Implemented at `POST /api/query/agent/run` |
| Agent evaluation and broader rollout | In progress |

Run `python -m pytest -q` from the workspace directory containing `hybrid_rag` for the current test suite. Test counts are intentionally omitted here because they change as tests are added.
