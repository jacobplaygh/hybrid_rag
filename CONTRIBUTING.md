# Repository Organization Guide

This repository is a production-oriented Hybrid RAG service. Keep changes close to the layer that owns the behavior, and preserve the public API unless a change is explicitly required.

## Directory Boundaries

| Directory | Responsibility |
|---|---|
| `api/` | FastAPI application setup, configuration, schemas, routes, and UI |
| `rag/` | Retrieval orchestration, context engineering, chains, memory, validation, and evaluation logic |
| `data/` | Vector-store adapters and local persistence integration |
| `observability/` | Logging, tracing, metrics, and telemetry setup |
| `tests/` | Unit, integration, and behavior regression tests |
| `tools/` | Evaluation scripts and developer utilities |
| `docs/` | Architecture, setup, development, and improvement plans |

## Where New Features Go

- Context selection, token budgeting, compression, and source diversity belong in `rag/`.
- LangGraph workflow state and graph construction belong in a dedicated `rag/graph/` package once introduced.
- Agent tools belong in a dedicated `rag/tools/` package once introduced; do not mix tool definitions into API routes.
- Request and response contracts belong in `api/schemas.py`.
- HTTP behavior belongs in `api/routes/`.
- New behavior must have a focused test in `tests/` before broad refactoring.
- New architectural decisions belong in `docs/architecture/` or `docs/improvements/`.

## Runtime Data and Generated Files

Keep local databases, uploaded documents, logs, evaluation outputs, caches, and virtual environments out of commits. The existing `.gitignore` is the source of truth for generated data. Do not add committed test fixtures under ignored runtime directories; use a dedicated test fixture path when a fixture must be versioned.

## Development Sequence

1. Identify the owning module and a nearby test.
2. State the smallest behavior change that will prove the hypothesis.
3. Make a focused edit.
4. Run the narrowest relevant test or compile check immediately.
5. Update documentation when behavior or architecture changes.
6. Review `git diff --check` and `git status` before committing.

## Context Engineering Roadmap

Follow [the context-engineering plan](docs/improvements/context-engineering-plan.md) phase by phase. Complete a phase with tests and measurements before starting the next one. Keep AI-agent or LangGraph work behind stable context and validation interfaces.

## Validation Commands

Run commands from the repository root with the project environment active:

```powershell
python -m pytest -q
python -m py_compile api\main.py rag\hybrid_rag.py
```

For a focused change, prefer the narrow test module over the full suite first.
