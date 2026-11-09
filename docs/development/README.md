# Development guide

## Setup and tests

These commands assume the current directory is the workspace directory containing the `hybrid_rag` project. This keeps the configured relative upload and vector-store paths rooted in the expected workspace:

```powershell
python -m pip install -r .\hybrid_rag\requirements_fastapi.txt
$env:PYTHONPATH = (Join-Path $PWD 'hybrid_rag')
python -m pytest -q
```

Start the API locally from the same directory:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## Project areas

- `api/` — FastAPI application, schemas, configuration, and routes.
- `rag/` — retrieval, orchestration, context management, validation, analytics, and agent logic.
- `data/` — vector-store implementations.
- `observability/` — metrics and tracing integrations.
- `tests/` — automated tests.
- `docs/` — setup, architecture, development, and improvement documentation.

## Making a change

1. Find the existing implementation and tests for the behavior.
2. Make a focused change and add or update regression tests.
3. Run focused tests, then `python -m pytest -q` when practical.
4. Update the relevant documentation when an API, configuration default, or user-facing behavior changes.

For repository-specific contribution conventions, see [CONTRIBUTING.md](../../CONTRIBUTING.md). For current evaluation status, see the [improvement summary](../improvements/improvement-summary.md).
