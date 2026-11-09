# Testing guide

The test suite lives in `hybrid_rag/tests/`. Run commands from the workspace directory containing `hybrid_rag`, with the project's Python environment active.

## Run tests

```powershell
# Full suite
python -m pytest -q

# Verbose output
python -m pytest -v

# One test module
python -m pytest hybrid_rag/tests/test_rag_agent.py -q

# One test by node ID
python -m pytest hybrid_rag/tests/test_rag_agent.py::test_agent_plans_tools_and_returns_source_citations -q
```

Use `python -m pytest --collect-only -q` to inspect the current collection. Test counts are not repeated in this document because they change as the suite grows.

## Test areas

The suite covers API routes, document and vector-store behavior, retrieval, query and chat flows, context selection, analytics, agent execution, and configuration/policy behavior. Check the current `tests/` directory for the module names and coverage; older test-count tables and timing targets in previous versions of this guide were not maintained.

## Adding tests

- Prefer deterministic tests with mocked external model calls.
- Test observable behavior and boundary conditions, including invalid requests and exhausted budgets.
- Keep performance thresholds in dedicated benchmark tests and document their measurement environment.
- Run the focused module after a change, then run the full suite before reporting broad compatibility.
