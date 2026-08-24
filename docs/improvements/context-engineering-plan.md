# Context Engineering Plan

## Decision

Improve context engineering before introducing an AI agent. The current system already has retrieval, query understanding, memory, reranking, validation, and streaming. Context engineering will make those components produce a reliable, bounded prompt before adding autonomous tool selection.

## Goals

- Build a query-focused, token-bounded context.
- Preserve useful conversation history without starving document context.
- Make source selection and truncation observable.
- Improve answer quality without adding agent planning latency.
- Establish evaluation signals that an eventual agent can reuse.

## Phases

### Phase 1: Budget-Safe Context Assembly

Status: Phase 1 slice complete.

- Separate document budget from system/query/history overhead.
- Prevent a configured reserve from consuming a small explicit budget.
- Preserve the highest-ranked usable documents within the budget.
- Add regression tests for small budgets, history overhead, and truncation.

Completed in the first slice:

- Capped the system reserve at 25% of small explicit budgets.
- Preserved the existing context-manager API and document ordering.
- Added a small-budget regression test.
- Confirmed the query-flow context-budget regression passes.

### Phase 2: Relevance and Diversity

Status: Fourth slice complete.

- Add explicit relevance-score ordering at the context boundary.
- Deduplicate repeated chunks from the same source.
- Apply source and section diversity limits.
- Record selected and dropped document identifiers.

Completed in the first slice:

- Deduplicated retrieved chunks by stable `doc_id`.
- Added a `(source, content)` fallback key for generic documents.
- Preserved first-seen ranking order and distinct chunks from the same source.
- Added regression tests for object and dictionary document shapes.
- Added a configurable two-document-per-source default.
- Preserved distinct sources and added an opt-out with `max_docs_per_source=None`.
- Added `last_selection_report` with selected documents, dropped documents, reasons, and token usage.
- Recorded `duplicate`, `source_limit`, `token_budget`, and `context_overhead` exclusion reasons.
- Ordered candidates by numeric relevance score in descending order with stable ties.
- Preserved retrieval order when scores are missing or unusable.
- Recorded the ordering policy in `last_selection_report`.

### Phase 3: Conversation Context

Status: Fourth slice complete.

- Select relevant prior turns instead of always using the latest fixed window.
- Summarize older turns when history exceeds its budget.
- Keep the current user request and recent constraints intact.

Completed in the first slice:

- Added query-aware history selection with lexical term overlap.
- Preserved the two most recent messages for conversational continuity.
- Enforced the configured conversation-context word budget.
- Used the same selected history for multi-turn prompt construction and document budgeting.
- Added regression tests for relevant older turns, recent turns, and budget limits.
- Compressed older selected turns into a bounded deterministic summary.
- Preserved the two most recent messages verbatim when the budget allows.
- Added regression coverage for summary markers and the history word budget.

### Phase 4: Evaluation and Optimization

Status: Seventh slice complete.

- Measure context precision, context recall, answer faithfulness, and time-to-first-token.
- Compare full, truncated, compressed, and cached contexts.
- Add per-stage latency metrics and representative evaluation queries.

Completed in the first slice:

- Added Prometheus time-to-first-chunk instrumentation for query streaming.
- Added Prometheus total stream-duration instrumentation.
- Recorded total duration on successful completion, errors, and disconnects.
- Added metric registration regression coverage.
- Added deterministic context-selection evaluation metrics.
- Measured selected and dropped counts, duplicate-drop rate, source diversity, budget utilization, and ordering policy.
- Corrected selection-report token usage for truncated documents.
- Added a repeatable JSONL context-evaluation CLI.
- Added tracked representative cases for duplicate filtering, source diversity, score ordering, and small budgets.
- Established a baseline run: 1.5 selected documents, 1.0 dropped document, 0.5 duplicate-drop rate, 1.5 source diversity, and 0.43 budget utilization per case on average.
- Added answer-quality evaluation from expected source coverage and answer-hint coverage.
- Reported per-case answer-quality scores and an aggregate answer-quality score.
- Added deterministic answer-faithfulness scoring against selected context terms.
- Added expected answers to representative cases and included faithfulness in the CLI baseline.
- Added model-free latency benchmarking for full, truncated, compressed, and cached context variants.
- Reported per-case average latency and token counts plus aggregate variant latency baselines.
- Added analytics correlation for query, retrieval, and first-token latency by cache status.
- Persisted first-token and total stream duration fields from the core stream wrapper.
- Added an `/analytics/latency` endpoint for stage-level latency summaries.

### Phase 5: Constrained Agent Readiness

Status: Sixth slice complete.

- Define stable retrieval, context, and validation interfaces.
- Expose narrowly scoped tools such as document search and analytics lookup.
- Add an agent only for tasks requiring multi-step tool selection.
- Enforce budgets, allowed tools, source requirements, and failure limits.

Completed in the first slice:

- Added bounded document search, context selection, and response validation tool wrappers.
- Preserved the existing retrieval, context, and validation implementations behind stable result contracts.
- Enforced a configurable maximum `top_k` for document search.
- Added a constrained executor with tool allowlists, required-source checks, and failure limits.
- Preserved failure counts across calls so repeated tool errors stop further execution.
- Added per-request call and result-size budgets to the constrained executor.
- Added a stable structured `ToolExecutionError` contract for tool failures and policy rejections.
- Added a bounded workflow composing search, context selection, and response validation.
- Applied required-source checks across both tool outputs and validation inputs.
- Added a validated `WorkflowPolicy` contract and executor factory for explicit policy configuration.
- Added a caller-facing workflow service and `/api/query/workflow` endpoint.
- Created a fresh per-request executor so call and failure budgets do not leak across callers.
- Preserved structured tool-policy errors at the HTTP boundary.

## Development Order

1. Complete one phase with tests and measurements.
2. Run the focused tests before starting the next phase.
3. Update this document with actual results and remaining risks.
4. Avoid introducing agent behavior until context quality and latency are measurable.

## Current Slice

Phase 4 implementation is complete for deterministic evaluation and persisted latency observability. Phase 5 now has a caller-facing service boundary; the next slice can add narrowly scoped agent orchestration on top of that contract.
