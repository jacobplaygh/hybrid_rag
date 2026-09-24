# Phase 2 Retrieval Rollout Evaluation

**Date:** September 24, 2026
**Dataset:** `tools/agentic_eval_queries.jsonl` (15 cases)
**Command:** `python tools/ab_testing.py --dataset tools/agentic_eval_queries.jsonl --limit 15`

## Results

| Metric | Baseline | Agentic | Delta |
|---|---:|---:|---:|
| Success rate | 100% | 80% | -20 pp |
| Mean confidence | 0.526 | 0.648 | +0.122 |
| Mean latency | 160 ms | 220 ms | +60 ms |
| Context relevance | 0.390 | 0.197 | -0.193 |
| Groundedness | 0.714 | 0.667 | -0.048 |
| Answer relevance | 0.044 | 0.083 | +0.039 |

## Decision

Do not make agentic retrieval the unconditional default yet. Keep `AGENTIC_LOOP_ENABLED` configurable and use the baseline path as the production-safe default until the agentic path preserves success rate and does not regress context relevance or groundedness.

The confidence improvement is useful, but confidence alone is not a sufficient rollout signal. The current evaluation also exposes a measurement weakness: answer-level relevance scores are sparse, so the dataset and scoring should be improved before using this as a release gate.

## Follow-up Gates

- Add explicit regression cases for low-confidence agentic outcomes.
- Require agentic success rate to be at least baseline success rate.
- Require no regression in mean groundedness and context relevance beyond the agreed tolerance.
- Re-run the full dataset after GraphRAG indexing and retrieval hardening.
