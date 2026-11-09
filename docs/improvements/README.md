# Improvements and evaluation

These documents describe implemented capabilities and the remaining evaluation work. They replace older links to planning documents that are not present in this repository.

## Current status

- **Implemented:** core retrieval, reranking, query understanding, context management, validation, analytics, and the bounded goal-directed agent.
- **In progress:** evaluation and rollout of the iterative agentic retrieval path and broader evaluation of the goal-directed agent.
- **Default behavior:** `AGENTIC_LOOP_ENABLED` defaults to `false`; set it to `true` to opt in to iterative retrieval. This follows the current [rollout evaluation](../development/phase-2-rollout-evaluation.md).

## Documents

- [Improvement summary](./improvement-summary.md) — current capabilities and next focus.
- [Implementation roadmap](./implementation-roadmap.md) — technical status and next milestone.
- [Strategic improvement plan](./improvement-plan.md) — phase-level status.
- [Context engineering plan](./context-engineering-plan.md) — implementation history and evaluation notes.
- [Agentic retrieval rollout evaluation](../development/phase-2-rollout-evaluation.md) — measured baseline comparison and release gates.
- [Agentic retrieval design notes](../development/phase-2-agentic-loop.md) — historical proposal, not runtime contract.
- [Goal-directed agent architecture](../architecture/agent.md) — current agent tools, bounds, and API.
