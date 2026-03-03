# ADR-0004: Gateway Pattern

## Status
Accepted

## Date
2026-02-23

## Context

The orchestrator model generates tool calls (e.g., `reasoner-1`, `answer-2`, `search-3`). Something must intercept those calls, route them to the correct specialist model or tool, execute them, and feed results back for multi-turn orchestration. This logic could live in several places.

## Decision

Build a dedicated FastAPI gateway service—the Orchestrator Gateway—that manages the multi-turn orchestration loop and tool routing.

## Consequences

- Clean separation between model serving (KServe) and orchestration logic.
- Gateway handles the multi-turn loop, retries, timeouts, and cost tracking.
- Model mapping is externalized in a ConfigMap; no code changes required to swap models.
- The gateway is the only component that understands the tool protocol; individual models simply serve completions.
- Easier to add metrics, tracing, and observability at the orchestration layer.
- Gateway can be scaled independently of model serving.

## Alternatives Considered

- **LangChain / LlamaIndex**: Adds heavy dependencies and hides the orchestration logic; less transparent for a reference implementation.
- **Embedding in KServe transformer**: Too tightly coupled to serving; complicates deployment and model updates.
- **Jupyter notebook**: Not production-grade; unsuitable for serving requests.
