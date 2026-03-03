# ADR-0002: Open Models Only

## Status
Accepted

## Date
2026-02-23

## Context

The NVIDIA ToolOrchestra paper uses GPT-5 as the premium backend for `reasoner-1`, `search-1`, and `answer-1`. An open-source project cannot depend on proprietary API access. We need a strategy that keeps the project self-contained and reproducible.

## Decision

Replace GPT-5 and GPT-5 Mini with open-weight models. Use DeepSeek-R1-Distill-Qwen-32B for reasoning, Llama-3.3-70B-Instruct for general answering, and Qwen3-32B for tier-appropriate fallbacks.

## Consequences

- The project is fully self-contained with no proprietary API dependency.
- Benchmark scores will be lower than the paper reports; we document this transparently.
- This quantifies the gap between open and proprietary models.
- Customers with GPT-5 access can swap in commercial APIs via the model-mapping ConfigMap.
- The orchestrator uses abstract tool names—it does not know which concrete model backs each tool.

## Alternatives Considered

- **Hybrid approach (open + proprietary)**: Would fragment the user experience and require API keys.
- **Exact paper replication**: Not feasible for an open-source project without proprietary access.
