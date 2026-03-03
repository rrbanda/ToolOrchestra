# ADR-0003: Tiered Deployment

## Status
Accepted

## Date
2026-02-23

## Context

GPU resources vary dramatically between customers—from small demos to production deployments. The project must work at multiple scales without requiring separate codebases or complex configuration.

## Decision

Define three Kustomize overlay profiles (Tier 1, Tier 2, Tier 3) with increasing GPU and model counts.

## Details

| Tier | GPUs | Models | Use Case |
|------|------|--------|----------|
| Tier 1 | 2 | 2 (Orchestrator-8B, Qwen3-32B) | Demo, POC |
| Tier 2 | 4 | 4 (+ DeepSeek-R1, Qwen-Coder) | Development |
| Tier 3 | 8–12 | 7 (all specialists) | Production |

Tier 1 provides a minimal viable deployment for demonstrations. Tier 2 adds reasoning and coding specialists. Tier 3 includes all models (Llama-70B, Qwen-Math-72B, Qwen-Math-7B) for full benchmark and production use.

## Consequences

- Customers can choose a profile matching their GPU budget.
- Overlays are applied via `make deploy-tier1`, `deploy-tier2`, or `deploy-tier3`.
- Tier-specific patches adjust resource limits, replica counts, and model inclusion.
- Simplifies documentation and support by having clear, named configurations.

## Alternatives Considered

- **Single monolithic deployment**: Would exclude users without 8+ GPUs.
- **Fully parameterized Helm**: Adds complexity; Kustomize overlays are sufficient.
