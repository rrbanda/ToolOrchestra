# ADR-0005: Kustomize over Helm

## Status
Accepted

## Date
2026-02-23

## Context

We need to package Kubernetes manifests for different environments (Tier 1/2/3) and potentially multiple clusters. Both Kustomize and Helm are common choices for Kubernetes deployment management.

## Decision

Use Kustomize with base + overlay profiles instead of Helm.

## Consequences

- Kustomize is built into `oc` and `kubectl`—no additional tooling required.
- Overlays map cleanly to our tier concept (base + tier1-minimal, tier2-standard, tier3-full).
- Manifests remain plain YAML, which is easier to review, audit, and learn from.
- Aligns with RHOAI documentation and common OpenShift patterns.
- Contributors can understand and modify manifests without learning Helm templating.
- Profiles can be validated with `kustomize build profiles/tier1-minimal`.

## Alternatives Considered

- **Helm**: More powerful templating (conditionals, loops, includes), but adds complexity and a learning curve for a reference project meant to be educational. Helm charts can be more opaque to newcomers.
