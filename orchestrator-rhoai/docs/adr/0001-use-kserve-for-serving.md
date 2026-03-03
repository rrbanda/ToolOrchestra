# ADR-0001: Use KServe for Serving

## Status
Accepted

## Date
2026-02-23

## Context

ToolOrchestra serves 10+ models via raw vLLM processes managed by SLURM in the original NVIDIA implementation. We need a Kubernetes-native equivalent for Red Hat OpenShift AI that provides production-grade serving, autoscaling, and API compatibility.

## Decision

Use KServe `InferenceService` with the vLLM `ServingRuntime` for all models (orchestrator and specialists).

## Consequences

- Models are deployed declaratively via InferenceService manifests.
- KServe provides autoscaling, canary rollouts, and health checks out of the box.
- vLLM ServingRuntime is pre-packaged in RHOAI 3.2.
- OpenAI-compatible `/v1/chat/completions` API matches the existing LLM_CALL interface.
- This decision demonstrates RHOAI's core model serving capability.
- Model endpoints are discoverable and routable within the cluster.

## Alternatives Considered

- **Raw vLLM Deployments**: More control over deployment details, but lose KServe autoscaling, routing, and ecosystem integration.
- **TGI (Text Generation Inference)**: Not as strong on tool-call support compared to vLLM.
- **Triton Inference Server**: Overkill for this use case; more complex configuration for LLM serving.
