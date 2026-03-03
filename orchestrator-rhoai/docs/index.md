# orchestrator-rhoai

Deploy NVIDIA's ToolOrchestra on Red Hat OpenShift AI. A small 8B orchestrator model coordinates specialist tools and models to solve complex tasks more efficiently than monolithic LLMs.

## Quick Links

- [Prerequisites](getting-started/prerequisites.md) — What you need before starting
- [Quick Start](getting-started/quickstart.md) — 15-minute first deployment
- [Architecture Overview](architecture/overview.md) — System design and components
- [Deploy Orchestrator](guides/deploy-orchestrator.md) — Step-by-step model serving

## Getting Started

1. Ensure you have a Red Hat OpenShift AI cluster with GPUs available.
2. Install `oc` and `kustomize`.
3. Clone the repository and run `make check-cluster` to verify prerequisites.
4. Deploy Tier 1 with `make deploy-tier1` and run `make smoke-test`.

See the [Getting Started](getting-started/prerequisites.md) section for detailed instructions.
