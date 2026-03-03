# orchestrator-rhoai

A production-ready reference implementation that deploys NVIDIA's ToolOrchestra system on Red Hat OpenShift AI.

[![CI](https://img.shields.io/badge/CI-status-lightgrey)](https://github.com/orchestrator-rhoai/orchestrator-rhoai/actions)
[![Release](https://img.shields.io/badge/release-v0.1.0-lightgrey)](https://github.com/orchestrator-rhoai/orchestrator-rhoai/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-lightgrey)](https://orchestrator-rhoai.github.io/)

## What Is This?

**ToolOrchestra** is a small 8B-parameter orchestrator model that coordinates specialist tools and models to solve complex tasks. Instead of routing every sub-step through a monolithic LLM, it learns to delegate each step to the cheapest model capable of handling it—achieving better results on the HLE benchmark than GPT-5 at approximately 30% of the cost.

The approach is described in the NVIDIA paper "[ToolOrchestra: Orchestrating Large Language Models for Cost-Effective Task Solving](https://arxiv.org/abs/2511.21689)" (arXiv:2511.21689). The model is trained via reinforcement learning to make tool-call decisions that optimize both quality and cost.

This project—**orchestrator-rhoai**—provides the Kubernetes manifests, gateway service, and documentation to deploy ToolOrchestra on Red Hat OpenShift AI. It replaces proprietary model backends (e.g., GPT-5) with open-weight models and demonstrates 15+ RHOAI platform features in a real-world AI system.

## Architecture

```
                    User
                      │
                      ▼
              ┌───────────────┐
              │   Gateway     │
              │  (FastAPI)    │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Orchestrator  │
              │  Model (8B)   │
              └───────┬───────┘
                      │ tool calls
          ┌───────────┼───────────┬─────────────┐
          ▼           ▼           ▼             ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐  ┌─────────────┐
    │Reasoner │ │Answerer │ │ Search   │  │ Code         │
    │(32B/70B)│ │(32B/70B)│ │(32B)     │  │ Execution    │
    └─────────┘ └─────────┘ └─────────┘  └───────────────┘
          │           │           │
          └───────────┴───────────┴───────────────► Final Answer
```

## Quick Start

1. **Prerequisites**: Red Hat OpenShift AI cluster with GPUs, `oc` CLI, and `kustomize` installed.
2. **Clone the repo**: `git clone https://github.com/orchestrator-rhoai/orchestrator-rhoai.git && cd orchestrator-rhoai`
3. **Verify cluster**: `make check-cluster`
4. **Deploy Tier 1**: `make deploy-tier1`
5. **Smoke test**: `make smoke-test`

## Deployment Tiers

| Tier | GPUs | Models | Use Case |
|------|------|--------|----------|
| Tier 1 | 2 | 2 (Orchestrator-8B, Qwen3-32B) | Demo, POC |
| Tier 2 | 4 | 4 (+ DeepSeek-R1, Qwen-Coder) | Development |
| Tier 3 | 8–12 | 7 (all specialists) | Production |

## RHOAI Features Demonstrated

- KServe InferenceService
- vLLM ServingRuntime
- Multi-model serving
- Model Registry
- Workbenches
- KubeRay (RayCluster, RayJob)
- Training Operator
- RHOAI Pipelines
- TrustyAI
- Data Connections
- KServe Autoscaling
- OpenShift Monitoring
- Service Mesh
- OpenShift Pipelines
- GPU Operator

## Project Structure

```
orchestrator-rhoai/
├── manifests/       # Kustomize base and serving configs
├── profiles/        # Tier 1/2/3 overlay profiles
├── src/             # Gateway, retrieval, sandbox, evaluation
├── scripts/         # setup-cluster, smoke-test, etc.
└── docs/            # MkDocs documentation
```

## Documentation

Full documentation is available at [https://orchestrator-rhoai.github.io/](https://orchestrator-rhoai.github.io/) (placeholder). Run `make docs-serve` to browse locally.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, branching, and pull request guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- **NVIDIA ToolOrchestra team** for the original research and model.
- **Paper**: [ToolOrchestra: Orchestrating Large Language Models for Cost-Effective Task Solving](https://arxiv.org/abs/2511.21689) (arXiv:2511.21689).
- **Upstream repo**: [https://github.com/NVlabs/ToolOrchestra](https://github.com/NVlabs/ToolOrchestra)
