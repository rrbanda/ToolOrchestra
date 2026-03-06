# ToolOrchestra on OpenShift AI

## Production-Grade Open Source Project Plan

> **Version**: 0.1.0-draft
> **Status**: Planning
> **Last Updated**: 2026-02-23
> **License**: Apache 2.0 (inheriting from upstream NVIDIA ToolOrchestra)

---

## Table of Contents

- [Part I — Project Identity & Vision](#part-i--project-identity--vision)
- [Part II — Open Source Foundations](#part-ii--open-source-foundations)
- [Part III — Repository Standards](#part-iii--repository-standards)
- [Part IV — Architecture & Technical Decisions](#part-iv--architecture--technical-decisions)
- [Part V — Implementation Phases](#part-v--implementation-phases)
- [Part VI — Testing Strategy](#part-vi--testing-strategy)
- [Part VII — Release & Versioning Strategy](#part-vii--release--versioning-strategy)
- [Part VIII — Documentation Strategy](#part-viii--documentation-strategy)
- [Part IX — Community & Adoption](#part-ix--community--adoption)
- [Part X — Risk Management](#part-x--risk-management)

---

# Part I — Project Identity & Vision

## 1.1 Project Name

**`orchestrator-rhoai`**

Descriptive, searchable, and immediately conveys what it is: the ToolOrchestra
system running on Red Hat OpenShift AI.

## 1.2 One-Line Description

> A production-ready reference implementation that deploys NVIDIA's
> ToolOrchestra system on Red Hat OpenShift AI, demonstrating how a small
> 8B-parameter orchestrator model coordinates specialist tools and models to
> solve complex tasks more efficiently than monolithic LLMs.

## 1.3 Problem Statement

Large language models are powerful but expensive. A single GPT-5 call costs
\$1.25/M input tokens and may take 96 seconds. Most tasks don't need that level
of capability for every step. ToolOrchestra proves that a small 8B model can
learn to route each sub-step of a complex task to the cheapest model that can
handle it — achieving better results at 30% of the cost.

However, the original implementation is tightly coupled to NVIDIA's internal
SLURM-based HPC infrastructure. There is no guide for deploying it on
Kubernetes-native AI platforms.

## 1.4 What This Project Delivers

1. **Kubernetes manifests** to deploy the full ToolOrchestra system on
   OpenShift AI
2. **An orchestrator gateway** service that manages multi-turn tool-call
   loops between the orchestrator model and specialist models
3. **Adapted evaluation scripts** that run official benchmarks (HLE, FRAMES,
   τ²-Bench) against the RHOAI deployment
4. **A training pipeline** that replaces SLURM with KubeRay for RL training
5. **Documentation** covering architecture, setup, customization, and results
6. **A showcase** of 13+ RHOAI platform features in a real-world AI system

## 1.5 Target Audience

- **AI/ML engineers** evaluating OpenShift AI for production model serving
- **Platform engineers** building multi-model AI systems on Kubernetes
- **Researchers** reproducing ToolOrchestra results in a cloud-native
  environment
- **Red Hat partners and customers** looking for RHOAI reference
  architectures

## 1.6 Non-Goals

- We are **not** forking or modifying the upstream ToolOrchestra training
  algorithms. We adapt the deployment and integration layers.
- We are **not** competing with the paper's benchmark numbers (we use open
  models instead of GPT-5, so scores will differ).
- We are **not** building a general-purpose model serving platform — this is
  a specific reference implementation.

---

# Part II — Open Source Foundations

## 2.1 License

**Apache License 2.0** — matching the upstream NVIDIA ToolOrchestra project.

All original code in this project is licensed under Apache 2.0. Third-party
dependencies and model weights have their own licenses, documented in a
`THIRD_PARTY_LICENSES.md` file.

### Model License Audit

Every model used in the project must pass a license review:

| Model                                  | License              | Commercial Use | Status    |
|----------------------------------------|----------------------|----------------|-----------|
| nvidia/Nemotron-Orchestrator-8B        | NVIDIA Open Model    | Check terms    | Review    |
| Qwen/Qwen3-8B (base)                  | Apache 2.0           | Yes            | Approved  |
| Qwen/Qwen3-32B                        | Apache 2.0           | Yes            | Approved  |
| Qwen/Qwen2.5-Coder-32B-Instruct       | Apache 2.0           | Yes            | Approved  |
| Qwen/Qwen2.5-Math-72B-Instruct        | Apache 2.0           | Yes            | Approved  |
| Qwen/Qwen2.5-Math-7B-Instruct         | Apache 2.0           | Yes            | Approved  |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | MIT               | Yes            | Approved  |
| meta-llama/Llama-3.3-70B-Instruct     | Llama 3.3 Community  | Conditional    | Review    |

**Action**: Complete license review for all "Review" status models before
Phase 1 deployment.

## 2.2 Governance

### GOVERNANCE.md

```
Project Type: Reference Implementation
Decision Making: Maintainer consensus (lazy consensus for minor changes,
                 explicit approval for architecture changes)
Maintainers: Listed in MAINTAINERS.md
Contribution Path: Contributor → Reviewer → Maintainer
```

### MAINTAINERS.md

Lists individuals with merge authority, their areas of ownership:

| Area                    | Maintainer(s)        |
|-------------------------|----------------------|
| Kubernetes manifests    | TBD                  |
| Orchestrator gateway    | TBD                  |
| Evaluation scripts      | TBD                  |
| Training pipeline       | TBD                  |
| Documentation           | TBD                  |
| CI/CD                   | TBD                  |

### CODEOWNERS

Maps directories to responsible maintainers for automatic PR review
assignment.

```
# Kubernetes manifests
/manifests/         @maintainer-platform
/profiles/          @maintainer-platform

# Application code
/src/               @maintainer-app
/evaluation/        @maintainer-eval
/training/          @maintainer-training

# Documentation
/docs/              @maintainer-docs
*.md                @maintainer-docs

# CI/CD
/.github/           @maintainer-cicd
/Makefile           @maintainer-cicd
```

## 2.3 Code of Conduct

Adopt the **Contributor Covenant v2.1** — the industry standard for open
source projects. Include as `CODE_OF_CONDUCT.md` in the repo root.

## 2.4 Security Policy

### SECURITY.md

```markdown
## Reporting Security Issues

Do NOT open public issues for security vulnerabilities.

Email: [security contact TBD]

We will acknowledge within 48 hours and provide a detailed response
within 7 days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x     | Yes       |

## Security Practices

- All container images are scanned with Trivy before release
- Dependencies are monitored via Dependabot / Renovate
- No secrets are committed to the repository
- Model weights are never included in container images
- Network policies restrict inter-service communication
```

## 2.5 Contributing Guide

### CONTRIBUTING.md

Covers:

1. **Development environment setup** — prerequisites, local development
2. **Branching strategy** — (see §7.3)
3. **Commit message format** — Conventional Commits
4. **Pull request process** — template, review requirements, CI checks
5. **Issue process** — how to file bugs, request features, ask questions
6. **Code style** — linting, formatting, type checking
7. **Testing requirements** — what tests are expected for each PR
8. **Documentation requirements** — what docs to update for each change
9. **DCO sign-off** — Developer Certificate of Origin required

---

# Part III — Repository Standards

## 3.1 Repository Structure

```
orchestrator-rhoai/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yaml                  # Structured bug report
│   │   ├── feature_request.yaml             # Feature request
│   │   ├── question.yaml                    # Support question
│   │   └── config.yml                       # Template chooser config
│   ├── PULL_REQUEST_TEMPLATE.md             # PR template with checklist
│   ├── workflows/
│   │   ├── ci.yaml                          # Lint + test on every PR
│   │   ├── container-build.yaml             # Build & scan container images
│   │   ├── docs.yaml                        # Build & deploy documentation
│   │   ├── release.yaml                     # Automated release workflow
│   │   └── dependency-review.yaml           # License & vulnerability check
│   ├── CODEOWNERS
│   ├── dependabot.yml                       # Dependency update automation
│   └── labels.yml                           # Standardized issue/PR labels
│
├── docs/
│   ├── mkdocs.yml                           # MkDocs Material configuration
│   ├── index.md                             # Landing page
│   ├── getting-started/
│   │   ├── prerequisites.md                 # What you need before starting
│   │   ├── quickstart.md                    # 15-minute first deployment
│   │   └── cluster-setup.md                 # RHOAI cluster preparation
│   ├── architecture/
│   │   ├── overview.md                      # System architecture
│   │   ├── model-strategy.md                # Model selection rationale
│   │   ├── tool-routing.md                  # How tool calls are routed
│   │   └── data-flow.md                     # Request lifecycle
│   ├── guides/
│   │   ├── deploy-orchestrator.md           # Phase 1 step-by-step
│   │   ├── deploy-specialists.md            # Phase 2 step-by-step
│   │   ├── configure-tools.md               # Phase 3 step-by-step
│   │   ├── run-evaluation.md                # Phase 5 step-by-step
│   │   ├── training-pipeline.md             # Phase 6 step-by-step
│   │   └── production-hardening.md          # Phase 7 step-by-step
│   ├── reference/
│   │   ├── api.md                           # Gateway API reference
│   │   ├── configuration.md                 # All configurable parameters
│   │   ├── model-mapping.md                 # Abstract → real model mapping
│   │   └── troubleshooting.md               # Common issues and fixes
│   ├── adr/                                 # Architecture Decision Records
│   │   ├── 0001-use-kserve-for-serving.md
│   │   ├── 0002-open-models-only.md
│   │   ├── 0003-tiered-deployment.md
│   │   ├── 0004-gateway-pattern.md
│   │   └── template.md
│   └── blog/                                # Technical blog posts
│       └── 001-why-orchestration.md
│
├── manifests/
│   ├── base/                                # Shared Kustomize base
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── rbac/
│   │   │   ├── service-account.yaml
│   │   │   ├── role.yaml
│   │   │   └── rolebinding.yaml
│   │   ├── config/
│   │   │   ├── model-mapping-configmap.yaml
│   │   │   └── tool-definitions-configmap.yaml
│   │   ├── secrets/
│   │   │   └── secrets-template.yaml        # Template only, no real values
│   │   ├── storage/
│   │   │   ├── model-weights-pvc.yaml
│   │   │   └── shared-data-pvc.yaml
│   │   └── network/
│   │       └── network-policy.yaml
│   │
│   ├── serving/                             # KServe model deployments
│   │   ├── orchestrator/
│   │   │   ├── kustomization.yaml
│   │   │   ├── serving-runtime.yaml
│   │   │   └── inference-service.yaml
│   │   ├── qwen3-32b/
│   │   │   ├── kustomization.yaml
│   │   │   ├── serving-runtime.yaml
│   │   │   └── inference-service.yaml
│   │   ├── deepseek-r1-32b/
│   │   │   └── ...
│   │   ├── qwen-coder-32b/
│   │   │   └── ...
│   │   ├── llama-70b/
│   │   │   └── ...
│   │   ├── qwen-math-72b/
│   │   │   └── ...
│   │   └── qwen-math-7b/
│   │       └── ...
│   │
│   ├── services/                            # Supporting services
│   │   ├── gateway/
│   │   │   ├── kustomization.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── route.yaml
│   │   │   └── hpa.yaml
│   │   ├── retrieval/
│   │   │   ├── kustomization.yaml
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── configmap.yaml
│   │   └── code-sandbox/
│   │       ├── kustomization.yaml
│   │       ├── deployment.yaml
│   │       ├── service.yaml
│   │       └── security-context.yaml
│   │
│   ├── observability/                       # Monitoring & alerting
│   │   ├── kustomization.yaml
│   │   ├── service-monitor.yaml
│   │   ├── prometheus-rules.yaml
│   │   ├── grafana-dashboard.yaml
│   │   └── alerting-rules.yaml
│   │
│   ├── training/                            # Training pipeline resources
│   │   ├── kustomization.yaml
│   │   ├── ray-cluster.yaml
│   │   ├── ray-job.yaml
│   │   └── training-pvc.yaml
│   │
│   └── pipelines/                           # Kubeflow / Tekton pipelines
│       ├── deploy-pipeline.yaml
│       ├── evaluation-pipeline.yaml
│       └── retrain-pipeline.yaml
│
├── profiles/                                # Kustomize overlays per tier
│   ├── tier1-minimal/
│   │   ├── kustomization.yaml               # Only orchestrator + qwen3-32b
│   │   └── patches/
│   ├── tier2-standard/
│   │   ├── kustomization.yaml               # + deepseek, coder
│   │   └── patches/
│   └── tier3-full/
│       ├── kustomization.yaml               # All models + training
│       └── patches/
│
├── src/
│   ├── gateway/                             # Orchestrator Gateway service
│   │   ├── Containerfile
│   │   ├── pyproject.toml
│   │   ├── gateway/
│   │   │   ├── __init__.py
│   │   │   ├── app.py                       # FastAPI application
│   │   │   ├── config.py                    # Configuration loading
│   │   │   ├── orchestrator.py              # Multi-turn orchestration loop
│   │   │   ├── tool_router.py               # Routes tool calls to models
│   │   │   ├── model_client.py              # OpenAI-compatible client
│   │   │   ├── metrics.py                   # Prometheus metrics
│   │   │   ├── schemas.py                   # Pydantic request/response models
│   │   │   └── middleware.py                # Logging, tracing, error handling
│   │   └── tests/
│   │       ├── conftest.py
│   │       ├── test_app.py
│   │       ├── test_orchestrator.py
│   │       ├── test_tool_router.py
│   │       └── test_model_client.py
│   │
│   ├── retrieval/                           # Retrieval service
│   │   ├── Containerfile
│   │   ├── pyproject.toml
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   ├── faiss_index.py
│   │   │   ├── tavily_client.py
│   │   │   └── config.py
│   │   └── tests/
│   │       └── ...
│   │
│   ├── code_sandbox/                        # Code execution sandbox
│   │   ├── Containerfile
│   │   ├── pyproject.toml
│   │   ├── sandbox/
│   │   │   ├── __init__.py
│   │   │   ├── app.py
│   │   │   ├── executor.py
│   │   │   └── config.py
│   │   └── tests/
│   │       └── ...
│   │
│   └── evaluation/                          # Adapted evaluation scripts
│       ├── pyproject.toml
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── run_hle.py
│       │   ├── run_frames.py
│       │   ├── run_tau2.py
│       │   └── reporter.py
│       └── tests/
│           └── ...
│
├── containers/                              # Shared Containerfile resources
│   ├── base-python/
│   │   └── Containerfile                    # Common Python base image
│   └── training/
│       └── Containerfile                    # Training-specific image
│
├── scripts/
│   ├── setup-cluster.sh                     # Verify RHOAI prerequisites
│   ├── download-models.sh                   # Download model weights
│   ├── smoke-test.sh                        # Post-deployment health check
│   ├── generate-sbom.sh                     # Software Bill of Materials
│   └── rotate-secrets.sh                    # Secret rotation helper
│
├── hack/                                    # Developer utilities
│   ├── local-dev.sh                         # Local development setup
│   ├── kind-cluster.sh                      # Kind cluster for local testing
│   └── mock-models.sh                       # Mock model endpoints for testing
│
├── examples/
│   ├── curl-examples.sh                     # API usage examples
│   ├── python-client.py                     # Python client example
│   └── custom-tool-mapping.yaml             # Example custom model mapping
│
├── .gitignore
├── .editorconfig                            # Consistent formatting across editors
├── .pre-commit-config.yaml                  # Pre-commit hooks
├── .hadolint.yaml                           # Dockerfile linting config
├── pyproject.toml                           # Root Python project config
├── Makefile                                 # Top-level developer commands
├── CHANGELOG.md                             # Keep a Changelog format
├── LICENSE                                  # Apache 2.0
├── THIRD_PARTY_LICENSES.md                  # Third-party license inventory
├── CODE_OF_CONDUCT.md                       # Contributor Covenant v2.1
├── CONTRIBUTING.md                          # How to contribute
├── GOVERNANCE.md                            # Project governance
├── SECURITY.md                              # Security policy
├── MAINTAINERS.md                           # Maintainer list
└── README.md                                # Project overview & quick start
```

## 3.2 Branching Strategy

**Trunk-based development** with short-lived feature branches:

```
main (protected)
  ├── feat/phase-1-orchestrator-serving
  ├── feat/phase-2-specialist-models
  ├── fix/vllm-tool-call-parser
  ├── docs/architecture-overview
  └── release/v0.1.0
```

**Rules**:
- `main` is always deployable
- All changes go through pull requests
- PRs require: 1 approval + all CI checks passing
- Squash merge to main (clean linear history)
- Release branches cut from main, tagged with semver

## 3.3 Commit Message Convention

**Conventional Commits v1.0.0**:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
Signed-off-by: Name <email>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`, `ci`,
`chore`, `perf`

Scopes: `gateway`, `retrieval`, `sandbox`, `manifests`, `evaluation`,
`training`, `docs`, `ci`

Examples:
```
feat(gateway): add multi-turn orchestration loop
fix(manifests): correct GPU resource limits for 70B models
docs(architecture): add tool routing sequence diagram
ci: add container image vulnerability scanning
```

## 3.4 Issue & PR Labels

| Label              | Color   | Description                            |
|--------------------|---------|----------------------------------------|
| `phase/0`–`phase/7`| blue   | Which implementation phase             |
| `tier/1`–`tier/3`  | purple  | Which deployment tier                  |
| `type/bug`         | red     | Something isn't working                |
| `type/feature`     | green   | New functionality                      |
| `type/docs`        | cyan    | Documentation improvement              |
| `type/infra`       | orange  | CI/CD, build, deployment               |
| `priority/critical`| red     | Must fix immediately                   |
| `priority/high`    | orange  | Important, next sprint                 |
| `priority/medium`  | yellow  | Should be done soon                    |
| `priority/low`     | grey    | Nice to have                           |
| `good-first-issue` | green   | Good for newcomers                     |
| `help-wanted`      | green   | Community contributions welcome        |
| `rhoai-feature`    | purple  | Demonstrates an RHOAI capability       |

## 3.5 CI/CD Pipeline

### On Every Pull Request (`ci.yaml`)

```
┌──────────────────────────────────────────────┐
│                  PR Opened                    │
└──────────┬──────────┬──────────┬─────────────┘
           │          │          │
     ┌─────▼────┐ ┌───▼───┐ ┌───▼──────────┐
     │  Lint &   │ │ Unit  │ │  Manifest    │
     │  Format   │ │ Tests │ │  Validation  │
     │           │ │       │ │  (kustomize  │
     │ - ruff    │ │pytest │ │   build)     │
     │ - mypy    │ │       │ │              │
     │ - hadolint│ │       │ │ - kubeconform│
     └─────┬─────┘ └───┬───┘ └───┬──────────┘
           │          │          │
     ┌─────▼──────────▼──────────▼─────────────┐
     │           All Checks Pass                │
     │      → Ready for review                  │
     └─────────────────────────────────────────┘
```

### On Merge to Main (`container-build.yaml`)

```
┌──────────────────────────────────────────────┐
│              Merged to main                   │
└──────────┬──────────┬────────────────────────┘
           │          │
   ┌───────▼──────┐ ┌─▼───────────────────┐
   │ Build Images │ │ Scan Images         │
   │              │ │                     │
   │ - gateway    │ │ - Trivy vuln scan   │
   │ - retrieval  │ │ - Syft SBOM gen     │
   │ - sandbox    │ │ - cosign signing    │
   │ - training   │ │                     │
   └───────┬──────┘ └─┬───────────────────┘
           │          │
   ┌───────▼──────────▼───────────────────┐
   │  Push to container registry          │
   │  (ghcr.io or quay.io)               │
   │  Tag: sha-<commit>, latest           │
   └──────────────────────────────────────┘
```

### On Release Tag (`release.yaml`)

```
┌──────────────────────────────────────────────┐
│           Tag: v0.1.0 pushed                  │
└──────────┬───────────────────────────────────┘
           │
   ┌───────▼──────────────────────────────┐
   │ Build & push images with release tag │
   │ Generate SBOM                        │
   │ Create GitHub Release with:          │
   │   - Changelog since last release     │
   │   - Container image references       │
   │   - SBOM attachments                 │
   │   - Deployment instructions          │
   └──────────────────────────────────────┘
```

## 3.6 Code Quality Standards

### Python

| Tool        | Purpose              | Config File          |
|-------------|----------------------|----------------------|
| **ruff**    | Linting + formatting | `pyproject.toml`     |
| **mypy**    | Type checking        | `pyproject.toml`     |
| **pytest**  | Testing              | `pyproject.toml`     |
| **coverage**| Code coverage ≥80%   | `pyproject.toml`     |

### Kubernetes Manifests

| Tool            | Purpose                      |
|-----------------|------------------------------|
| **kustomize**   | Overlay management           |
| **kubeconform** | Schema validation            |
| **kube-linter** | Best practice enforcement    |
| **yamllint**    | YAML formatting              |

### Container Images

| Tool          | Purpose                        |
|---------------|--------------------------------|
| **hadolint**  | Dockerfile best practices      |
| **Trivy**     | Vulnerability scanning         |
| **Syft**      | SBOM generation                |
| **cosign**    | Image signing                  |

## 3.7 Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: detect-private-key
      - id: no-commit-to-branch
        args: ['--branch', 'main']
  - repo: https://github.com/hadolint/hadolint
    hooks:
      - id: hadolint
  - repo: https://github.com/compilerla/conventional-pre-commit
    hooks:
      - id: conventional-pre-commit
```

## 3.8 Makefile

```makefile
.PHONY: help lint test build deploy clean

help:              ## Show this help
lint:              ## Run all linters (ruff, mypy, hadolint, yamllint, kubeconform)
test:              ## Run all tests with coverage
test-unit:         ## Run unit tests only
test-integration:  ## Run integration tests (requires cluster)
build:             ## Build all container images
build-gateway:     ## Build gateway image
build-retrieval:   ## Build retrieval image
build-sandbox:     ## Build sandbox image
push:              ## Push images to registry
deploy-tier1:      ## Deploy Tier 1 (minimal) profile
deploy-tier2:      ## Deploy Tier 2 (standard) profile
deploy-tier3:      ## Deploy Tier 3 (full) profile
undeploy:          ## Remove all deployed resources
smoke-test:        ## Run smoke tests against deployed system
evaluate:          ## Run evaluation benchmarks
docs-serve:        ## Serve documentation locally
docs-build:        ## Build documentation site
clean:             ## Clean build artifacts
sbom:              ## Generate Software Bill of Materials
```

---

# Part IV — Architecture & Technical Decisions

## 4.1 Architecture Decision Records (ADRs)

All significant technical decisions are documented as ADRs in `docs/adr/`.
Each ADR follows the format:

```markdown
# ADR-NNNN: Title

## Status: [Proposed | Accepted | Deprecated | Superseded]
## Date: YYYY-MM-DD

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing?

## Consequences
What becomes easier or more difficult because of this change?

## Alternatives Considered
What other options were evaluated?
```

### ADR-0001: Use KServe for All Model Serving

**Context**: ToolOrchestra serves 10+ models via raw vLLM processes managed
by SLURM. We need a Kubernetes-native equivalent.

**Decision**: Use KServe `InferenceService` with the RHOAI vLLM
`ServingRuntime` for every model.

**Rationale**:
- KServe is the model serving standard in RHOAI
- Provides autoscaling, canary rollouts, health checks out of the box
- vLLM ServingRuntime is pre-packaged in RHOAI 3.2
- OpenAI-compatible API endpoints match the existing `LLM_CALL.py` interface
- Demonstrates RHOAI's core model serving capability

**Alternatives Considered**:
- Raw vLLM Deployments: more control, but lose KServe autoscaling/routing
- TGI (Text Generation Inference): not as strong on tool-call support
- Triton: overkill for this use case, more complex

### ADR-0002: Replace GPT-5 with Open Models

**Context**: The paper uses GPT-5 as the premium backend for `reasoner-1`,
`search-1`, and `answer-1`. We cannot depend on proprietary API access for
an open-source project.

**Decision**: Substitute GPT-5 with the strongest available open model
(DeepSeek-R1-Distill-Qwen-32B for reasoning, Llama-3.3-70B for general).

**Rationale**:
- Open-source project must be self-contained
- The orchestrator uses abstract tool names — it doesn't know which model
  is behind `reasoner-1`
- Performance will differ from the paper — document this transparently
- Customers can swap in commercial APIs if they have access

**Consequences**:
- Benchmark scores will be lower than the paper reports
- This is itself a valuable data point (quantifies the open vs. proprietary gap)

### ADR-0003: Three-Tier Deployment Profiles

**Context**: GPU resources vary dramatically between customers. The project
must work at small scale for demos and large scale for production.

**Decision**: Define three Kustomize overlay profiles (Tier 1/2/3) with
increasing GPU and model counts.

**Details**:

| Profile   | GPUs        | Unique Models | Use Case          |
|-----------|-------------|---------------|-------------------|
| Tier 1    | 2x A100-80  | 2 (orch + qwen3-32b) | Demo, POC    |
| Tier 2    | 4x A100-80  | 4 (+ deepseek, coder) | Development  |
| Tier 3    | 8–12x A100  | 7 (all models)        | Production   |

### ADR-0004: Gateway Pattern for Tool Routing

**Context**: The orchestrator model generates tool calls. Something needs to
intercept those calls, route them to the correct specialist model, and feed
results back for multi-turn orchestration.

**Decision**: Build a dedicated "Orchestrator Gateway" FastAPI service.

**Rationale**:
- Clean separation between model serving (KServe) and orchestration logic
- Gateway handles the multi-turn loop, retries, timeouts, cost tracking
- Model mapping is externalized in a ConfigMap — no code changes to swap models
- The gateway is the only component that needs to understand the tool
  protocol; individual models just serve completions

**Alternatives Considered**:
- Embed orchestration in a Jupyter notebook: not production-grade
- Use LangChain/LlamaIndex: adds heavy dependencies, hides the logic
- Put routing logic in a KServe transformer: too tightly coupled to serving

### ADR-0005: Kustomize over Helm

**Context**: We need to package Kubernetes manifests for different
environments.

**Decision**: Use Kustomize with base + overlay profiles.

**Rationale**:
- Kustomize is built into `kubectl` and `oc` — no additional tooling
- Overlays map cleanly to our tier concept
- Manifests remain plain YAML (easier to review, audit, and learn from)
- RHOAI documentation uses Kustomize patterns

**Alternatives Considered**:
- Helm: more powerful templating, but adds complexity and a learning curve
  for a reference project meant to be educational

## 4.2 System Architecture

```
                          ┌──────────────────┐
                          │   User / Client  │
                          └────────┬─────────┘
                                   │ HTTPS
                          ┌────────▼─────────┐
                          │  OpenShift Route  │
                          └────────┬─────────┘
                                   │
                 ┌─────────────────▼─────────────────┐
                 │       Orchestrator Gateway         │
                 │       (FastAPI service)            │
                 │                                    │
                 │  1. Receive user question          │
                 │  2. Send to orchestrator model     │
                 │  3. Parse tool calls               │
                 │  4. Route to specialist model      │
                 │  5. Return tool result              │
                 │  6. Repeat until final answer      │
                 │  7. Track cost & latency           │
                 └───┬──────┬──────┬──────┬──────────┘
                     │      │      │      │
          ┌──────────▼──┐ ┌─▼────┐ │   ┌──▼───────────┐
          │ Orchestrator│ │Search│ │   │ Code Sandbox │
          │   8B Model  │ │  /   │ │   │  (isolated)  │
          │  (KServe)   │ │Retrv.│ │   └──────────────┘
          └─────────────┘ └──────┘ │
                                   │
        ┌──────────────────────────┼─────────────────────┐
        │              Specialist Models (KServe)         │
        │                                                 │
        │  ┌────────────┐ ┌────────────┐ ┌─────────────┐ │
        │  │ Qwen3-32B  │ │ DeepSeek   │ │ Qwen-Coder  │ │
        │  │ (answer,   │ │ R1-32B     │ │ 32B         │ │
        │  │  search)   │ │ (reasoner) │ │ (reasoner)  │ │
        │  └────────────┘ └────────────┘ └─────────────┘ │
        │  ┌────────────┐ ┌────────────┐ ┌─────────────┐ │
        │  │ Llama-70B  │ │ Qwen-Math  │ │ Qwen-Math   │ │
        │  │ (answer)   │ │ 72B        │ │ 7B          │ │
        │  └────────────┘ └────────────┘ └─────────────┘ │
        └─────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────▼─────────────────────┐
        │              Platform Services                  │
        │                                                 │
        │  ┌────────────┐ ┌────────────┐ ┌─────────────┐ │
        │  │ Model      │ │ Prometheus │ │ RHOAI       │ │
        │  │ Registry   │ │ + Grafana  │ │ Pipelines   │ │
        │  └────────────┘ └────────────┘ └─────────────┘ │
        └─────────────────────────────────────────────────┘
```

## 4.3 Model Strategy

### The Original Mapping (from upstream `eval_hle.py`)

| Abstract Tool      | Upstream Model             | Our Substitute                        |
|--------------------|----------------------------|---------------------------------------|
| `reasoner-1`       | GPT-5                      | DeepSeek-R1-Distill-Qwen-32B         |
| `reasoner-2`       | GPT-5 Mini                 | Qwen3-32B                            |
| `reasoner-3`       | Qwen2.5-Coder-32B         | Qwen2.5-Coder-32B (same)             |
| `answer-1`         | GPT-5                      | Llama-3.3-70B-Instruct               |
| `answer-2`         | GPT-5 Mini                 | Qwen3-32B                            |
| `answer-3`         | Llama-3.3-70B              | Llama-3.3-70B (same)                 |
| `answer-4`         | Qwen3-32B                  | Qwen3-32B (same)                     |
| `answer-math-1`    | Qwen2.5-Math-72B           | Qwen2.5-Math-72B (same)              |
| `answer-math-2`    | Qwen2.5-Math-7B            | Qwen2.5-Math-7B (same)               |
| `search-1`         | GPT-5                      | DeepSeek-R1-Distill-Qwen-32B         |
| `search-2`         | GPT-5 Mini                 | Qwen3-32B                            |
| `search-3`         | Qwen3-32B                  | Qwen3-32B (same)                     |

### Unique Models to Serve (deduplicated)

| #  | Model                              | VRAM (fp16) | Tensor Parallel | Tier  |
|----|-------------------------------------|-------------|-----------------|-------|
| 1  | nvidia/Nemotron-Orchestrator-8B    | ~16 GB      | 1               | 1+    |
| 2  | Qwen/Qwen3-32B                    | ~64 GB      | 1               | 1+    |
| 3  | deepseek-ai/DeepSeek-R1-Distill   | ~64 GB      | 1               | 2+    |
| 4  | Qwen/Qwen2.5-Coder-32B-Instruct   | ~64 GB      | 1               | 2+    |
| 5  | meta-llama/Llama-3.3-70B-Instruct | ~140 GB     | 2               | 3     |
| 6  | Qwen/Qwen2.5-Math-72B-Instruct    | ~144 GB     | 2               | 3     |
| 7  | Qwen/Qwen2.5-Math-7B-Instruct     | ~14 GB      | 1               | 3     |

## 4.4 Infrastructure Requirements

### Cluster Sizing by Tier

| Component              | Tier 1            | Tier 2             | Tier 3              |
|------------------------|-------------------|--------------------|----------------------|
| **GPU worker nodes**   | 1x p4de.24xlarge  | 1x p4de.24xlarge   | 2–3x p4de.24xlarge   |
| **GPUs per node**      | 8x A100-80GB      | 8x A100-80GB       | 8x A100-80GB         |
| **Total GPU VRAM**     | 640 GB            | 640 GB             | 1,280–1,920 GB       |
| **CPU workers**        | 1x m5.4xlarge     | 2x m5.4xlarge      | 2x m5.4xlarge        |
| **Storage (models)**   | 200 GB gp3 (RWO)  | 500 GB EFS (RWX)  | 2 TB FSx Lustre      |
| **Storage (training)** | —                 | —                  | 2 TB FSx Lustre      |

### Required RHOAI Operators

| Operator                            | Version     | Phase Required |
|-------------------------------------|-------------|----------------|
| Red Hat OpenShift AI                | ≥ 3.2.0     | 0              |
| NVIDIA GPU Operator                 | ≥ 25.3      | 0              |
| Node Feature Discovery              | ≥ 4.20      | 0              |
| Red Hat OpenShift Serverless         | ≥ 1.37      | 1              |
| Red Hat OpenShift Service Mesh 3     | ≥ 3.1       | 1              |
| Red Hat OpenShift Pipelines          | ≥ 1.21      | 5              |

### External Dependencies

| Service        | Purpose                         | Required? | Self-Hosted Alt         |
|----------------|----------------------------------|-----------|-------------------------|
| HuggingFace    | Model weight downloads           | Phase 1   | Pre-download to S3/PVC  |
| Tavily API     | Web search for `search` tool     | Phase 3   | SearXNG                 |
| W&B / MLflow   | Training metrics                 | Phase 6   | MLflow on RHOAI         |
| Container Reg  | Image storage (ghcr.io/quay.io) | Phase 0   | OpenShift internal reg  |

---

# Part V — Implementation Phases

## Phase 0 — Project Bootstrap (Week 1)

### Objectives
- Initialize the repository with all open-source project files
- Verify RHOAI cluster readiness
- Set up CI/CD pipelines

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Create repository with LICENSE, README, CODE_OF_CONDUCT  | Repo      |
| 02 | Set up `.github/` (issue templates, PR template, labels) | Repo      |
| 03 | Create CONTRIBUTING.md and GOVERNANCE.md                 | Docs      |
| 04 | Write `.pre-commit-config.yaml` and `Makefile`           | Tooling   |
| 05 | Create `pyproject.toml` with ruff, mypy, pytest config   | Tooling   |
| 06 | Set up GitHub Actions: ci.yaml (lint, test, validate)    | CI        |
| 07 | Set up GitHub Actions: container-build.yaml              | CI        |
| 08 | Create base Kustomize manifests (namespace, RBAC, PVC)   | Manifests |
| 09 | Write `scripts/setup-cluster.sh` — verify operators      | Scripts   |
| 10 | Create ADR directory with initial ADRs (0001–0005)       | Docs      |
| 11 | Set up MkDocs with initial documentation structure       | Docs      |
| 12 | Create `model-mapping-configmap.yaml`                    | Manifests |
| 13 | Audit model licenses, document in THIRD_PARTY_LICENSES   | Legal     |

### Exit Criteria
- [ ] `make lint` passes
- [ ] `make test` passes (with placeholder tests)
- [ ] `kustomize build profiles/tier1-minimal` produces valid YAML
- [ ] `scripts/setup-cluster.sh` passes against target cluster
- [ ] CI pipeline runs successfully on PR

---

## Phase 1 — Orchestrator Model Serving (Week 2)

### Objectives
- Deploy Nemotron-Orchestrator-8B on RHOAI via KServe
- Verify the model generates tool calls

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Write `download-models.sh` for Orchestrator-8B           | Scripts   |
| 02 | Create ServingRuntime manifest (vLLM + tool-call parser)  | Manifests |
| 03 | Create InferenceService manifest                         | Manifests |
| 04 | Write smoke test for tool-call generation                | Tests     |
| 05 | Register model in RHOAI Model Registry                   | RHOAI     |
| 06 | Document in `docs/guides/deploy-orchestrator.md`         | Docs      |
| 07 | Write `scripts/smoke-test.sh`                            | Scripts   |

### Key Technical Details

```yaml
# ServingRuntime critical args
args:
  - --dtype=half
  - --max-model-len=24000
  - --gpu-memory-utilization=0.90
  - --enable-auto-tool-choice
  - --tool-call-parser=hermes    # Verify with actual checkpoint
  - --port=8080
```

### Exit Criteria
- [ ] InferenceService shows `READY=True`
- [ ] `curl /v1/chat/completions` with tools returns a `tool_calls` response
- [ ] Latency to first token < 5s on A100
- [ ] `make smoke-test` passes
- [ ] Guide documentation is complete and reviewable

### RHOAI Features Demonstrated
- KServe InferenceService
- vLLM ServingRuntime
- Model Registry

---

## Phase 2 — Specialist Model Ecosystem (Weeks 3–4)

### Objectives
- Deploy specialist models in priority order
- Validate each model serves correct responses
- Set up per-tier Kustomize overlays

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Create ServingRuntime + InferenceService for Qwen3-32B    | Manifests |
| 02 | Create ServingRuntime + InferenceService for DeepSeek-R1  | Manifests |
| 03 | Create ServingRuntime + InferenceService for Qwen-Coder   | Manifests |
| 04 | Create ServingRuntime + InferenceService for Llama-70B    | Manifests |
| 05 | Create ServingRuntime + InferenceService for Qwen-Math-72B| Manifests |
| 06 | Create ServingRuntime + InferenceService for Qwen-Math-7B | Manifests |
| 07 | Configure tensor parallelism for 70B+ models             | Manifests |
| 08 | Create Kustomize overlays for Tier 1, 2, 3               | Manifests |
| 09 | Write integration tests for each model                   | Tests     |
| 10 | Set up Grafana dashboard for GPU utilization              | Observ.   |
| 11 | Document in `docs/guides/deploy-specialists.md`          | Docs      |

### Deployment Priority Order

| Order | Model          | Reason                                          |
|-------|----------------|-------------------------------------------------|
| 1     | Qwen3-32B      | Covers 4 abstract tools; enables Tier 1         |
| 2     | DeepSeek-R1    | Premium reasoning substitute; enables Tier 2    |
| 3     | Qwen-Coder     | Code generation for enhance_reasoning tool      |
| 4     | Llama-70B      | Strongest general answering; enables Tier 3     |
| 5     | Qwen-Math-72B  | Math specialist                                 |
| 6     | Qwen-Math-7B   | Budget math fallback                            |

### Exit Criteria
- [ ] All models for target tier show `READY=True`
- [ ] Each model produces correct output for representative prompts
- [ ] `kustomize build profiles/tier{1,2,3}` produces valid manifests
- [ ] GPU utilization dashboard shows all models loaded
- [ ] Integration tests pass

### RHOAI Features Demonstrated
- Multi-model KServe serving
- Tensor-parallel vLLM serving
- GPU Operator scheduling
- OpenShift monitoring (DCGM Exporter)

---

## Phase 3 — Tool Integration Layer (Weeks 5–6)

### Objectives
- Build the Orchestrator Gateway service
- Deploy the retrieval service
- Deploy the code execution sandbox
- Wire everything together via the model-mapping ConfigMap

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Implement Gateway FastAPI app with config loading        | Code      |
| 02 | Implement multi-turn orchestration loop                  | Code      |
| 03 | Implement tool router (reads ConfigMap, routes calls)    | Code      |
| 04 | Implement OpenAI-compatible model client                 | Code      |
| 05 | Add Prometheus metrics (latency, cost, tool usage)       | Code      |
| 06 | Add structured logging with request tracing              | Code      |
| 07 | Write unit tests (≥80% coverage)                        | Tests     |
| 08 | Build and publish gateway container image                | CI        |
| 09 | Create gateway Kubernetes manifests                      | Manifests |
| 10 | Implement retrieval service (FAISS + Tavily)             | Code      |
| 11 | Build and publish retrieval container image              | CI        |
| 12 | Create retrieval Kubernetes manifests                    | Manifests |
| 13 | Implement code sandbox service                           | Code      |
| 14 | Create sandbox manifests with security constraints       | Manifests |
| 15 | Write integration tests for full routing flow            | Tests     |
| 16 | Document in `docs/guides/configure-tools.md`             | Docs      |
| 17 | Add API reference to `docs/reference/api.md`             | Docs      |

### Gateway Design

```python
# Simplified orchestration loop
async def orchestrate(question: str, max_turns: int = 30):
    messages = [system_prompt, {"role": "user", "content": question}]
    
    for turn in range(max_turns):
        response = await call_orchestrator(messages)
        
        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                result = await route_tool_call(tool_call)
                messages.append(tool_call_message)
                messages.append(tool_result_message)
                metrics.track(tool_call, result)
        else:
            return response.content  # Final answer
    
    return timeout_response
```

### Exit Criteria
- [ ] Gateway serves on `/v1/orchestrate` endpoint
- [ ] A question triggers multi-turn orchestration across multiple models
- [ ] Prometheus metrics are exposed at `/metrics`
- [ ] Cost tracking correctly sums tool-call pricing
- [ ] Unit test coverage ≥ 80%
- [ ] Integration test passes end-to-end

### RHOAI Features Demonstrated
- Workbenches for development
- Data connections (S3/PVC for FAISS indexes)
- Kubernetes-native service networking

---

## Phase 4 — End-to-End Demonstration (Week 7)

### Objectives
- Run full orchestration loops against real questions
- Build a demo UI
- Validate cost-aware routing behavior

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Create demo question set (10 diverse questions)          | Data      |
|    | See [Demo Question Set](#demo-question-set) below        |           |
| 02 | Run full orchestration, collect traces                   | Testing   |
| 03 | Build Gradio demo UI                                     | Code      |
| 04 | Deploy demo UI as RHOAI Workbench or standalone          | Deploy    |
| 05 | Create Grafana dashboard for orchestration metrics       | Observ.   |
| 06 | Record demo walkthrough (terminal or video)              | Docs      |
| 07 | Write `docs/architecture/data-flow.md` with real traces  | Docs      |

### Demo Question Set

Ten prompts designed to exercise every aspect of the paper's thesis:
cost-aware routing, multi-turn orchestration, search, code execution, and
specialist model selection.

| #  | Category              | Prompt                                                                                                                                                   | Tests                                      |
|----|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| 01 | Factual comparison    | What is the population of Tokyo compared to Paris?                                                                                                       | search → answer routing; cheap search model |
| 02 | Multi-hop factual     | What is the approximate ratio of the population of Tokyo to the population of Paris, and which city has higher population density?                        | search + computation + answer               |
| 03 | Multi-hop biography   | What is the birth year of the chemist who first synthesized captopril, and what university did they attend?                                               | multi-search → answer                       |
| 04 | Computation (primes)  | What is the sum of all prime numbers between 10,000 and 10,100?                                                                                          | enhance_reasoning (code execution)          |
| 05 | Computation (finance) | If you invest $5,000 at 7.3% annual interest compounded monthly, how much do you have after 17 years and 4 months? Round to the nearest cent.            | enhance_reasoning (code execution)          |
| 06 | Expert STEM           | In the Diels-Alder reaction between cyclopentadiene and maleic anhydride, what is the endo:exo product ratio at room temperature?                        | search → enhance_reasoning → answer         |
| 07 | Mixed easy + hard     | What is the capital of France, and what is the closed-form solution to the integral of e^(-x²) from negative infinity to positive infinity?               | cost routing: cheap model for easy, expensive for hard |
| 08 | Combinatorics         | How many distinct ways can you tile a 4×10 grid using 1×2 dominoes?                                                                                      | enhance_reasoning (code execution)          |
| 09 | Current events        | Who won the Nobel Prize in Physics in 2024 and for what contribution?                                                                                    | search (requires web retrieval)             |
| 10 | Multi-step reasoning  | The architect who designed the Sydney Opera House also designed a church in what city, and in what year was that church completed?                         | multi-search → answer                       |

**What to verify for each prompt:**

- Tool trace shows the orchestrator chose the right tool type (search vs
  reasoning vs answer) for each turn.
- Cost-aware routing: cheaper specialist models (e.g., `search-3`,
  `answer-4`) are used for straightforward sub-tasks; expensive models
  (e.g., `answer-1`, `reasoner-1`) are reserved for hard steps.
- Multi-turn behavior: the orchestrator does not try to answer immediately
  when it lacks information — it searches or reasons first.
- The orchestrated answer is more accurate than a direct single-model answer
  (validated via the comparison pipeline).

**Validated example (Population Compare — prompt 01):**

```
Prompt: "What is the population of Tokyo compared to Paris?"
Trace:
  Turn 0: search via Llama-3.2-3B → 5 results (0.5s, 98+12 tok)
  Turn 1: search via Llama-3.2-3B → 5 results (0.5s, 410+12 tok)
  Turn 2: answer via Gemini 2.5 Pro → final answer (17.4s, 621+769 tok)
Result: 3 turns, 6,507 tokens, $0.0086
Key observation: Orchestrator used cheap search model, escalated to
expensive answer model only for the final synthesis.
```

### Exit Criteria
- [ ] 5+ diverse questions answered correctly through full orchestration
- [ ] Demo UI shows live orchestration steps, tool choices, and costs
- [ ] Grafana dashboard visualizes tool-call distribution and latency
- [ ] Orchestrator demonstrably picks cheaper models for easier sub-tasks

### RHOAI Features Demonstrated
- RHOAI Workbenches
- OpenShift Routes for external access
- Full KServe ecosystem

---

## Phase 5 — Evaluation & Benchmarking (Weeks 8–9)

### Objectives
- Run official benchmarks adapted for RHOAI
- Document results transparently (including gap vs. paper)
- Automate evaluation via RHOAI Pipelines

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Adapt `eval_hle.py` to use KServe endpoints              | Code      |
| 02 | Adapt `eval_frames.py` to use KServe endpoints           | Code      |
| 03 | Adapt `tau2-bench/run.py` for RHOAI                      | Code      |
| 04 | Run HLE evaluation, collect results                      | Eval      |
| 05 | Run FRAMES evaluation, collect results                   | Eval      |
| 06 | Run τ²-Bench evaluation, collect results                 | Eval      |
| 07 | Create evaluation results comparison table               | Docs      |
| 08 | Build Kubeflow Pipeline for automated evaluation         | Pipeline  |
| 09 | Write `docs/benchmarks.md` with full results             | Docs      |
| 10 | Tag evaluated model versions in Model Registry           | RHOAI     |

### Expected Results Table

| Benchmark | Paper (GPT-5 tools) | Ours (Open Models) | Delta | Notes          |
|-----------|---------------------|--------------------|-------|----------------|
| HLE       | 37.1%               | TBD                | TBD   | Expected lower |
| FRAMES    | TBD                 | TBD                | TBD   |                |
| τ²-Bench  | TBD                 | TBD                | TBD   |                |

### Exit Criteria
- [ ] All three benchmarks produce numeric results
- [ ] Results are documented with methodology and caveats
- [ ] Evaluation pipeline runs end-to-end on RHOAI Pipelines
- [ ] Model versions are tagged in Model Registry with scores

### RHOAI Features Demonstrated
- RHOAI Pipelines (Kubeflow)
- TrustyAI evaluation
- Model Registry (version tagging with metadata)

---

## Phase 6 — Training Pipeline (Weeks 10–13)

### Objectives
- Reproduce RL training on RHOAI using KubeRay
- Replace SLURM orchestration with Kubeflow Pipelines
- Produce a trained checkpoint and compare to pre-trained

### Prerequisites
- Tier 3 cluster (16+ A100-80GB GPUs)
- RWX shared storage (FSx for Lustre)
- Training data downloaded (`nvidia/ToolScale`)

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Build training container image (PyTorch, vLLM, verl, Ray)| Container |
| 02 | Create RayCluster manifest (2 nodes × 8 GPUs)           | Manifests |
| 03 | Create RayJob manifest for GRPO training                 | Manifests |
| 04 | Adapt `main_grpo_quick3.py` for KServe model discovery   | Code      |
| 05 | Build Kubeflow Pipeline: deploy → train → evaluate       | Pipeline  |
| 06 | Run training for 1 epoch, verify loss decreases          | Training  |
| 07 | Run training for 10 epochs, produce checkpoint           | Training  |
| 08 | Deploy trained checkpoint, run evaluation                | Eval      |
| 09 | Register trained model in Model Registry                 | RHOAI     |
| 10 | Document full training process                           | Docs      |

### Exit Criteria
- [ ] RayJob launches Ray cluster on RHOAI via KubeRay
- [ ] Training runs for multiple epochs without error
- [ ] Loss curve shows convergence
- [ ] Trained checkpoint serves correctly on KServe
- [ ] Benchmark scores are measurable on trained checkpoint

### RHOAI Features Demonstrated
- KubeRay (RayCluster + RayJob)
- Kubeflow Training Operator (alternative path)
- RHOAI Pipelines for training orchestration
- Model Registry for checkpoint management

---

## Phase 7 — Production Readiness (Weeks 14–15)

### Objectives
- Harden the deployment for production use
- Add autoscaling, monitoring, security, CI/CD

### Deliverables

| #  | Task                                                     | Type      |
|----|----------------------------------------------------------|-----------|
| 01 | Configure KServe autoscaling (scale-to-zero for rarely   | Manifests |
|    | used specialists, min=1 for orchestrator)                |           |
| 02 | Create ServiceMonitor for gateway metrics                | Observ.   |
| 03 | Create PrometheusRule for alerts (latency, errors, OOM)  | Observ.   |
| 04 | Create comprehensive Grafana dashboard                   | Observ.   |
| 05 | Add NetworkPolicies (restrict cross-namespace traffic)   | Security  |
| 06 | Configure Service Mesh mTLS for model traffic            | Security  |
| 07 | Add PodDisruptionBudgets for model serving pods          | Manifests |
| 08 | Create Tekton Pipeline for CI/CD deployment              | CI/CD     |
| 09 | Add container image signing with cosign                  | Security  |
| 10 | Generate SBOM for all container images                   | Security  |
| 11 | Write production runbook                                 | Docs      |
| 12 | Write troubleshooting guide                              | Docs      |
| 13 | Create examples/ directory with client code              | Docs      |
| 14 | Final documentation review and polish                    | Docs      |

### Exit Criteria
- [ ] Autoscaling works: specialist models scale down after idle period
- [ ] Alerts fire correctly for simulated failure scenarios
- [ ] Grafana dashboard provides full operational visibility
- [ ] CI/CD pipeline deploys a change from PR to production
- [ ] All container images are signed and have SBOMs
- [ ] Documentation is complete and reviewed

### RHOAI Features Demonstrated
- KServe autoscaling (scale-to-zero)
- OpenShift Monitoring (Prometheus, Grafana)
- OpenShift Service Mesh (mTLS)
- OpenShift Pipelines (Tekton CI/CD)
- TrustyAI for ongoing model quality

---

# Part VI — Testing Strategy

## 6.1 Testing Pyramid

```
         ╱╲
        ╱  ╲           E2E Tests
       ╱ E2E╲          (5-10 scenarios, run on cluster)
      ╱──────╲
     ╱        ╲        Integration Tests
    ╱ Integr.  ╲       (model routing, multi-turn loops)
   ╱────────────╲
  ╱              ╲     Unit Tests
 ╱   Unit Tests   ╲    (business logic, config parsing,
╱──────────────────╲    schema validation)
```

## 6.2 Unit Tests

- **Framework**: pytest
- **Coverage target**: ≥ 80% for all `src/` code
- **What to test**:
  - Config loading and validation
  - Tool call parsing
  - Model mapping resolution
  - Cost calculation
  - Request/response schema validation
  - Error handling paths
- **Mocking**: All external calls (KServe, Tavily) are mocked
- **Run with**: `make test-unit`

## 6.3 Integration Tests

- **Framework**: pytest with test fixtures
- **What to test**:
  - Gateway → KServe endpoint connectivity
  - Multi-turn orchestration with mock models
  - Retrieval service → FAISS index querying
  - Code sandbox execution and timeout handling
  - Model mapping ConfigMap hot-reload
- **Environment**: Requires either mock model endpoints (`hack/mock-models.sh`)
  or a running cluster
- **Run with**: `make test-integration`

## 6.4 End-to-End Tests

- **Framework**: Shell scripts + pytest
- **What to test**:
  - Deploy Tier 1 profile → smoke test → undeploy
  - Submit a question → receive correct multi-turn orchestration → final answer
  - Model autoscaling (scale to zero, scale up on request)
  - Failure scenarios (model endpoint down, timeout)
- **Environment**: Requires RHOAI cluster with GPUs
- **Run with**: `make test-e2e`

## 6.5 Manifest Validation

- **kubeconform**: Validate all manifests against OpenShift API schemas
- **kube-linter**: Check for security and best-practice issues
- **kustomize build**: Verify all overlays produce valid output
- **Run with**: `make lint-manifests`

---

# Part VII — Release & Versioning Strategy

## 7.1 Versioning

**Semantic Versioning 2.0.0**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes to the API, manifest structure, or
  configuration format
- **MINOR**: New features, new model support, new deployment tier
- **PATCH**: Bug fixes, documentation updates, dependency bumps

Pre-release versions: `0.x.y` (no stability guarantees until 1.0.0)

## 7.2 Release Cadence

| Milestone        | Version | Target         | Scope                          |
|------------------|---------|----------------|--------------------------------|
| Project bootstrap| v0.1.0  | End of Phase 0 | Repo structure, CI, manifests  |
| First model      | v0.2.0  | End of Phase 1 | Orchestrator serving on KServe |
| Tool ecosystem   | v0.3.0  | End of Phase 3 | Gateway + specialists + tools  |
| Evaluation       | v0.4.0  | End of Phase 5 | Benchmark results              |
| Training         | v0.5.0  | End of Phase 6 | Training pipeline on KubeRay   |
| Production       | v1.0.0  | End of Phase 7 | Production-ready release       |

## 7.3 Release Process

1. Cut release branch from `main`: `release/v0.2.0`
2. Update `CHANGELOG.md` with changes since last release
3. Update version references in documentation
4. Create annotated Git tag: `git tag -a v0.2.0`
5. GitHub Actions builds and pushes tagged container images
6. GitHub Release created with:
   - Changelog excerpt
   - Container image references (with digest)
   - SBOM attachments
   - Upgrade instructions from previous version

## 7.4 Changelog

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

## [0.2.0] - 2026-XX-XX

### Added
- KServe InferenceService for Nemotron-Orchestrator-8B
- vLLM ServingRuntime with tool-call support
- Smoke test script for model validation

### Changed
- Updated base Kustomize manifests with GPU resource requests

### Fixed
- N/A
```

---

# Part VIII — Documentation Strategy

## 8.1 Documentation Framework

**MkDocs Material** — clean, searchable, versioned documentation site.

- Hosted via GitHub Pages (auto-deployed from `main`)
- Versioned docs per release (via mike)
- Search enabled
- Dark mode supported

## 8.2 Documentation Types

| Type              | Location         | Purpose                              |
|-------------------|------------------|--------------------------------------|
| **README**        | `/README.md`     | First impression, quick start        |
| **Getting Started**| `/docs/getting-started/` | Step-by-step setup         |
| **Guides**        | `/docs/guides/`  | Phase-by-phase deployment guides     |
| **Architecture**  | `/docs/architecture/` | System design, decisions        |
| **Reference**     | `/docs/reference/` | API docs, config reference         |
| **ADRs**          | `/docs/adr/`     | Architecture Decision Records        |
| **Troubleshooting**| `/docs/reference/troubleshooting.md` | Common issues   |
| **Blog**          | `/docs/blog/`    | Technical deep-dives, announcements  |
| **API Docs**      | Auto-generated   | OpenAPI spec from FastAPI            |

## 8.3 README Structure

```markdown
# orchestrator-rhoai

One-line description.

Badges: CI status, latest release, license, docs link

## What Is This?
2-paragraph explanation with architecture diagram

## Quick Start
5-step guide to get running (prerequisites → deploy → test)

## Documentation
Link to full docs site

## Deployment Tiers
Brief table of Tier 1/2/3

## RHOAI Features
List of demonstrated capabilities

## Contributing
Link to CONTRIBUTING.md

## License
Apache 2.0

## Acknowledgments
Credit to NVIDIA ToolOrchestra paper and team
```

## 8.4 Documentation Requirements for PRs

Every PR must include:
- **Code changes**: Update relevant guide or reference docs
- **New features**: Add to CHANGELOG.md under `[Unreleased]`
- **API changes**: Update API reference
- **Config changes**: Update configuration reference
- **Architecture changes**: Create or update ADR

---

# Part IX — Community & Adoption

## 9.1 Community Building

| Activity                     | When          | Purpose                           |
|------------------------------|---------------|-----------------------------------|
| GitHub Discussions enabled   | Phase 0       | Q&A, ideas, show & tell           |
| `good-first-issue` labels    | Phase 1+      | Onboard new contributors          |
| Blog post: "Why we built it" | Phase 4       | Announce project, explain value   |
| Conference talk / demo       | Post Phase 5  | Showcase at Red Hat Summit, etc.  |
| Monthly community call       | Post v1.0     | Ongoing engagement                |

## 9.2 Adoption Path for Customers

```
Step 1: Read docs, understand architecture          (30 min)
Step 2: Run setup-cluster.sh on their RHOAI cluster (15 min)
Step 3: Deploy Tier 1 with make deploy-tier1         (30 min)
Step 4: Run smoke-test.sh                            (5 min)
Step 5: Try the demo UI with their own questions     (open-ended)
Step 6: Scale to Tier 2/3 based on needs             (1-2 hours)
Step 7: Customize model mapping for their domain     (30 min)
```

**Target**: A customer goes from zero to a working demo in under 2 hours.

---

# Part X — Risk Management

## 10.1 Risk Register

| ID   | Risk                                      | Likelihood | Impact | Mitigation                                         |
|------|-------------------------------------------|-----------|--------|------------------------------------------------------|
| R-01 | Benchmark scores significantly lower      | High      | Medium | Document transparently; this quantifies the open vs. proprietary gap and is itself a contribution |
| R-02 | 70B/72B models OOM on available GPUs      | Medium    | High   | Support AWQ/GPTQ quantized variants as fallback; reduce `max_model_len` |
| R-03 | Tool-call parser mismatch with model      | Medium    | High   | Test all parsers (hermes, llama3_json, qwen) against actual checkpoint in Phase 1 |
| R-04 | Multi-turn loop hangs or loops infinitely | Low       | High   | Hard cap at 30 turns; per-turn timeout; circuit breaker in gateway |
| R-05 | KubeRay training perf worse than SLURM    | Medium    | Medium | Benchmark and document; optimize NCCL; use GDRCopy for GPU Direct RDMA |
| R-06 | Shared storage bandwidth bottleneck       | Medium    | High   | FSx Lustre for training; gp3 is fine for inference-only |
| R-07 | Upstream ToolOrchestra code changes       | Low       | Low    | Pin to specific upstream commit; periodically sync |
| R-08 | Model license prevents commercial use     | Low       | High   | Complete license audit in Phase 0; have alternatives ready |
| R-09 | RHOAI version incompatibility             | Medium    | Medium | Test against RHOAI 3.2+; document minimum version |
| R-10 | Customer lacks sufficient GPU quota       | High      | High   | Tier 1 works with 2 GPUs; provide cloud GPU guidance |

## 10.2 Open Questions

| #  | Question                                                  | Owner | Target Phase |
|----|-----------------------------------------------------------|-------|--------------|
| Q1 | Which vLLM tool-call parser works with Orchestrator-8B?   | Eng   | Phase 1      |
| Q2 | Can AWQ-quantized 70B models maintain quality?            | Eng   | Phase 2      |
| Q3 | Is ToolScale dataset licensed for commercial training?     | Legal | Phase 0      |
| Q4 | Does customer need air-gapped deployment (no Tavily)?     | PM    | Phase 3      |
| Q5 | What is the customer's specific domain / use case?        | PM    | Phase 3      |
| Q6 | Will RHOAI 3.3 change any KServe or KubeRay APIs?        | Eng   | Phase 7      |
| Q7 | Should we support non-AWS clouds (GCP, Azure)?            | PM    | Post v1.0    |

---

# RHOAI Feature Coverage Summary

This project demonstrates **15 RHOAI platform features** across 7 phases:

| #  | RHOAI Feature                | Phase | How It's Used                                    |
|----|------------------------------|-------|--------------------------------------------------|
| 1  | DataScienceCluster           | 0     | Verify all components are managed                |
| 2  | Data Science Projects        | 0     | Namespace organization                           |
| 3  | KServe InferenceService      | 1–4   | All models served via KServe                     |
| 4  | vLLM ServingRuntime          | 1–4   | GPU-accelerated inference with tool-call support |
| 5  | Multi-model Serving          | 2     | 7 models served simultaneously                   |
| 6  | Model Registry               | 1–7   | Version tracking, metadata, evaluation scores    |
| 7  | Workbenches                  | 3–5   | Development, demo UI, analysis                   |
| 8  | Data Connections             | 3     | S3/PVC for FAISS indexes and model weights       |
| 9  | KubeRay (RayCluster/RayJob) | 6     | Distributed RL training                          |
| 10 | Training Operator            | 6     | PyTorchJob as alternative                        |
| 11 | RHOAI Pipelines              | 5–7   | Evaluation, training, deployment automation      |
| 12 | TrustyAI                     | 5, 7  | Model evaluation and quality monitoring          |
| 13 | KServe Autoscaling           | 7     | Scale-to-zero for specialist models              |
| 14 | OpenShift Monitoring         | 2–7   | GPU metrics, inference metrics, alerting         |
| 15 | OpenShift Service Mesh       | 7     | mTLS for model-to-model traffic                  |

---

# Timeline Summary

| Phase | Scope                         | Duration  | Cumulative | Release   |
|-------|-------------------------------|-----------|------------|-----------|
| 0     | Project bootstrap             | 1 week    | Week 1     | v0.1.0    |
| 1     | Orchestrator serving          | 1 week    | Week 2     | v0.2.0    |
| 2     | Specialist models             | 2 weeks   | Week 4     |           |
| 3     | Gateway + tools               | 2 weeks   | Week 6     | v0.3.0    |
| 4     | End-to-end demo               | 1 week    | Week 7     |           |
| 5     | Evaluation & benchmarks       | 2 weeks   | Week 9     | v0.4.0    |
| 6     | Training pipeline             | 4 weeks   | Week 13    | v0.5.0    |
| 7     | Production hardening          | 2 weeks   | Week 15    | v1.0.0    |

**Minimum viable demo**: Weeks 1–7 (Phase 0–4)
**Full production release**: Week 15 (v1.0.0)
