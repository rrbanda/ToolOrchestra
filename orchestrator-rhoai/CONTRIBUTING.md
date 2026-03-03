# Contributing to orchestrator-rhoai

Thank you for your interest in contributing. This guide will help you get
started.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Process](#issue-process)
- [Developer Certificate of Origin](#developer-certificate-of-origin)

---

## Prerequisites

- **Python** ≥ 3.12
- **oc** CLI (OpenShift client)
- **kustomize** ≥ 5.0
- **pre-commit** (`pip install pre-commit`)
- **podman** or **docker** (for building container images)

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/orchestrator-rhoai.git
cd orchestrator-rhoai

# 2. Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install development dependencies
pip install -e "src/gateway[dev]"
pip install -e "src/retrieval[dev]"
pip install -e "src/code_sandbox[dev]"

# 4. Install pre-commit hooks
pre-commit install --install-hooks

# 5. Verify your setup
make lint
make test-unit
```

## Branching Strategy

We use **trunk-based development** with short-lived feature branches.

- `main` is the only long-lived branch and is always deployable
- Create feature branches from `main`: `feat/phase-1-orchestrator-serving`
- Keep branches short-lived (days, not weeks)
- All changes go through pull requests

```bash
git checkout main
git pull origin main
git checkout -b feat/my-feature
# ... make changes ...
git push -u origin feat/my-feature
# Open a PR on GitHub
```

## Commit Messages

We follow [Conventional Commits v1.0.0](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
Signed-off-by: Your Name <your.email@example.com>
```

### Types

| Type       | When to use                              |
|------------|------------------------------------------|
| `feat`     | A new feature                            |
| `fix`      | A bug fix                                |
| `docs`     | Documentation only                       |
| `style`    | Formatting, no code change               |
| `refactor` | Code change that neither fixes nor adds  |
| `test`     | Adding or updating tests                 |
| `build`    | Build system or dependencies             |
| `ci`       | CI/CD configuration                      |
| `chore`    | Maintenance tasks                        |
| `perf`     | Performance improvements                 |

### Scopes

`gateway`, `retrieval`, `sandbox`, `manifests`, `evaluation`, `training`,
`docs`, `ci`

### Examples

```
feat(gateway): add multi-turn orchestration loop
fix(manifests): correct GPU resource limits for 70B models
docs(architecture): add tool routing sequence diagram
ci: add container image vulnerability scanning
```

## Pull Request Process

1. **Before opening a PR**:
   - Ensure `make lint` passes
   - Ensure `make test-unit` passes
   - Update relevant documentation

2. **PR template**: Fill out the template completely, including:
   - Description of changes
   - Related issue(s)
   - Testing performed
   - Documentation updates
   - Checklist items

3. **Review requirements**:
   - At least 1 maintainer approval
   - All CI checks passing
   - No unresolved review comments

4. **Merge strategy**: Squash merge to `main`

## Code Style

### Python

| Tool     | Purpose          | Command          |
|----------|------------------|------------------|
| `ruff`   | Lint + format    | `make lint`      |
| `mypy`   | Type checking    | `make typecheck` |

- Line length: 99 characters
- Quote style: double quotes
- Import sorting: isort via ruff

### Kubernetes Manifests

- Use `kustomize` for all manifest management
- Validate with `kubeconform` and `kube-linter`
- Use `yamllint` for formatting

### Container Images

- Use `Containerfile` (not `Dockerfile`)
- Validate with `hadolint`
- Base images from trusted registries only

## Testing

### Requirements for PRs

- **New code**: Must include unit tests
- **New endpoints**: Must include integration tests
- **Bug fixes**: Must include a regression test
- **Coverage**: Must not decrease below 80%

### Running Tests

```bash
make test-unit          # Unit tests only (fast, no external deps)
make test-integration   # Integration tests (may need mock services)
make test-e2e           # End-to-end (requires RHOAI cluster)
make test               # All tests with coverage
```

### Test Markers

```python
import pytest

@pytest.mark.unit
def test_config_loading():
    ...

@pytest.mark.integration
def test_gateway_routes_to_model():
    ...

@pytest.mark.e2e
def test_full_orchestration():
    ...
```

## Documentation

### Requirements for PRs

| Change Type         | Documentation Update Required          |
|---------------------|----------------------------------------|
| New feature         | Guide or reference doc + CHANGELOG     |
| API change          | API reference + CHANGELOG              |
| Config change       | Configuration reference + CHANGELOG    |
| Architecture change | ADR + architecture docs                |
| Bug fix             | CHANGELOG                              |

### Building Docs Locally

```bash
make docs-serve    # Serve at http://localhost:8000
make docs-build    # Build static site
```

## Issue Process

1. **Search first**: Check existing issues before creating a new one
2. **Use templates**: Choose the appropriate template (bug, feature, question)
3. **Be specific**: Include steps to reproduce, expected vs. actual behavior
4. **Label appropriately**: Use phase, tier, and type labels

## Developer Certificate of Origin

All contributions require a DCO sign-off. This certifies that you have the
right to submit the contribution under the project's license.

Add a `Signed-off-by` line to your commit messages:

```
Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically with `git commit -s`.

By signing off, you agree to the
[Developer Certificate of Origin](https://developercertificate.org/).
