.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT_DIR := $(shell pwd)
SRC_DIR := $(PROJECT_DIR)/src
MANIFESTS_DIR := $(PROJECT_DIR)/manifests
PROFILES_DIR := $(PROJECT_DIR)/profiles

REGISTRY ?= ghcr.io/orchestrator-rhoai
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# ─── Help ────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo "orchestrator-rhoai — Deploy ToolOrchestra on Red Hat OpenShift AI"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Development ─────────────────────────────────────────────────────────────

.PHONY: setup
setup: ## Set up development environment
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install pre-commit ruff mypy pytest pytest-cov pytest-asyncio httpx
	@if [ -f src/gateway/pyproject.toml ]; then .venv/bin/pip install -e "src/gateway[dev]"; fi
	@if [ -f src/retrieval/pyproject.toml ]; then .venv/bin/pip install -e "src/retrieval[dev]"; fi
	@if [ -f src/code_sandbox/pyproject.toml ]; then .venv/bin/pip install -e "src/code_sandbox[dev]"; fi
	@if [ -f src/evaluation/pyproject.toml ]; then .venv/bin/pip install -e "src/evaluation[dev]"; fi
	.venv/bin/pre-commit install --install-hooks
	@echo ""
	@echo "Setup complete. Activate with: source .venv/bin/activate"

# ─── Linting & Formatting ───────────────────────────────────────────────────

.PHONY: lint
lint: lint-python lint-manifests ## Run all linters

.PHONY: lint-python
lint-python: ## Lint Python code with ruff
	$(RUFF) check src/ --config pyproject.toml
	$(RUFF) format --check src/ --config pyproject.toml

.PHONY: format
format: ## Auto-format Python code
	$(RUFF) check --fix src/ --config pyproject.toml
	$(RUFF) format src/ --config pyproject.toml

.PHONY: typecheck
typecheck: ## Run mypy type checking
	$(MYPY) src/ --config-file pyproject.toml

.PHONY: lint-manifests
lint-manifests: ## Validate Kubernetes manifests
	@echo "Validating Kustomize overlays..."
	@for profile in tier1-minimal tier2-standard tier3-full; do \
		echo "  Checking profiles/$$profile..."; \
		kustomize build profiles/$$profile > /dev/null 2>&1 && \
			echo "    ✓ Valid" || \
			echo "    ✗ Failed (profile may not exist yet)"; \
	done

# ─── Testing ─────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run all tests with coverage
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=html -m "not e2e"

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTEST) -m unit

.PHONY: test-integration
test-integration: ## Run integration tests (may require mock services)
	$(PYTEST) -m integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests (requires RHOAI cluster)
	$(PYTEST) -m e2e

# ─── Container Images ───────────────────────────────────────────────────────

.PHONY: build
build: build-gateway build-retrieval build-sandbox ## Build all container images

.PHONY: build-gateway
build-gateway: ## Build gateway container image
	podman build -t $(REGISTRY)/gateway:$(TAG) -f src/gateway/Containerfile src/gateway/

.PHONY: build-retrieval
build-retrieval: ## Build retrieval container image
	podman build -t $(REGISTRY)/retrieval:$(TAG) -f src/retrieval/Containerfile src/retrieval/

.PHONY: build-sandbox
build-sandbox: ## Build code sandbox container image
	podman build -t $(REGISTRY)/code-sandbox:$(TAG) -f src/code_sandbox/Containerfile src/code_sandbox/

.PHONY: push
push: ## Push all container images to registry
	podman push $(REGISTRY)/gateway:$(TAG)
	podman push $(REGISTRY)/retrieval:$(TAG)
	podman push $(REGISTRY)/code-sandbox:$(TAG)

# ─── Deployment ──────────────────────────────────────────────────────────────

.PHONY: deploy-tier1
deploy-tier1: ## Deploy Tier 1 (minimal) profile to cluster
	kustomize build profiles/tier1-minimal | oc apply -f -

.PHONY: deploy-tier2
deploy-tier2: ## Deploy Tier 2 (standard) profile to cluster
	kustomize build profiles/tier2-standard | oc apply -f -

.PHONY: deploy-tier3
deploy-tier3: ## Deploy Tier 3 (full) profile to cluster
	kustomize build profiles/tier3-full | oc apply -f -

.PHONY: undeploy
undeploy: ## Remove all deployed resources
	@echo "Removing orchestrator-rhoai resources..."
	oc delete project orchestrator-rhoai --ignore-not-found

# ─── Cluster Verification ───────────────────────────────────────────────────

.PHONY: check-cluster
check-cluster: ## Verify RHOAI cluster prerequisites
	@bash scripts/setup-cluster.sh

.PHONY: smoke-test
smoke-test: ## Run smoke tests against deployed system
	@bash scripts/smoke-test.sh

# ─── Documentation ──────────────────────────────────────────────────────────

.PHONY: docs-serve
docs-serve: ## Serve documentation locally at http://localhost:8000
	mkdocs serve -f docs/mkdocs.yml

.PHONY: docs-build
docs-build: ## Build documentation site
	mkdocs build -f docs/mkdocs.yml

# ─── Security & Compliance ──────────────────────────────────────────────────

.PHONY: sbom
sbom: ## Generate Software Bill of Materials for all images
	@bash scripts/generate-sbom.sh

.PHONY: scan
scan: ## Scan container images for vulnerabilities
	@for img in gateway retrieval code-sandbox; do \
		echo "Scanning $(REGISTRY)/$$img:$(TAG)..."; \
		trivy image $(REGISTRY)/$$img:$(TAG) --severity HIGH,CRITICAL; \
	done

# ─── Cleanup ─────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf .venv/ dist/ build/ site/ htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
