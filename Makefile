# Copyright The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: help uv install-dev verify format test-python test-scripts test test-cov benchmark clean inspector release changelog

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

# Setting SED for compatibility with macos
ifeq ($(shell command -v gsed 2>/dev/null),)
    SED ?= $(shell command -v sed)
else
    SED ?= $(shell command -v gsed)
endif
ifeq ($(shell ${SED} --version 2>&1 | grep -q GNU; echo $$?),1)
    $(error !!! GNU sed is required. If on OS X, use 'brew install gnu-sed'.)
endif

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

uv: ## Install uv
	@command -v uv &> /dev/null || { \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	  echo "uv has been installed."; \
	}

install-dev: uv ## Install all development dependencies
	@uv sync --all-extras --group dev
	@uv run pre-commit install

##@ Quality

verify: install-dev ## Run the same checks CI runs (pre-commit + lockfile)
	@uv lock --check
	@uv run pre-commit run --all-files
	@echo "All checks passed!"

format: ## Auto-format and fix lint issues
	@uv run --group dev ruff check --fix .
	@uv run --group dev ruff format .

##@ Testing

test-python: ## Run unit tests
	@uv sync --all-extras --group dev
	@uv run pytest --cov=kubeflow_mcp --cov-report=$(or $(report),term)

test-scripts: ## Run GitHub Actions script tests
	@uv sync --all-extras --group dev
	@uv run pytest .github/scripts/test_scripts.py -v

test: ## Run all tests (unit + integration)
	@uv sync --all-extras --group dev
	@uv run pytest tests/ kubeflow_mcp/ -v --tb=short

test-cov: ## Run tests with HTML coverage report
	@uv sync --all-extras --group dev
	@uv run pytest --cov=kubeflow_mcp --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

benchmark: ## Run the benchmark suite (excluded from the other test targets)
	@uv sync --all-extras --group dev
	@uv run pytest tests/benchmarks/ -m benchmark

##@ Dev Tools

.PHONY: conformance
conformance: install-dev ## Run MCP protocol conformance suite against a local HTTP server
	@echo "Starting kubeflow-mcp on http://localhost:8000/mcp (no cluster required)..."
	@KUBECONFIG=/nonexistent uv run kubeflow-mcp serve --transport http --no-banner > /tmp/kubeflow-mcp-conformance.log 2>&1 & \
	  SERVER_PID=$$!; \
	  trap "kill $$SERVER_PID 2>/dev/null || true" EXIT; \
	  timeout 60 bash -c 'until curl -sf -o /dev/null http://localhost:8000/mcp -X POST -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"make\",\"version\":\"1\"}}}"; do sleep 1; done'; \
	  npx -y @modelcontextprotocol/conformance@0.1.16 server \
	    --url http://localhost:8000/mcp \
	    --suite active \
	    --expected-failures tests/conformance/expected-failures.yaml

TRANSPORT ?= stdio

inspector: install-dev ## Launch MCP Inspector (TRANSPORT=stdio|http|sse)
ifeq ($(TRANSPORT),stdio)
	@npx @modelcontextprotocol/inspector uv run kubeflow-mcp serve
else ifeq ($(TRANSPORT),sse)
	@echo "Start the server first in another terminal:"
	@echo "  uv run kubeflow-mcp serve --transport sse"
	@echo ""
	@npx @modelcontextprotocol/inspector --transport sse --server-url $(or $(SERVER_URL),http://127.0.0.1:8000/sse)
else
	@echo "Start the server first in another terminal:"
	@echo "  uv run kubeflow-mcp serve --transport http"
	@echo ""
	@npx @modelcontextprotocol/inspector --transport http --server-url $(or $(SERVER_URL),http://127.0.0.1:8000/mcp)
endif

##@ Release

.PHONY: release
release: install-dev ## Bump version in __init__.py and server.json (VERSION=X.Y.Z[rcN])
	@if [ -z "$(VERSION)" ] || ! echo "$(VERSION)" | grep -E -q '^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$$'; then \
		echo "Error: VERSION must be set in X.Y.Z or X.Y.ZrcN format. Usage: make release VERSION=X.Y.Z[rcN]"; \
		exit 1; \
	fi
	@if [ ! -f server.json ]; then \
		echo "Error: server.json not found (required for MCP Registry metadata)"; \
		exit 1; \
	fi
	@$(SED) -i 's/^__version__ = ".*"/__version__ = "$(VERSION)"/' kubeflow_mcp/__init__.py
	@echo "Version bumped to $(VERSION) in kubeflow_mcp/__init__.py"
	@$(SED) -E -i 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?"/"version": "$(VERSION)"/g' server.json
	@echo "Version bumped to $(VERSION) in server.json"
	@if echo "$(VERSION)" | grep -E -q 'rc[0-9]+$$'; then \
		echo "Skipping changelog generation for RC release $(VERSION)"; \
	else \
		$(MAKE) changelog VERSION=$(VERSION); \
	fi
	@echo ""
	@echo "Release commit for $(VERSION) is ready."
	@echo "Review the changelog changes if needed, then commit with:"
	@echo "git add -A && git commit -s -m 'Prepare Release $(VERSION)'"

# Generate or prepend a changelog entry with git-cliff (Docker), matching kubeflow/sdk.
# Usage: make changelog VERSION=0.1.0
# Dry-run (stdout only): make changelog VERSION=0.1.0 DRY_RUN=1
# Optional: GITHUB_TOKEN=... (or export it) for contributor attribution / New Contributors.
# Scope: prefer PREV_TAG..upstream/release-X.Y, else --unreleased.
.PHONY: changelog
changelog: ## Generate changelog. Usage: make changelog VERSION=X.Y.Z [DRY_RUN=1]
	@if [ -z "$(VERSION)" ] || ! echo "$(VERSION)" | grep -E -q '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
		echo "Error: VERSION must be set in X.Y.Z format. Usage: make changelog VERSION=0.1.0"; \
		exit 1; \
	fi
	@git fetch upstream --tags --prune
	@MAJOR_MINOR=$$(echo "$(VERSION)" | cut -d. -f1,2); \
	CHANGELOG_PATH="CHANGELOG/CHANGELOG-$$MAJOR_MINOR.md"; \
	RELEASE_BRANCH="release-$$MAJOR_MINOR"; \
	RELEASE_SHA=$$(git rev-parse --verify --quiet "refs/remotes/upstream/$$RELEASE_BRANCH"); \
	if [ -n "$$RELEASE_SHA" ]; then \
		PREV_TAG=$$(git describe --tags --abbrev=0 --match '[0-9]*' --exclude '*rc*' "$$RELEASE_SHA" 2>/dev/null || true); \
		if [ -n "$$PREV_TAG" ]; then \
			CLIFF_SCOPE="$$PREV_TAG..$$RELEASE_SHA"; \
			echo "Generating changelog for $(VERSION) (range: $$PREV_TAG..$$RELEASE_BRANCH @ $$RELEASE_SHA)"; \
		else \
			CLIFF_SCOPE="--unreleased"; \
			echo "Generating changelog for $(VERSION) (no prior tag on $$RELEASE_BRANCH; using --unreleased)"; \
		fi; \
	elif [ ! -f "$$CHANGELOG_PATH" ]; then \
		CLIFF_SCOPE="--unreleased"; \
		echo "Generating changelog for $(VERSION) (new release line $$MAJOR_MINOR, branch $$RELEASE_BRANCH not created yet; using --unreleased)"; \
	else \
		echo "Error: branch $$RELEASE_BRANCH not found locally or on upstream, but $$CHANGELOG_PATH exists. Run: git fetch upstream $$RELEASE_BRANCH"; \
		exit 1; \
	fi; \
	mkdir -p CHANGELOG; \
	export GITHUB_TOKEN="$(GITHUB_TOKEN)"; \
	CLIFF_CMD="docker run --rm -u $$(id -u):$$(id -g) -v $(PROJECT_DIR):/app"; \
	if [ -n "$$GITHUB_TOKEN" ]; then \
		CLIFF_CMD="$$CLIFF_CMD -e GITHUB_TOKEN"; \
	fi; \
	CLIFF_CMD="$$CLIFF_CMD -w /app ghcr.io/orhun/git-cliff/git-cliff:latest $$CLIFF_SCOPE --tag $(VERSION)"; \
	# Prepend only when a prior release heading exists; stubs/empty files use -o
	# so the first entry is not written above a '# Changelog' intro.
	if [ "$(DRY_RUN)" = "1" ]; then \
		echo "DRY_RUN=1: printing changelog to stdout (not writing $$CHANGELOG_PATH)"; \
		$$CLIFF_CMD; \
	elif [ -f "$$CHANGELOG_PATH" ] && grep -qE '^# \[[0-9]+\.[0-9]+\.[0-9]+\]' "$$CHANGELOG_PATH"; then \
		$$CLIFF_CMD --prepend "$$CHANGELOG_PATH"; \
		echo "Changelog written to $$CHANGELOG_PATH"; \
	else \
		$$CLIFF_CMD -o "$$CHANGELOG_PATH"; \
		echo "Changelog written to $$CHANGELOG_PATH"; \
	fi

##@ Cleanup

clean: ## Remove all build and cache artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"
