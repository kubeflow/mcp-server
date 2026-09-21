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

.PHONY: help uv install-dev verify format test-python test-scripts test test-e2e test-cov benchmark clean inspector release changelog

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

.PHONY: update-schema-snapshot
update-schema-snapshot: ## Regenerate the MCP tool schema snapshot baseline (after an approved schema change)
	@uv sync --all-extras --group dev
	@UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest tests/conformance/tool_schema_snapshot_test.py -q
	@echo "Schema snapshot updated: tests/conformance/snapshots/tool_schema_platform_admin.json"
	@echo "Review the diff and commit it alongside your schema change."

test-scripts: ## Run GitHub Actions script tests
	@uv sync --all-extras --group dev
	@uv run pytest .github/scripts/test_scripts.py -v

test: ## Run all tests (unit + integration)
	@uv sync --all-extras --group dev
	@uv run pytest tests/ kubeflow_mcp/ -v --tb=short

test-e2e: ## Run Kubernetes E2E tests (requires KUBEFLOW_MCP_E2E=true and Kubeconfig)
	@uv sync --all-extras --group dev
	@KUBEFLOW_MCP_E2E=true uv run pytest tests/e2e/test_kubernetes_e2e.py -v

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

# Generate or prepend a changelog entry with git-cliff, matching kubeflow/sdk.
# Usage: make changelog VERSION=0.1.0
# Dry-run (stdout only): make changelog VERSION=0.1.0 DRY_RUN=1
# Optional: GITHUB_TOKEN=... (or export it) for contributor attribution / New Contributors.
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
	RELEASE_SHA=$$(git rev-parse --verify --quiet "refs/remotes/upstream/$$RELEASE_BRANCH" || true); \
	if [ -z "$$RELEASE_SHA" ]; then \
		if [ -f "$$CHANGELOG_PATH" ]; then \
			echo "Error: branch $$RELEASE_BRANCH not found on upstream, but $$CHANGELOG_PATH exists. Run: git fetch upstream $$RELEASE_BRANCH"; \
			exit 1; \
		fi; \
		RELEASE_SHA=$$(git rev-parse HEAD); \
		echo "Branch $$RELEASE_BRANCH does not exist yet (new release line $$MAJOR_MINOR, created by the release workflow); using HEAD"; \
	fi; \
	PATCH=$$(echo "$(VERSION)" | cut -d. -f3); \
	if [ "$$PATCH" -gt 0 ]; then \
		PREV_TAG="$$(echo "$(VERSION)" | cut -d. -f1,2).$$((PATCH - 1))"; \
	else \
		PREV_MINOR=$$(( $$(echo "$(VERSION)" | cut -d. -f2) - 1 )); \
		PREV_TAG=$$(git tag --list "$$(echo "$(VERSION)" | cut -d. -f1).$$PREV_MINOR.*" | grep -vE -- '(rc)' | sort -t. -k3,3nr | head -1 || true); \
	fi; \
	if [ -z "$$PREV_TAG" ]; then \
		echo "Error: cannot determine the previous release tag for $(VERSION)"; \
		exit 1; \
	fi; \
	if [ "$$(git rev-list --count $$PREV_TAG..$$RELEASE_SHA 2>/dev/null || echo 0)" -eq 0 ]; then \
		RELEASE_SHA=$$(git rev-parse HEAD); \
		echo "Generating changelog for $(VERSION) (release branch at $$PREV_TAG; range: $$PREV_TAG..HEAD)"; \
	else \
		echo "Generating changelog for $(VERSION) (range: $$PREV_TAG..$$RELEASE_SHA)"; \
	fi; \
	mkdir -p CHANGELOG; \
	touch "$$CHANGELOG_PATH"; \
	export GITHUB_TOKEN="$(GITHUB_TOKEN)"; \
	if command -v git-cliff >/dev/null 2>&1; then \
		CLIFF_CMD="git-cliff $$PREV_TAG..$$RELEASE_SHA --tag $(VERSION) --config cliff.toml"; \
	else \
		CLIFF_CMD="docker run --rm -u $$(id -u):$$(id -g) -e HOME=/tmp -e GITHUB_TOKEN"; \
		CLIFF_CMD="$$CLIFF_CMD -v $(PROJECT_DIR):/app -w /app ghcr.io/orhun/git-cliff/git-cliff:latest"; \
		CLIFF_CMD="$$CLIFF_CMD $$PREV_TAG..$$RELEASE_SHA --tag $(VERSION)"; \
	fi; \
	if grep -qF "# [$(VERSION)]" "$$CHANGELOG_PATH" 2>/dev/null; then \
		echo "Changelog already contains $(VERSION) — skipping (delete the entry to regenerate)"; \
	elif [ "$(DRY_RUN)" = "1" ]; then \
		echo "DRY_RUN=1: printing changelog to stdout (not writing $$CHANGELOG_PATH)"; \
		$$CLIFF_CMD; \
	else \
		$$CLIFF_CMD --prepend "$$CHANGELOG_PATH"; \
		echo "Changelog written to $$CHANGELOG_PATH"; \
	fi

##@ Cleanup

clean: ## Remove all build and cache artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"
