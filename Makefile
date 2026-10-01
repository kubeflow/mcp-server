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

.PHONY: help uv install-dev verify format test-python test-scripts test test-e2e test-cov benchmark clean inspector release

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

# Export credentials to recipes and recursive make invocations without placing
# their values in command-line arguments.
export GITHUB_TOKEN

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

CONTAINER_RUNTIME ?= docker

.PHONY: release changelog
release: install-dev ## Create a release commit. Usage: export GITHUB_TOKEN=<token> && make release VERSION=X.Y.Z
	@if [ -z "$(VERSION)" ] || ! echo "$(VERSION)" | grep -E -q '^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$$'; then \
		echo "Error: VERSION must be set in X.Y.Z or X.Y.ZrcN format. Usage: export GITHUB_TOKEN=<token> && make release VERSION=X.Y.Z[rcN]"; \
		exit 1; \
	fi
	@if [ ! -f server.json ]; then \
		echo "Error: server.json not found (required for MCP Registry metadata)"; \
		exit 1; \
	fi
	@if echo "$(VERSION)" | grep -E -q 'rc[0-9]+$$'; then \
		echo "Skipping changelog generation for RC release $(VERSION)"; \
		$(SED) -i 's/^__version__ = ".*"/__version__ = "$(VERSION)"/' kubeflow_mcp/__init__.py; \
		echo "Version bumped to $(VERSION) in kubeflow_mcp/__init__.py"; \
		$(SED) -E -i 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?"/"version": "$(VERSION)"/g' server.json; \
		echo "Version bumped to $(VERSION) in server.json"; \
	else \
		$(MAKE) changelog VERSION=$(VERSION); \
		$(SED) -i 's/^__version__ = ".*"/__version__ = "$(VERSION)"/' kubeflow_mcp/__init__.py && \
		echo "Version bumped to $(VERSION) in kubeflow_mcp/__init__.py" && \
		$(SED) -E -i 's/"version": "[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?"/"version": "$(VERSION)"/g' server.json && \
		echo "Version bumped to $(VERSION) in server.json"; \
	fi
	@echo ""
	@echo "Release commit for $(VERSION) is ready."
	@echo "Review the changelog changes if needed, then commit with:"
	@echo "git add -A && git commit -s -m 'Prepare Release $(VERSION)'"

changelog: ## Generate changelog. Usage: make changelog VERSION=X.Y.Z [DRY_RUN=1]
	@if [ -z "$(VERSION)" ] || ! echo "$(VERSION)" | grep -E -q '^[0-9]+\.[0-9]+\.[0-9]+$$'; then \
		echo "Error: VERSION must be set in X.Y.Z format. Usage: make changelog VERSION=0.1.0"; \
		exit 1; \
	fi
	@git fetch upstream --tags --prune
	@MAJOR_MINOR=$$(echo "$(VERSION)" | cut -d. -f1,2); \
	CHANGELOG_PATH="CHANGELOG/CHANGELOG-$$MAJOR_MINOR.md"; \
	TARGET_MAJOR=$$(echo "$(VERSION)" | cut -d. -f1); \
	TARGET_MINOR=$$(echo "$(VERSION)" | cut -d. -f2); \
	TARGET_PATCH=$$(echo "$(VERSION)" | cut -d. -f3); \
	PREV_TAG=""; \
	for candidate in $$(git tag --list --sort=-version:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$'); do \
		CANDIDATE_MAJOR=$${candidate%%.*}; \
		CANDIDATE_MINOR=$${candidate#*.}; CANDIDATE_MINOR=$${CANDIDATE_MINOR%%.*}; \
		CANDIDATE_PATCH=$${candidate##*.}; \
		IS_OLDER=0; \
		if [ "$$CANDIDATE_MAJOR" -lt "$$TARGET_MAJOR" ] || \
			[ "$$CANDIDATE_MAJOR" -eq "$$TARGET_MAJOR" ] && [ "$$CANDIDATE_MINOR" -lt "$$TARGET_MINOR" ]; then \
			IS_OLDER=1; \
		elif [ "$$CANDIDATE_MAJOR" -eq "$$TARGET_MAJOR" ] && \
			[ "$$CANDIDATE_MINOR" -eq "$$TARGET_MINOR" ] && [ "$$CANDIDATE_PATCH" -lt "$$TARGET_PATCH" ]; then \
			IS_OLDER=1; \
		fi; \
		if [ "$$IS_OLDER" -eq 1 ]; then \
			PREV_TAG="$$candidate"; \
			break; \
		fi; \
	done; \
	if [ -z "$$PREV_TAG" ]; then \
		PREV_REF=""; \
		CLIFF_SCOPE="--unreleased"; \
		echo "No older stable release tag found; using unreleased commits"; \
	else \
		if ! git rev-parse --verify --quiet "refs/tags/$$PREV_TAG" >/dev/null; then \
			echo "Error: selected stable release tag is unavailable: $$PREV_TAG"; \
			exit 1; \
		fi; \
		PREV_REF="refs/tags/$$PREV_TAG"; \
		if ! git merge-base --is-ancestor "$$PREV_REF" HEAD; then \
			PREV_REF="$$(git log -n 1 --format=%H -S "__version__ = \"$$PREV_TAG\"" -- kubeflow_mcp/__init__.py)"; \
			if [ -z "$$PREV_REF" ]; then \
				echo "Error: previous release tag $$PREV_TAG is not an ancestor and its version bump was not found"; \
				exit 1; \
			fi; \
			echo "Previous release tag is not an ancestor; using version bump $$PREV_REF as the changelog base"; \
		fi; \
		CLIFF_SCOPE="$$PREV_REF..$$(git rev-parse HEAD)"; \
	fi; \
	echo "Generating changelog for $(VERSION) (range: $$CLIFF_SCOPE)"; \
	CONTAINER_USER_ARGS="-u $$(id -u):$$(id -g)"; \
	if [ "$(CONTAINER_RUNTIME)" = "podman" ]; then CONTAINER_USER_ARGS=""; fi; \
	CLIFF_CMD="$(CONTAINER_RUNTIME) run --rm $$CONTAINER_USER_ARGS -v $(PROJECT_DIR):/app"; \
	if [ -n "$${GITHUB_TOKEN:-}" ]; then \
		CLIFF_CMD="$$CLIFF_CMD -e GITHUB_TOKEN"; \
	fi; \
	CLIFF_OFFLINE=""; \
	if [ "$(OFFLINE)" = "1" ]; then CLIFF_OFFLINE="--offline"; fi; \
	CLIFF_CMD="$$CLIFF_CMD -w /app ghcr.io/orhun/git-cliff/git-cliff:latest $$CLIFF_OFFLINE $$CLIFF_SCOPE --tag $(VERSION)"; \
	if [ "$(DRY_RUN)" = "1" ]; then \
		echo "DRY_RUN=1: printing changelog to stdout (not writing $$CHANGELOG_PATH)"; \
		$$CLIFF_CMD; \
	elif [ -f "$$CHANGELOG_PATH" ] && grep -qE '^# \[[0-9]+\.[0-9]+\.[0-9]+\]' "$$CHANGELOG_PATH"; then \
		$$CLIFF_CMD --prepend "$$CHANGELOG_PATH"; \
		echo "Changelog written to $$CHANGELOG_PATH"; \
	else \
		mkdir -p CHANGELOG; \
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
