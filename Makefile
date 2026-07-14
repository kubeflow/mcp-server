SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: uv
uv: ## Install UV
	@command -v uv &> /dev/null || { \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	  echo "✅ uv has been installed."; \
	}

.PHONY: verify
verify: install-dev ## Run linting and formatting checks
	@uv lock --check
	@uv run ruff check .
	@uv run ruff format --check .

.PHONY: format
format: ## Auto-fix lint and formatting issues
	@uv run ruff check --fix .
	@uv run ruff format .

.PHONY: test-python
test-python: ## Run Python unit tests
	@uv sync --all-extras
	@uv run pytest --cov=kubeflow_mcp --cov-report=$(or $(report),term)

.PHONY: install-dev
install-dev: uv ## Install dependencies and tools
	@echo "Syncing dependencies with uv..."
	@uv sync --all-extras
	@echo "Environment is ready."

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

.PHONY: inspector
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
