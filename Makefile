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

.PHONY: help uv install-dev verify format test-python test test-cov clean inspector

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

uv: ## Install uv
	@command -v uv &> /dev/null || { \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	  echo "uv has been installed."; \
	}

install-dev: uv ## Install all development dependencies
	@uv sync --all-extras
	@uv run pre-commit install

##@ Quality

verify: ## Run linting and formatting checks
	@uv lock --check
	@uv run ruff check .
	@uv run ruff format --check .

format: ## Auto-format and fix lint issues
	@uv run ruff check --fix .
	@uv run ruff format .

##@ Testing

test-python: ## Run unit tests
	@uv sync --all-extras
	@uv run pytest --cov=kubeflow_mcp --cov-report=$(or $(report),term)

test: ## Run all tests (unit + integration)
	@uv sync --all-extras
	@uv run pytest tests/ kubeflow_mcp/ -v --tb=short

test-cov: ## Run tests with HTML coverage report
	@uv sync --all-extras
	@uv run pytest --cov=kubeflow_mcp --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

##@ Dev Tools

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

##@ Cleanup

clean: ## Remove all build and cache artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"
