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

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY kubeflow_mcp ./kubeflow_mcp
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

RUN groupadd --gid 65532 kubeflow-mcp \
 && useradd  --uid 65532 --gid 65532 --no-create-home --shell /sbin/nologin kubeflow-mcp

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Sensible defaults for in-cluster operation; override at deploy time
    MCP_TRANSPORT=http \
    KUBEFLOW_MCP_LOG_FORMAT=json

EXPOSE 8000

USER 65532:65532

ENTRYPOINT ["kubeflow-mcp", "serve", "--transport", "http"]
