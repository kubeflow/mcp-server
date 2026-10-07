#!/usr/bin/env bash
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
#
# Runtime image smoke tests.
#
# Usage:
#   image_smoke_test.sh <IMAGE_TAG>
#
# Runs black-box checks against a locally-built Docker image to verify
# expected runtime behavior (CLI, permissions, health, and MCP protocol).
#
# Note:
# Container logs are captured to /tmp/smoke-container.log before teardown
# to allow CI steps to inspect them on failure.

set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' is required to run these tests. Please install it." >&2
    exit 1
fi

# Configuration

IMAGE="${1:?Usage: image_smoke_test.sh <IMAGE_TAG>}"
CONTAINER_NAME="${SMOKE_CONTAINER_NAME:-kubeflow-mcp-smoke}"
HOST_PORT="${SMOKE_TEST_PORT:-8000}"
MAX_WAIT="${SMOKE_MAX_WAIT:-60}"
LOG_FILE="${SMOKE_LOG_FILE:-/tmp/smoke-container.log}"

MCP_URL="http://localhost:${HOST_PORT}/mcp"

passed=0
failed=0

# Helpers

_log()  { echo "[smoke] $*"; }
_pass() { echo "  ✓  $1"; passed=$((passed + 1)); }
_fail() {
    echo "  ✗  $1" >&2
    [ -z "${2:-}" ] || echo "     → ${2}" >&2
    failed=$((failed + 1))
}

# Extract JSON from an SSE "data: <json>" response, or return the body as-is
# when the server already replied with application/json (non-streaming).
_parse_mcp_body() {
    local body="$1"
    local sse_line
    # grep returns 1 on no match; silence it with || true to satisfy pipefail.
    sse_line=$(printf '%s\n' "$body" | grep "^data: " | head -1 || true)
    if [ -n "$sse_line" ]; then
        printf '%s\n' "$sse_line" | sed 's/^data: //'
    else
        printf '%s\n' "$body"
    fi
}

# Poll GET /health until the server responds with HTTP 200 or we time out.
# /health is exempt from DNS rebinding middleware (transport_security.py) so
# it is usable as a readiness probe even before the MCP session is open.
_wait_for_server() {
    local port="$1" max="$2"
    _log "Waiting for server on port ${port} (timeout: ${max}s) ..."
    local i=0
    while [ "$i" -lt "$max" ]; do
        if curl -sf -o /dev/null "http://localhost:${port}/health" 2>/dev/null; then
            _log "Server is ready (after ${i}s)."
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    _log "ERROR: Server did not become healthy after ${max}s." >&2
    return 1
}

# Cleanup

_cleanup() {
    # Capture logs first so the CI "dump logs on failure" step can read them
    # even after the container is gone.
    _log "Capturing container logs → ${LOG_FILE}"
    docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 || true
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
}
trap _cleanup EXIT

# Static tests (no server required)

_log "════ Static Tests ════════════════════════════════════════════════════"

# Test 1: CLI version
_log "Test 1: cli-version"
cli_exit=0
cli_out=$(docker run --rm --entrypoint kubeflow-mcp "$IMAGE" --version 2>&1) || cli_exit=$?
# click.version_option emits "kubeflow-mcp, version X.Y.Z"
if [ "$cli_exit" -eq 0 ] && printf '%s\n' "$cli_out" | grep -qE "[0-9]+\.[0-9]+\.[0-9]+"; then
    _pass "cli-version: ${cli_out}"
else
    _fail "cli-version" "exit=${cli_exit}, output: ${cli_out}"
fi

# Test 2: Non-root user
_log "Test 2: non-root-user"
uid_exit=0
uid_out=$(docker run --rm --entrypoint id "$IMAGE" -u 2>&1) || uid_exit=$?
if [ "$uid_exit" -eq 0 ] && [ "$uid_out" = "65532" ]; then
    _pass "non-root-user: uid=${uid_out}"
else
    _fail "non-root-user" "expected uid=65532, got: ${uid_out} (exit=${uid_exit})"
fi

# Test 3: No dev/test packages
_log "Test 3: no-dev-packages"
dev_found=""
for pkg in pytest ruff pre_commit; do
    if docker run --rm --entrypoint python "$IMAGE" -c "import $pkg" > /dev/null 2>&1; then
        dev_found="${dev_found} ${pkg}"
    fi
done

if [ -z "$dev_found" ]; then
    _pass "no-dev-packages: pytest, ruff, pre-commit are absent"
else
    _fail "no-dev-packages" "unexpected dev packages in image:${dev_found}"
fi


# Test 4: OTel absent from base image
_log "Test 4: no-otel-in-base"
otel_exit=0
docker run --rm --entrypoint python "$IMAGE" -c "import opentelemetry.sdk" > /dev/null 2>&1 \
    || otel_exit=$?
if [ "$otel_exit" -ne 0 ]; then
    _pass "no-otel-in-base: ImportError as expected (opentelemetry is an optional dep group)"
else
    _fail "no-otel-in-base" "opentelemetry was importable — should not be present in base image"
fi

# Server tests

_log "════ Server Tests ════════════════════════════════════════════════════"

_log "Starting container: ${CONTAINER_NAME}"
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${HOST_PORT}:8000" \
    -e KUBECONFIG=/nonexistent \
    -e KUBEFLOW_MCP_DNS_REBINDING_PROTECTION=false \
    "$IMAGE"

_wait_for_server "$HOST_PORT" "$MAX_WAIT"

# Test 5: /health endpoint
_log "Test 5: health-endpoint"
health_exit=0
health_resp=$(curl -sf --max-time 5 "http://localhost:${HOST_PORT}/health" 2>&1) \
    || health_exit=$?
if [ "$health_exit" -eq 0 ] && \
   printf '%s\n' "$health_resp" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    _pass "health-endpoint: $(printf '%s\n' "$health_resp" | jq -c .)"
else
    _fail "health-endpoint" "exit=${health_exit}, response: ${health_resp}"
fi

# Test 6: /ready endpoint
_log "Test 6: ready-endpoint"
ready_exit=0
ready_resp=$(curl -sf --max-time 5 "http://localhost:${HOST_PORT}/ready" 2>&1) \
    || ready_exit=$?
if [ "$ready_exit" -eq 0 ] && \
   printf '%s\n' "$ready_resp" | jq -e '.status == "ready"' > /dev/null 2>&1; then
    _pass "ready-endpoint: $(printf '%s\n' "$ready_resp" | jq -c .)"
else
    _fail "ready-endpoint" "exit=${ready_exit}, response: ${ready_resp}"
fi

# Test 7: MCP initialize
_log "Test 7: mcp-initialize"
headers_file=$(mktemp)
init_exit=0
init_raw=$(curl -sf --max-time 15 -X POST "$MCP_URL" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -D "$headers_file" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"1"}}}' \
    2>&1) || init_exit=$?

init_json=$(_parse_mcp_body "$init_raw")
session_id=""

if [ "$init_exit" -eq 0 ] && \
   printf '%s\n' "$init_json" | jq -e '.result.protocolVersion' > /dev/null 2>&1; then
    proto=$(printf '%s\n' "$init_json" | jq -r '.result.protocolVersion')
    _pass "mcp-initialize: protocolVersion=${proto}"
    session_id=$(grep -i "^mcp-session-id:" "$headers_file" \
        | tr -d '\r' | awk '{print $2}' | tr -d '[:space:]' || true)
    _log "  → session ID: ${session_id:-<none>}"
else
    _fail "mcp-initialize" \
        "exit=${init_exit}, response: $(printf '%s\n' "$init_json" | head -c 200)"
fi
rm -f "$headers_file"

# Test 8: MCP tools/list
_log "Test 8: mcp-tools-list"
if [ -z "$session_id" ]; then
    _fail "mcp-tools-list" "skipped — no session ID from Test 7 (mcp-initialize)"
else
    # notifications/initialized is a fire-and-forget notification (no id field,
    # no response expected).  Ignore errors; some server versions may close the
    # SSE stream before we read the body.
    curl -s --max-time 5 -X POST "$MCP_URL" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -H "Mcp-Session-Id: ${session_id}" \
        -d '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' \
        > /dev/null 2>&1 || true

    tools_exit=0
    tools_raw=$(curl -sf --max-time 15 -X POST "$MCP_URL" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json, text/event-stream" \
        -H "Mcp-Session-Id: ${session_id}" \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
        2>&1) || tools_exit=$?

    tools_json=$(_parse_mcp_body "$tools_raw")
    # Extract tool names; tolerate jq parse failures (empty string on error)
    tool_names=$(printf '%s\n' "$tools_json" | jq -r '.result.tools[].name' 2>/dev/null \
        || true)

    tools_ok=true
    for expected in health_check list_training_jobs; do
        if printf '%s\n' "$tool_names" | grep -qxF "$expected" 2>/dev/null; then
            _log "  → found: ${expected}"
        else
            _log "  → MISSING: ${expected}"
            tools_ok=false
        fi
    done

    if $tools_ok; then
        tool_count=$(printf '%s\n' "$tool_names" | grep -c '.' || true)
        _pass "mcp-tools-list: health_check and list_training_jobs present (${tool_count} tools total)"
    else
        _fail "mcp-tools-list" \
            "exit=${tools_exit}, tools found: $(printf '%s\n' "$tool_names" | tr '\n' ' ')"
    fi
fi

# Summary

echo ""
echo "══════════════════════════════════════════════════════════════════════"
total=$((passed + failed))
if [ "$failed" -eq 0 ]; then
    printf "Results: %d/%d passed — all smoke tests passed ✓\n" "$passed" "$total"
else
    printf "Results: %d/%d passed — %d test(s) FAILED ✗\n" "$passed" "$total" "$failed"
fi
echo "══════════════════════════════════════════════════════════════════════"

[ "$failed" -eq 0 ]
