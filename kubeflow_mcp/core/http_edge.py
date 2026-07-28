# Copyright The Kubeflow Authors
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

"""HTTP liveness and readiness probes for network transports."""

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


def register_probe_routes(mcp: FastMCP, *, is_ready: bool = True) -> None:
    """Register unauthenticated probes on the FastMCP HTTP application."""

    @mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def health(_request: Request) -> Response:
        return JSONResponse({"status": "healthy"})

    @mcp.custom_route("/ready", methods=["GET"], include_in_schema=False)
    async def ready(_request: Request) -> Response:
        if not is_ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})
