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

"""MCP resources loaded dynamically from client modules.

Each client module declares a ``CLIENT_RESOURCES`` dict mapping MCP URIs to
``(relative_path, description)`` tuples.  Resource files live alongside the
client module (e.g. ``trainer/resources/*.md``).

Content is cached at server startup — no repeated disk reads.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_resources(mcp: "FastMCP", loaded_modules: dict[str, Any]) -> None:
    """Register MCP resources from client modules with startup caching.

    All resources are always registered regardless of persona — persona
    filtering only controls which resources are *referenced* in instructions.
    """
    cache: dict[str, str] = {}

    for module in loaded_modules.values():
        client_resources = getattr(module, "CLIENT_RESOURCES", {})
        if not client_resources:
            continue

        resources_dir = Path(module.__file__).parent
        for uri, (filename, description) in client_resources.items():
            path = resources_dir / filename
            if not path.exists():
                logger.warning(
                    f"MCP resource file not found: {path}. "
                    f"Ensure {filename} exists relative to {resources_dir}"
                )
                continue

            cache[uri] = path.read_text(encoding="utf-8")

            def _make_handler(cached_uri: str, desc: str):
                def handler() -> str:
                    return cache[cached_uri]

                handler.__doc__ = desc
                handler.__name__ = cached_uri.rsplit("/", 1)[-1].replace("-", "_")
                return handler

            mcp.resource(uri)(_make_handler(uri, description))
            logger.debug(f"Registered resource: {uri}")

    logger.info(f"Registered {len(cache)} MCP resources")
