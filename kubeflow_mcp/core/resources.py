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

Client modules may additionally declare ``CLIENT_SKILLS`` (SEP-2640 Agent
Skills): each skill maps ``skill://`` URIs onto existing ``CLIENT_RESOURCES``
URIs.  Aliased URIs share the same cached content, and an aggregate
``skill://index.json`` resource lists every skill the server exposes.

Content is cached at server startup — no repeated disk reads.
"""

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

logger = logging.getLogger(__name__)

SKILL_INDEX_URI = "skill://index.json"


def _make_handler(cache: dict[str, str], uri: str, description: str):
    def handler() -> str:
        return cache[uri]

    handler.__doc__ = description
    handler.__name__ = re.sub(r"\W", "_", uri.rsplit("/", 1)[-1])
    return handler


def register_resources(mcp: "FastMCP", loaded_modules: dict[str, Any]) -> bool:
    """Register MCP resources from client modules with startup caching.

    All resources are always registered regardless of persona — persona
    filtering only controls which resources are *referenced* in instructions.
    """
    cache: dict[str, str] = {}
    complete = True
    descriptions: dict[str, str] = {}
    skills_index: list[dict[str, Any]] = []

    for module in loaded_modules.values():
        client_resources = getattr(module, "CLIENT_RESOURCES", {})
        if not client_resources:
            continue

        resources_dir = Path(module.__file__).parent
        for uri, (filename, description) in client_resources.items():
            path = resources_dir / filename
            if not path.exists():
                complete = False
                logger.warning(
                    f"MCP resource file not found: {path}. "
                    f"Ensure {filename} exists relative to {resources_dir}"
                )
                continue

            cache[uri] = path.read_text(encoding="utf-8")
            descriptions[uri] = description
            mcp.resource(uri)(_make_handler(cache, uri, description))
            logger.debug(f"Registered resource: {uri}")

        # Agent Skills (SEP-2640): alias guide content under skill:// URIs.
        for skill_name, spec in getattr(module, "CLIENT_SKILLS", {}).items():
            skill_resources: list[dict[str, str]] = []
            for skill_uri, source_uri in spec.get("aliases", {}).items():
                if source_uri not in cache:
                    complete = False
                    logger.warning(
                        f"Skill alias {skill_uri} points to unknown or unloaded "
                        f"resource {source_uri}; skipping"
                    )
                    continue
                cache[skill_uri] = cache[source_uri]
                description = descriptions[source_uri]
                mcp.resource(skill_uri)(_make_handler(cache, skill_uri, description))
                skill_resources.append({"uri": skill_uri, "description": description})
                logger.debug(f"Registered skill alias: {skill_uri} -> {source_uri}")
            if not skill_resources:
                logger.warning(
                    f"Skill {skill_name!r} has no resolvable aliases; omitting from index"
                )
                continue
            skills_index.append(
                {
                    "name": skill_name,
                    "description": spec.get("description", ""),
                    "resources": skill_resources,
                }
            )

    if skills_index:
        cache[SKILL_INDEX_URI] = json.dumps({"version": 1, "skills": skills_index}, indent=2)
        mcp.resource(SKILL_INDEX_URI, mime_type="application/json")(
            _make_handler(
                cache,
                SKILL_INDEX_URI,
                "Index of Agent Skills served by this server (SEP-2640).",
            )
        )
        logger.debug(f"Registered skill index: {SKILL_INDEX_URI}")

    logger.info(f"Registered {len(cache)} MCP resources")
    return complete
