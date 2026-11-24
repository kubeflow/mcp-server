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

"""Hugging Face Hub reference checks shared by the trainer and pipelines clients.

``hf://`` references are checked in two steps: a local format check that is safe in
a preview, and a Hub lookup on submission that only rejects a repo the Hub confirms
missing. Other schemes (``s3://`` and so on) are left alone.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError

logger = logging.getLogger(__name__)

HF_SCHEME = "hf://"
_HF_REPO_PART_RE = re.compile(r"[a-zA-Z0-9._-]+")
HF_MODEL_EXAMPLE = "hf://[<org>/]<model>"
HF_DATASET_EXAMPLE = "hf://<org>/<dataset>[/<path>]"


def hf_repo_id(uri: str, *, dataset: bool) -> str | None:
    """Return the repo ID of an ``hf://`` URI, or None if it is malformed.

    Models and datasets differ because the Trainer initializers parse them
    differently. The model initializer passes the whole path to
    ``snapshot_download``, so an unscoped model such as ``hf://gpt2`` works. The
    dataset initializer reads the second path segment, so a dataset needs
    ``org/name`` and may add a subpath (``hf://org/ds/subpath``).
    """
    parts = uri.removeprefix(HF_SCHEME).rstrip("/").split("/")
    # A subpath is not checked against the repo pattern, so block path traversal here.
    if "" in parts or any(part in (".", "..") for part in parts):
        return None
    if dataset:
        repo_parts = parts[:2] if len(parts) >= 2 else []
    else:
        repo_parts = parts if len(parts) <= 2 else []
    if not repo_parts or not all(_HF_REPO_PART_RE.fullmatch(p) for p in repo_parts):
        return None
    return "/".join(repo_parts)


def validate_hf_reference(uri: str, field: str, *, dataset: bool) -> ToolError | None:
    """Check that an ``hf://`` reference is well formed. Local check, safe in a preview.

    Non-HF references are left alone, including ones that contain whitespace.
    """
    if (uri.startswith(HF_SCHEME) or uri.strip().startswith(HF_SCHEME)) and any(
        ch.isspace() for ch in uri
    ):
        return ToolError(
            error=f"Invalid Hugging Face {field} reference '{uri}' (must not contain whitespace)",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    if uri.startswith(HF_SCHEME) and hf_repo_id(uri, dataset=dataset) is None:
        example = HF_DATASET_EXAMPLE if dataset else HF_MODEL_EXAMPLE
        return ToolError(
            error=f"Invalid Hugging Face {field} reference '{uri}' (expected {example})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None


def is_confirmed_missing(error: Exception) -> bool:
    """Return True when a Hub error confirms the repo does not exist.

    The Hub raises RepositoryNotFoundError for private or gated repos and missing
    auth too (401, GatedRepoError), and a 401 is ambiguous: an anonymous request
    gets it for a missing repo as well. Only a plain 404 is treated as a
    nonexistent repo, so an access failure is never reported as bad input.
    """
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if not isinstance(error, RepositoryNotFoundError) or isinstance(error, GatedRepoError):
        return False
    return getattr(getattr(error, "response", None), "status_code", None) == 404


def hub_confirms_missing(fetch: Callable[..., Any], repo_id: str, kind: str) -> bool:
    """Return True only when the Hub confirms ``repo_id`` missing; log other failures."""
    try:
        fetch(repo_id, timeout=10)
    except Exception as e:
        if is_confirmed_missing(e):
            return True
        logger.warning("Could not verify %s %s, submitting anyway: %s", kind, repo_id, e)
    return False


def check_hf_dataset_exists(dataset: str) -> ToolError | None:
    """On submission, reject an ``hf://`` dataset the Hub confirms missing (404).

    Gated, private and unreachable datasets do not block submission, and neither does
    a Hub outage. Without an ``HF_TOKEN`` on the MCP host the Hub answers 401 even for
    a missing repo, so in that setup this check never rejects anything. Call
    ``validate_hf_reference`` first: a malformed reference is not looked up.
    """
    if not dataset.startswith(HF_SCHEME):
        return None
    repo_id = hf_repo_id(dataset, dataset=True)
    if repo_id is None:
        return None
    # Imported at call time so tests can patch ``huggingface_hub.dataset_info``.
    from huggingface_hub import dataset_info

    if hub_confirms_missing(dataset_info, repo_id, "dataset"):
        return ToolError(
            error=f"Hugging Face dataset '{repo_id}' does not exist",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return None
