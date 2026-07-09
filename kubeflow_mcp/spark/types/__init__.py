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

"""Serialization helpers for Spark SDK types.

The SDK returns ``SparkConnectInfo`` dataclasses whose fields include an
``Enum`` state and a ``datetime`` timestamp — neither is JSON-serializable as
returned. These helpers convert SDK objects into plain, MCP-safe dicts without
importing ``kubeflow.spark`` (so this module loads without the Spark extra).
"""

from typing import Any


def session_info_to_dict(info: Any) -> dict[str, Any]:
    """Convert a ``SparkConnectInfo`` object into a JSON-safe dict.

    Uses duck typing (``getattr``) rather than importing the SDK dataclass so
    the conversion works even when ``kubeflow[spark]`` is not importable at the
    call site and so it stays resilient to additive SDK field changes.

    Args:
        info: A ``kubeflow.spark.types.types.SparkConnectInfo`` instance.

    Returns:
        dict with ``name``, ``namespace``, ``state``, ``driver_pod_name``,
        ``pod_ip``, ``service_name`` and ``creation_timestamp`` (ISO 8601).
    """
    state = getattr(info, "state", None)
    # SparkConnectState is a ``str`` Enum — prefer its ``.value`` but fall back
    # to ``str()`` so a plain string state still serializes cleanly.
    state_str = getattr(state, "value", None) or (str(state) if state is not None else None)

    created = getattr(info, "creation_timestamp", None)
    created_str = created.isoformat() if hasattr(created, "isoformat") else created

    return {
        "name": getattr(info, "name", None),
        "namespace": getattr(info, "namespace", None),
        "state": state_str,
        "driver_pod_name": getattr(info, "driver_pod_name", None),
        "pod_ip": getattr(info, "pod_ip", None),
        "service_name": getattr(info, "service_name", None),
        "creation_timestamp": created_str,
    }


__all__ = ["session_info_to_dict"]
