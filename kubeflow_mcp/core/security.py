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

"""Input validation and security checks."""

import ast
import re
from typing import Any

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError

K8S_NAME_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MAX_NAME_LENGTH = 63


def validate_k8s_name(name: str, field: str = "name") -> ToolError | None:
    """Validate Kubernetes resource name.

    Returns ToolError if invalid, None if valid.
    """
    if not name:
        return ToolError(
            error=f"{field} cannot be empty",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if len(name) > MAX_NAME_LENGTH:
        return ToolError(
            error=f"{field} too long (max {MAX_NAME_LENGTH})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if not K8S_NAME_PATTERN.match(name):
        return ToolError(
            error=f"{field} must be lowercase alphanumeric with hyphens",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"value": name, "pattern": K8S_NAME_PATTERN.pattern},
        )

    return None


def validate_namespace(namespace: str) -> ToolError | None:
    """Validate namespace name format."""
    return validate_k8s_name(namespace, "namespace")


def check_namespace_allowed(namespace: str | None) -> ToolError | None:
    """Check if namespace is allowed by policy.

    When *namespace* is ``None`` the effective default namespace from the
    TrainerClient backend is resolved and checked against the allowlist so
    that implicit-default usage cannot bypass namespace restrictions.

    Args:
        namespace: Namespace to check. ``None`` resolves to the effective default.

    Returns:
        ToolError if namespace is restricted, None if allowed
    """
    # Import here to avoid circular import
    from kubeflow_mcp.core.policy import get_allowed_namespaces

    allowed = get_allowed_namespaces()
    if allowed is None:
        return None

    effective = namespace
    if effective is None:
        try:
            from kubeflow_mcp.common.utils import get_trainer_effective_namespace

            effective = get_trainer_effective_namespace(None)
        except Exception:
            return ToolError(
                error="Cannot resolve effective namespace; refusing request (fail closed)",
                error_code=ErrorCode.PERMISSION_DENIED,
            )

    if effective not in allowed:
        return ToolError(
            error=f"Namespace '{effective}' not allowed by policy",
            error_code=ErrorCode.PERMISSION_DENIED,
            details={"allowed_namespaces": allowed, "effective_namespace": effective},
        )

    return None


_DANGEROUS_CALLS = frozenset(
    {
        "eval",
        "exec",
        "__import__",
        "compile",
        "execfile",
    }
)
_DANGEROUS_ATTR_CALLS = {
    "os": frozenset({"system", "popen", "popen2", "popen3", "popen4"}),
    "subprocess": frozenset({"call", "run", "Popen", "check_call", "check_output"}),
    "shutil": frozenset({"rmtree"}),
}
_DANGEROUS_IMPORTS = frozenset(
    {
        "ctypes",
        "socket",
    }
)
_DANGEROUS_DUNDER = frozenset(
    {
        "__builtins__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__import__",
        "__loader__",
    }
)


class _ScriptSafetyVisitor(ast.NodeVisitor):
    """AST visitor that collects dangerous patterns from Python scripts."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_CALLS:
            self.warnings.append(f"Dangerous call: {node.func.id}() at line {node.lineno}")
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                attr = node.func.attr
                blocked = _DANGEROUS_ATTR_CALLS.get(module, frozenset())
                if attr in blocked:
                    self.warnings.append(f"Dangerous call: {module}.{attr}() at line {node.lineno}")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _DANGEROUS_IMPORTS:
                self.warnings.append(f"Dangerous import: {alias.name} at line {node.lineno}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            top = node.module.split(".")[0]
            if top in _DANGEROUS_IMPORTS:
                self.warnings.append(f"Dangerous import: from {node.module} at line {node.lineno}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in _DANGEROUS_DUNDER:
            self.warnings.append(f"Dangerous attribute access: {node.attr} at line {node.lineno}")
        self.generic_visit(node)


def is_safe_python_code(code: str) -> tuple[bool, str]:
    """AST-based scan for dangerous patterns in training scripts.

    .. warning::

        This is a **best-effort** filter. It catches common dangerous patterns
        via AST walking but is bypassable (e.g. ``getattr`` indirection) and
        must **not** be relied upon as a security boundary. The script runs
        with full privileges inside the training pod.

    **Flagged patterns** (via AST, not substring matching):

    - Calls: ``eval``, ``exec``, ``compile``, ``__import__``
    - Module calls: ``os.system``, ``os.popen``, ``subprocess.*``, ``shutil.rmtree``
    - Imports: ``ctypes``, ``socket``
    - Dunder access: ``__builtins__``, ``__subclasses__``, ``__globals__``, ``__code__``

    Standard ML imports (``os``, ``torch``, ``transformers``) are allowed.

    Returns:
        ``(is_safe, reason)`` — reason lists the first flagged pattern or ``"OK"``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    visitor = _ScriptSafetyVisitor()
    visitor.visit(tree)

    if visitor.warnings:
        return False, "; ".join(visitor.warnings)

    return True, "OK"


def truncate_log_output(output: str, max_length: int = 10000) -> str:
    """Truncate log output to a safe length for MCP responses."""
    if len(output) > max_length:
        output = output[:max_length] + f"\n... (truncated, {len(output)} total chars)"
    return output


def validate_resource_limits(
    cpu: str | None,
    memory: str | None,
    gpu: int | None,
) -> ToolError | None:
    """Validate resource limit specifications."""
    if cpu:
        if not re.match(r"^\d+(\.\d+)?m?$", cpu):
            return ToolError(
                error="Invalid CPU format (use '100m', '1', or '0.5')",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    if memory:
        if not re.match(r"^\d+(Ki|Mi|Gi|Ti)?$", memory):
            return ToolError(
                error="Invalid memory format (use '256Mi' or '1Gi')",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    if gpu is not None and gpu < 0:
        return ToolError(
            error="GPU count cannot be negative",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    return None


MAX_SCRIPT_SIZE = 1_000_000  # 1 MB
MAX_PACKAGES = 50
MAX_NODES = 100
MAX_GPU_PER_NODE = 16
MAX_BATCH_SIZE = 1024
MAX_EPOCHS = 1_000
MAX_LORA_RANK = 256


def validate_training_bounds(
    *,
    batch_size: int | None = None,
    epochs: int | None = None,
    num_nodes: int | None = None,
    gpu_per_node: int | None = None,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    script: str | None = None,
    packages: list[str] | None = None,
) -> ToolError | None:
    """Validate training parameter bounds. Returns ToolError on violation, None if OK."""
    checks: list[tuple[str, int | None, int, int]] = [
        ("batch_size", batch_size, 1, MAX_BATCH_SIZE),
        ("epochs", epochs, 1, MAX_EPOCHS),
        ("num_nodes", num_nodes, 1, MAX_NODES),
        ("gpu_per_node", gpu_per_node, 0, MAX_GPU_PER_NODE),
        ("lora_rank", lora_rank, 1, MAX_LORA_RANK),
        ("lora_alpha", lora_alpha, 1, MAX_LORA_RANK * 4),
    ]
    for name, value, lo, hi in checks:
        if value is not None and not (lo <= value <= hi):
            return ToolError(
                error=f"{name} must be between {lo} and {hi}, got {value}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

    if lora_dropout is not None and not (0.0 <= lora_dropout <= 1.0):
        return ToolError(
            error=f"lora_dropout must be between 0.0 and 1.0, got {lora_dropout}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if script is not None and not script.strip():
        return ToolError(
            error="Script cannot be empty",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if script is not None and len(script) > MAX_SCRIPT_SIZE:
        return ToolError(
            error=f"Script too large ({len(script)} bytes, max {MAX_SCRIPT_SIZE})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    if packages is not None and len(packages) > MAX_PACKAGES:
        return ToolError(
            error=f"Too many packages ({len(packages)}, max {MAX_PACKAGES})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    return None


_SENSITIVE_EXACT = {"hf_token", "access_token", "secret_access_key", "s3_secret_access_key"}
_SENSITIVE_SUBSTRINGS = {
    "password",
    "secret",
    "credential",
    "authorization",
    "bearer",
    "kubeconfig",
    "sa_token",
    "private_key",
    "certificate",
    "client_cert",
    "tls_cert",
    "tls_key",
}
_SAFE_KEYS = {"public_key", "keyword", "key_format", "key_name"}


def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    """Mask sensitive fields in data for logging."""
    result: dict[str, Any] = {}

    for k, v in data.items():
        k_lower = k.lower()
        if k_lower in _SAFE_KEYS:
            result[k] = v
        elif k_lower in _SENSITIVE_EXACT:
            result[k] = "***"
        elif any(s in k_lower for s in _SENSITIVE_SUBSTRINGS):
            result[k] = "***"
        elif k_lower.endswith("_key") and k_lower not in _SAFE_KEYS:
            result[k] = "***"
        elif k_lower.endswith("_token"):
            result[k] = "***"
        elif isinstance(v, dict):
            result[k] = mask_sensitive_data(v)
        elif isinstance(v, list):
            result[k] = [mask_sensitive_data(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v

    return result
