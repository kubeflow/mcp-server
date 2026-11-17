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

"""Check, compile and inspect agent-written KFP pipeline code for ``create_pipeline``.

Compiling a KFP pipeline runs part of its source: the module's top level and the
body of the ``@dsl.pipeline`` function, which wires tasks together. ``@dsl.component``
bodies are only captured as text and run later in the cluster. So the code is
checked in three steps:

1. ``check_pipeline_code`` parses the source (nothing runs) and only accepts a small
   shape: ``kfp`` / ``typing`` imports, literal constants, ``@dsl.component``
   functions and exactly one ``@dsl.pipeline`` function whose body only calls
   components, task methods and ``dsl`` / ``kfp.kubernetes`` helpers. Component
   bodies are free, since they run in the cluster.
2. ``compile_pipeline_code`` compiles in a child interpreter (``python -I``) with a
   stripped environment and a timeout, never in the server process.
3. ``inspect_pipeline_spec`` reads the compiled YAML (it is never rewritten, so the
   uploaded file keeps KFP's usual header) and checks its images and accelerators.

The AST check is the boundary that keeps pipeline code from doing anything but
building a graph at compile time; the child process limits the damage if it misses
something.
"""

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import ToolError
from kubeflow_mcp.core.security import (
    MAX_GPU_PER_NODE,
    is_safe_python_code,
    validate_image_reference,
)
from kubeflow_mcp.pipelines.api._common import INSTALL_HINT

MAX_PIPELINE_CODE_CHARS = 100_000
MAX_SPEC_BYTES = 2_000_000
MAX_EXECUTORS = 50
MAX_ERRORS = 10
COMPILE_TIMEOUT_SECONDS = 120
MAX_COMPILE_ERROR_CHARS = 2_000
# Image kfp uses for a @dsl.component without base_image (kfp 2.17).
KFP_DEFAULT_BASE_IMAGE = "python:3.11"
_WORKER = Path(__file__).with_name("_compile_worker.py")
# Only what a Python interpreter needs to start; credentials and kubeconfig stay out.
_WORKER_ENV_KEYS = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL")

_DSL_TYPES = frozenset(
    {
        "Input",
        "Output",
        "InputPath",
        "OutputPath",
        "Artifact",
        "Dataset",
        "Model",
        "Metrics",
        "ClassificationMetrics",
        "SlicedClassificationMetrics",
        "HTML",
        "Markdown",
        "PipelineTaskFinalStatus",
    }
)
_DSL_DECORATORS = frozenset({"component", "pipeline"})
# Context managers usable as ``with dsl.X(...)`` in a pipeline body.
_DSL_CONTEXTS = frozenset({"If", "Elif", "Else", "Condition", "ParallelFor", "ExitHandler"})
# Plain calls usable as values in a pipeline body.
_DSL_CALLS = frozenset({"Collected", "OneOf", "importer"})
_DSL_PLACEHOLDERS = frozenset(
    {
        "PIPELINE_JOB_NAME_PLACEHOLDER",
        "PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER",
        "PIPELINE_JOB_ID_PLACEHOLDER",
        "PIPELINE_TASK_NAME_PLACEHOLDER",
        "PIPELINE_TASK_ID_PLACEHOLDER",
        "PIPELINE_TASK_EXECUTOR_OUTPUT_PATH_PLACEHOLDER",
        "PIPELINE_TASK_EXECUTOR_INPUT_PLACEHOLDER",
        "PIPELINE_ROOT_PLACEHOLDER",
        "PIPELINE_JOB_CREATE_TIME_UTC_PLACEHOLDER",
        "PIPELINE_JOB_SCHEDULE_TIME_UTC_PLACEHOLDER",
        "WORKSPACE_PATH_PLACEHOLDER",
    }
)
_DSL_NAMES = _DSL_TYPES | _DSL_DECORATORS | _DSL_CONTEXTS | _DSL_CALLS | _DSL_PLACEHOLDERS
# kfp.kubernetes helpers (kfp-kubernetes 2.17): they only record platform config on a task.
_KUBERNETES_NAMES = frozenset(
    {
        "CreatePVC",
        "DeletePVC",
        "add_ephemeral_volume",
        "add_node_affinity",
        "add_node_affinity_json",
        "add_node_selector",
        "add_node_selector_json",
        "add_pod_annotation",
        "add_pod_label",
        "add_toleration",
        "add_toleration_json",
        "empty_dir_mount",
        "mount_pvc",
        "set_image_pull_policy",
        "set_image_pull_secrets",
        "set_security_context",
        "set_timeout",
        "use_config_map_as_env",
        "use_config_map_as_volume",
        "use_field_path_as_env",
        "use_secret_as_env",
        "use_secret_as_volume",
    }
)
_TYPING_NAMES = frozenset({"NamedTuple", "List", "Dict", "Optional"})
_BUILTIN_TYPES = frozenset({"str", "int", "float", "bool", "dict", "list"})
_ALLOWED_FROM_IMPORTS: dict[str, frozenset[str]] = {
    "kfp": frozenset({"dsl", "kubernetes"}),
    "kfp.dsl": _DSL_NAMES,
    "kfp.kubernetes": _KUBERNETES_NAMES,
    "typing": _TYPING_NAMES,
}
# PipelineTask methods a pipeline body may call. They only set fields on the task.
_TASK_METHODS = frozenset(
    {
        "after",
        "ignore_upstream_failure",
        "set_caching_options",
        "set_cpu_request",
        "set_cpu_limit",
        "set_memory_request",
        "set_memory_limit",
        "set_accelerator_type",
        "set_accelerator_limit",
        "set_gpu_limit",
        "set_retry",
        "set_display_name",
        "set_env_variable",
        "add_node_selector_constraint",
        "set_container_image",
    }
)
_TASK_OUTPUT_ATTRS = frozenset({"output", "outputs"})
_COMPONENT_KWARGS = frozenset(
    {
        "base_image",
        "packages_to_install",
        "pip_index_urls",
        "pip_trusted_hosts",
        "install_kfp_package",
        "use_venv",
    }
)
_PIPELINE_KWARGS = frozenset({"name", "description", "display_name", "pipeline_root"})


@dataclass
class PipelineCodeSummary:
    """What the source declares, read from its AST without running it."""

    pipeline_function: str
    pipeline_name: str | None
    components: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    safety_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_function": self.pipeline_function,
            "pipeline_name": self.pipeline_name,
            "components": self.components,
            "parameters": self.parameters,
        }


class _CodeError(Exception):
    """A shape the checker does not accept; the message names the line."""

    def __init__(self, node: ast.AST, message: str) -> None:
        super().__init__(f"line {getattr(node, 'lineno', '?')}: {message}")


class _PipelineCodeChecker:
    """Walks the module AST and rejects anything outside the accepted shape."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.constants: dict[str, Any] = {}
        self.components: dict[str, dict[str, Any]] = {}
        self.pipelines: list[tuple[ast.FunctionDef, dict[str, Any]]] = []
        self.errors: list[str] = []

    # ── name resolution ────────────────────────────────────────────────────

    def qualified(self, node: ast.AST) -> str | None:
        """Return ``kfp.dsl.X`` / ``kfp.kubernetes.X`` / ``typing.X`` for *node*, if any."""
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = self.aliases.get(node.value.id)
            if module == "kfp.dsl" and node.attr in _DSL_NAMES:
                return f"kfp.dsl.{node.attr}"
            if module == "kfp.kubernetes" and node.attr in _KUBERNETES_NAMES:
                return f"kfp.kubernetes.{node.attr}"
        return None

    def static_value(self, node: ast.AST) -> Any:
        """Return the value of a literal or a top-level constant, else raise."""
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.static_value(item) for item in node.elts]
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            raise _CodeError(node, "expected a literal value or a top-level constant") from None

    # ── top level ──────────────────────────────────────────────────────────

    def check_module(self, tree: ast.Module) -> None:
        for index, stmt in enumerate(tree.body):
            try:
                self.check_top_level(stmt, is_first=index == 0)
            except _CodeError as e:
                self.errors.append(str(e))
                if len(self.errors) >= MAX_ERRORS:
                    return
        if not self.errors and len(self.pipelines) != 1:
            self.errors.append(
                f"expected exactly one @dsl.pipeline function, found {len(self.pipelines)}"
            )

    def check_top_level(self, stmt: ast.stmt, *, is_first: bool) -> None:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if is_first and isinstance(stmt.value.value, str):
                return
            raise _CodeError(stmt, "only a module docstring may stand alone at the top level")
        if isinstance(stmt, ast.ImportFrom):
            self.check_import_from(stmt)
        elif isinstance(stmt, ast.Import):
            raise _CodeError(
                stmt,
                "use 'from kfp import dsl' (and 'from kfp import kubernetes'); "
                "put other imports inside @dsl.component functions",
            )
        elif isinstance(stmt, ast.Assign):
            self.check_constant(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            self.check_function(stmt)
        else:
            raise _CodeError(
                stmt,
                f"{type(stmt).__name__} is not allowed at the top level; only imports, "
                "literal constants, @dsl.component and @dsl.pipeline functions are",
            )

    def check_import_from(self, stmt: ast.ImportFrom) -> None:
        allowed = _ALLOWED_FROM_IMPORTS.get(stmt.module or "")
        if stmt.level or allowed is None:
            raise _CodeError(
                stmt,
                f"import from '{stmt.module}' is not allowed at the top level; allowed: "
                f"{', '.join(sorted(_ALLOWED_FROM_IMPORTS))}. Import other packages "
                "inside @dsl.component functions",
            )
        for alias in stmt.names:
            if alias.name not in allowed:
                raise _CodeError(stmt, f"'{alias.name}' cannot be imported from '{stmt.module}'")
            local = alias.asname or alias.name
            if local in self.aliases or local in self.constants:
                raise _CodeError(stmt, f"name '{local}' is defined twice")
            self.aliases[local] = f"{stmt.module}.{alias.name}"

    def check_constant(self, stmt: ast.Assign) -> None:
        names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if len(names) != len(stmt.targets):
            raise _CodeError(stmt, "top-level assignments must assign to plain names")
        value = self.static_value(stmt.value)
        for name in names:
            if name in self.aliases or name in self.components:
                raise _CodeError(stmt, f"name '{name}' is defined twice")
            self.constants[name] = value

    # ── functions ──────────────────────────────────────────────────────────

    def decorator_kind(self, func: ast.FunctionDef) -> tuple[str, dict[str, Any]]:
        if len(func.decorator_list) != 1:
            raise _CodeError(
                func,
                f"function '{func.name}' needs exactly one decorator: "
                "@dsl.component or @dsl.pipeline",
            )
        decorator = func.decorator_list[0]
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        kind = (self.qualified(target) or "").removeprefix("kfp.dsl.")
        if kind not in _DSL_DECORATORS:
            raise _CodeError(
                decorator, f"function '{func.name}' must be a @dsl.component or a @dsl.pipeline"
            )
        allowed = _COMPONENT_KWARGS if kind == "component" else _PIPELINE_KWARGS
        options: dict[str, Any] = {}
        if isinstance(decorator, ast.Call):
            if decorator.args:
                raise _CodeError(decorator, f"pass @dsl.{kind} options as keywords")
            for keyword in decorator.keywords:
                if keyword.arg not in allowed:
                    raise _CodeError(
                        decorator,
                        f"@dsl.{kind} option '{keyword.arg}' is not supported; "
                        f"allowed: {', '.join(sorted(allowed))}",
                    )
                options[keyword.arg] = self.static_value(keyword.value)
        return kind, options

    def check_function(self, func: ast.FunctionDef) -> None:
        kind, options = self.decorator_kind(func)
        if func.name in self.aliases or func.name in self.constants or func.name in self.components:
            raise _CodeError(func, f"name '{func.name}' is defined twice")
        self.check_signature(func)
        if kind == "component":
            self.components[func.name] = {
                "name": func.name,
                "base_image": options.get("base_image") or KFP_DEFAULT_BASE_IMAGE,
                "packages_to_install": options.get("packages_to_install") or [],
            }
            return
        self.pipelines.append((func, options))
        scope = {a.arg for a in [*func.args.args, *func.args.kwonlyargs]}
        _PipelineBodyChecker(self, scope).check_block(func.body, allow_docstring=True)

    def check_signature(self, func: ast.FunctionDef) -> None:
        args = func.args
        if args.vararg or args.kwarg or args.posonlyargs:
            raise _CodeError(func, f"'{func.name}' may only take named parameters")
        for arg in [*args.args, *args.kwonlyargs]:
            self.check_annotation(arg.annotation, func.name)
        self.check_annotation(func.returns, func.name)
        for default in [*args.defaults, *args.kw_defaults]:
            if default is not None:
                self.static_value(default)

    def check_annotation(self, node: ast.AST | None, func_name: str) -> None:
        """Accept KFP parameter and artifact types; annotations run at definition time."""
        if node is None or (isinstance(node, ast.Constant) and node.value is None):
            return
        if isinstance(node, ast.Name) and node.id in _BUILTIN_TYPES:
            return
        resolved = self.qualified(node) or ""
        if resolved.removeprefix("kfp.dsl.") in _DSL_TYPES or resolved.startswith("typing."):
            return
        if isinstance(node, ast.Subscript):
            self.check_annotation(node.value, func_name)
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for element in elements:
                self.check_annotation(element, func_name)
            return
        if isinstance(node, ast.Call) and resolved == "" and self._is_typed_call(node):
            return
        raise _CodeError(
            node,
            f"type annotation '{ast.unparse(node)}' in '{func_name}' is not supported; use "
            "str, int, float, bool, list, dict, dsl.Input[...], dsl.Output[...] or NamedTuple",
        )

    def _is_typed_call(self, node: ast.Call) -> bool:
        """``NamedTuple("Outputs", [("x", float)])`` or ``InputPath("Dataset")``."""
        name = (self.qualified(node.func) or "").rsplit(".", 1)[-1]
        if node.keywords or not node.args:
            return False
        if name in ("InputPath", "OutputPath"):
            return len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
        if name != "NamedTuple" or len(node.args) != 2:
            return False
        if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[1], ast.List):
            return False
        for item in node.args[1].elts:
            if not (
                isinstance(item, ast.Tuple)
                and len(item.elts) == 2
                and isinstance(item.elts[0], ast.Constant)
            ):
                return False
            self.check_annotation(item.elts[1], "NamedTuple")
        return True


class _PipelineBodyChecker:
    """Checks a ``@dsl.pipeline`` body, which runs on the server at compile time."""

    def __init__(self, module: _PipelineCodeChecker, scope: set[str]) -> None:
        self.module = module
        self.scope = set(scope)
        self.loop_vars: set[str] = set()

    def check_block(self, body: list[ast.stmt], *, allow_docstring: bool = False) -> None:
        for index, stmt in enumerate(body):
            is_docstring = (
                allow_docstring
                and index == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
            if not is_docstring:
                self.check_stmt(stmt)

    def check_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Pass):
            return
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self.check_call(stmt.value)
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise _CodeError(stmt, "assign each task to one plain name")
            self.check_expr(stmt.value)
            self.define(stmt.targets[0].id, stmt)
        elif isinstance(stmt, ast.With):
            self.check_with(stmt)
        else:
            raise _CodeError(
                stmt,
                f"{type(stmt).__name__} is not allowed in a @dsl.pipeline function; it may "
                "only call components, set task options and use 'with dsl.If / "
                "dsl.ParallelFor / dsl.ExitHandler'",
            )

    def define(self, name: str, node: ast.AST) -> None:
        if name in self.module.aliases or name in self.module.components:
            raise _CodeError(node, f"'{name}' shadows an import or a component")
        self.scope.add(name)

    def check_with(self, stmt: ast.With) -> None:
        for item in stmt.items:
            context = item.context_expr
            target = context.func if isinstance(context, ast.Call) else None
            kind = (self.module.qualified(target) if target else None) or ""
            if kind.removeprefix("kfp.dsl.") not in _DSL_CONTEXTS:
                raise _CodeError(
                    stmt,
                    "only 'with dsl.If / dsl.Elif / dsl.Else / dsl.Condition / "
                    "dsl.ParallelFor / dsl.ExitHandler' blocks are allowed",
                )
            self.check_arguments(context)
            if item.optional_vars is not None:
                if not isinstance(item.optional_vars, ast.Name):
                    raise _CodeError(stmt, "bind the 'with' target to one plain name")
                self.define(item.optional_vars.id, stmt)
                self.loop_vars.add(item.optional_vars.id)
        self.check_block(stmt.body)

    def check_arguments(self, call: ast.Call) -> None:
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                raise _CodeError(arg, "*args are not allowed in a @dsl.pipeline function")
            self.check_expr(arg)
        for keyword in call.keywords:
            if keyword.arg is None:
                raise _CodeError(call, "**kwargs are not allowed in a @dsl.pipeline function")
            self.check_expr(keyword.value)

    def check_call(self, call: ast.Call) -> None:
        func = call.func
        if isinstance(func, ast.Name) and func.id in self.module.components:
            self.check_arguments(call)
            return
        kind = self.module.qualified(func) or ""
        dsl_name = kind.removeprefix("kfp.dsl.")
        if (
            dsl_name in _DSL_CALLS
            or dsl_name in _DSL_CONTEXTS
            or kind.startswith("kfp.kubernetes.")
        ):
            self.check_arguments(call)
            return
        if isinstance(func, ast.Attribute) and func.attr in _TASK_METHODS:
            self.check_expr(func.value)
            self.check_arguments(call)
            return
        raise _CodeError(
            call,
            f"calling '{ast.unparse(func)}' is not allowed in a @dsl.pipeline function; call "
            "components, task methods (after, set_cpu_limit, ...) or dsl / kubernetes helpers",
        )

    def check_expr(self, node: ast.AST) -> None:  # noqa: C901 - one branch per node type
        if isinstance(node, ast.Constant):
            return
        if isinstance(node, ast.Call):
            self.check_call(node)
        elif isinstance(node, ast.Name):
            self.check_name(node)
        elif isinstance(node, ast.Attribute):
            self.check_attribute(node)
        elif isinstance(node, ast.Subscript):
            self.check_expr(node.value)
            self.check_expr(node.slice)
        elif isinstance(node, (ast.List, ast.Tuple)):
            for element in node.elts:
                self.check_expr(element)
        elif isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise _CodeError(node, "'**' in a dict is not allowed")
            for child in [*node.keys, *node.values]:
                self.check_expr(child)
        elif isinstance(node, ast.Compare):
            for child in [node.left, *node.comparators]:
                self.check_expr(child)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            self.check_expr(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            self.check_expr(node.left)
            self.check_expr(node.right)
        elif isinstance(node, ast.JoinedStr):
            self.check_joined_str(node)
        else:
            raise _CodeError(
                node, f"{type(node).__name__} is not allowed in a @dsl.pipeline function"
            )

    def check_joined_str(self, node: ast.JoinedStr) -> None:
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                if value.format_spec is not None:
                    raise _CodeError(node, "format specs are not allowed in f-strings here")
                self.check_expr(value.value)

    def check_name(self, node: ast.Name) -> None:
        if node.id in self.scope or node.id in self.module.constants:
            return
        resolved = (self.module.aliases.get(node.id) or "").removeprefix("kfp.dsl.")
        if resolved in _DSL_TYPES or resolved in _DSL_PLACEHOLDERS:
            return
        if node.id in self.module.components:
            raise _CodeError(node, f"call component '{node.id}' instead of referencing it")
        raise _CodeError(node, f"unknown name '{node.id}' in the @dsl.pipeline function")

    def check_attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise _CodeError(node, f"private attribute '{node.attr}' is not allowed")
        resolved = (self.module.qualified(node) or "").removeprefix("kfp.dsl.")
        if resolved in _DSL_TYPES or resolved in _DSL_PLACEHOLDERS:
            return
        is_loop_field = isinstance(node.value, ast.Name) and node.value.id in self.loop_vars
        if node.attr in _TASK_OUTPUT_ATTRS or is_loop_field:
            self.check_expr(node.value)
            return
        raise _CodeError(
            node,
            f"attribute '{node.attr}' is not allowed here; read task results with "
            ".output or .outputs['name']",
        )


def check_pipeline_code(code: Any) -> tuple[PipelineCodeSummary | None, ToolError | None]:
    """Parse and check pipeline source without running any of it."""
    if not isinstance(code, str) or not code.strip():
        return None, ToolError(
            error="pipeline_code is required", error_code=ErrorCode.VALIDATION_ERROR
        )
    if len(code) > MAX_PIPELINE_CODE_CHARS:
        return None, ToolError(
            error=f"pipeline_code is too large ({len(code)} chars, max {MAX_PIPELINE_CODE_CHARS})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError) as e:
        line = getattr(e, "lineno", None)
        return None, ToolError(
            error=f"pipeline_code has a syntax error{f' at line {line}' if line else ''}: "
            f"{getattr(e, 'msg', e)}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    checker = _PipelineCodeChecker()
    try:
        checker.check_module(tree)
    except RecursionError:
        checker.errors.append("pipeline_code is nested too deeply")
    if checker.errors:
        return None, ToolError(
            error=f"pipeline_code is not supported: {checker.errors[0]}",
            error_code=ErrorCode.VALIDATION_ERROR,
            details={"problems": checker.errors},
            hint="Read pipelines://guides/authoring for the accepted pipeline shape",
        )

    func, options = checker.pipelines[0]
    defaults = [None] * (len(func.args.args) - len(func.args.defaults)) + list(func.args.defaults)
    parameters = [
        {
            "name": arg.arg,
            "type": ast.unparse(arg.annotation) if arg.annotation is not None else None,
            "default": checker.static_value(default) if default is not None else None,
            "required": default is None,
        }
        for arg, default in zip(func.args.args, defaults, strict=True)
    ]
    safe, reason = is_safe_python_code(code)
    name = options.get("name")
    return PipelineCodeSummary(
        pipeline_function=func.name,
        pipeline_name=name if isinstance(name, str) else None,
        components=list(checker.components.values()),
        parameters=parameters,
        safety_warnings=[] if safe else [reason],
    ), None


@dataclass
class CompiledPipeline:
    """A compiled pipeline: the YAML file to upload and its parsed documents."""

    path: str
    documents: list[Any]
    size_bytes: int


def _worker_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _WORKER_ENV_KEYS if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _short(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > MAX_COMPILE_ERROR_CHARS:
        text = "..." + text[-MAX_COMPILE_ERROR_CHARS:]
    return text


def compile_pipeline_code(
    code: str, workdir: str, timeout: float = COMPILE_TIMEOUT_SECONDS
) -> tuple[CompiledPipeline | None, ToolError | None]:
    """Compile *code* to pipeline YAML in a child interpreter, inside *workdir*.

    Call ``check_pipeline_code`` first. The caller owns *workdir* and removes it.
    """
    source = os.path.join(workdir, "pipeline_source.py")
    output = os.path.join(workdir, "pipeline.yaml")
    if not sys.executable:
        return None, ToolError(
            error="Cannot compile pipelines: the server's Python interpreter path is unknown",
            error_code=ErrorCode.SDK_ERROR,
        )
    with open(source, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(_WORKER), source, output],
            cwd=workdir,
            env=_worker_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, ToolError(
            error=f"Compiling the pipeline took longer than {timeout:g}s and was stopped",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Keep the @dsl.pipeline body to component calls and task options",
        )
    except OSError as e:
        return None, ToolError(
            error=f"Could not start the pipeline compiler: {e}",
            error_code=ErrorCode.SDK_ERROR,
        )
    if proc.returncode == 3:
        return None, ToolError(
            error=f"Kubeflow Pipelines support is not installed: {_short(proc.stderr)}",
            error_code=ErrorCode.SDK_ERROR,
            hint=INSTALL_HINT,
        )
    if proc.returncode != 0:
        return None, ToolError(
            error=f"The pipeline failed to compile: {_short(proc.stderr) or 'no error output'}",
            error_code=ErrorCode.VALIDATION_ERROR,
            hint="Fix the pipeline code and preview again; see pipelines://guides/authoring",
        )
    return _read_compiled(output)


def _read_compiled(output: str) -> tuple[CompiledPipeline | None, ToolError | None]:
    try:
        size = os.path.getsize(output)
        if size > MAX_SPEC_BYTES:
            return None, ToolError(
                error=f"The compiled pipeline is too large ({size} bytes, max {MAX_SPEC_BYTES})",
                error_code=ErrorCode.VALIDATION_ERROR,
            )
        with open(output, encoding="utf-8") as f:
            documents = [doc for doc in yaml.safe_load_all(f) if doc is not None]
    except (OSError, yaml.YAMLError) as e:
        return None, ToolError(
            error=f"Could not read the compiled pipeline: {e}",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    return CompiledPipeline(path=output, documents=documents, size_bytes=size), None


def _accelerator_count(resources: dict[str, Any]) -> int:
    accelerator = resources.get("accelerator") or {}
    raw = accelerator.get("resourceCount") or accelerator.get("count") or 0
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return -1


def inspect_pipeline_spec(documents: list[Any]) -> tuple[dict[str, Any] | None, ToolError | None]:
    """Check the compiled pipeline spec and summarize what it will run."""
    spec = documents[0] if documents else None
    if not isinstance(spec, dict) or not {"pipelineInfo", "root", "deploymentSpec"} <= set(spec):
        return None, ToolError(
            error="The compiler did not produce a KFP pipeline spec",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    executors = (spec.get("deploymentSpec") or {}).get("executors") or {}
    if len(executors) > MAX_EXECUTORS:
        return None, ToolError(
            error=f"The pipeline has too many steps ({len(executors)}, max {MAX_EXECUTORS})",
            error_code=ErrorCode.VALIDATION_ERROR,
        )
    steps = []
    for name, executor in sorted(executors.items()):
        container = (executor or {}).get("container") or {}
        image = container.get("image")
        image_err = validate_image_reference(image, f"image of step '{name}'")
        if image is None or image_err is not None:
            return None, image_err or ToolError(
                error=f"Step '{name}' has no container image",
                error_code=ErrorCode.VALIDATION_ERROR,
            )
        resources = container.get("resources") or {}
        accelerators = _accelerator_count(resources)
        if not 0 <= accelerators <= MAX_GPU_PER_NODE:
            return None, ToolError(
                error=f"Step '{name}' asks for {accelerators} accelerators "
                f"(allowed 0 to {MAX_GPU_PER_NODE})",
                error_code=ErrorCode.VALIDATION_ERROR,
            )
        steps.append({"executor": name, "image": image, "accelerators": accelerators})

    parameters = ((spec.get("root") or {}).get("inputDefinitions") or {}).get("parameters") or {}
    return {
        "pipeline_name": (spec.get("pipelineInfo") or {}).get("name"),
        "steps": steps,
        "images": sorted({step["image"] for step in steps}),
        "parameters": {
            name: {
                "type": (value or {}).get("parameterType"),
                "default": (value or {}).get("defaultValue"),
                "optional": bool((value or {}).get("isOptional")),
            }
            for name, value in parameters.items()
        },
        "has_platform_config": len(documents) > 1,
    }, None
