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

"""Tests for the pipeline code check, the compile worker and the spec inspection.

The checker is the boundary that keeps agent-written code from running anything on
the server at compile time, so most tests here try to get code past it.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from kubeflow_mcp.pipelines.api import _compile, _compile_worker
from kubeflow_mcp.pipelines.api._compile import (
    KFP_DEFAULT_BASE_IMAGE,
    check_pipeline_code,
    compile_pipeline_code,
    inspect_pipeline_spec,
)
from kubeflow_mcp.pipelines.conftest import PIPELINE_CODE

HEADER = """\
from typing import NamedTuple

from kfp import dsl, kubernetes
from kfp.dsl import Dataset, Input, Output


@dsl.component
def step(x: int, data: Output[Dataset]):
    pass


@dsl.component(base_image="python:3.11-slim")
def use(data: Input[Dataset]) -> NamedTuple("Outputs", [("score", float), ("label", str)]):
    return (1.0, "ok")

"""


def _pipeline(body: str, params: str = "n: int = 1") -> str:
    """A module with HEADER and one pipeline whose body is *body*."""
    return (
        HEADER
        + f"\n@dsl.pipeline\ndef p({params}):\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
    )


def _error(code: str) -> str:
    summary, err = check_pipeline_code(code)
    assert summary is None, "expected the code to be rejected"
    assert err.error_code == "VALIDATION_ERROR"
    return err.error


class TestAccepted:
    def test_sample_pipeline_summary(self):
        summary, err = check_pipeline_code(PIPELINE_CODE)

        assert err is None
        assert summary.pipeline_function == "churn"
        assert summary.pipeline_name == "churn-prediction"
        assert summary.components == [
            {
                "name": "load",
                "base_image": "python:3.11-slim",
                "packages_to_install": ["pandas==2.2.3"],
            },
            {"name": "train", "base_image": "python:3.11-slim", "packages_to_install": []},
        ]
        assert summary.parameters == [
            {"name": "rows", "type": "int", "default": 100, "required": False},
            {"name": "epochs", "type": "int", "default": 2, "required": False},
        ]
        assert summary.safety_warnings == []

    def test_component_without_base_image_uses_the_kfp_default(self):
        summary, _ = check_pipeline_code(_pipeline("step(x=n)"))
        assert summary.components[0]["base_image"] == KFP_DEFAULT_BASE_IMAGE

    def test_required_parameter_is_reported(self):
        summary, _ = check_pipeline_code(_pipeline("step(x=n)", params="n: int"))
        assert summary.parameters == [
            {"name": "n", "type": "int", "default": None, "required": True}
        ]

    @pytest.mark.parametrize(
        "body",
        [
            '"""Docstring."""\nt = step(x=n)\nuse(data=t.outputs["data"]).after(t)',
            "t = step(x=n)\nt.set_cpu_limit('2').set_memory_limit('4G').set_caching_options(False)",
            "t = step(x=n)\nt.set_accelerator_type('nvidia.com/gpu').set_accelerator_limit(1)",
            "t = step(x=n)\nt.set_env_variable('A', 'b').set_retry(2).set_display_name('s')",
            "t = step(x=n)\nwith dsl.If(t.output == 'x'):\n    step(x=1)",
            "t = step(x=n)\nwith dsl.Condition(n > 2):\n    step(x=-1)",
            "with dsl.ParallelFor(items=[{'a': 1}, {'a': 2}]) as item:\n    step(x=item.a)",
            "with dsl.ParallelFor(items=[1, 2], parallelism=1) as i:\n    t = step(x=i)\nuse(data=dsl.Collected(t.outputs['data']))",
            "t = step(x=n)\nwith dsl.ExitHandler(t):\n    step(x=n + 1)",
            "t = step(x=n)\nkubernetes.use_secret_as_env(t, secret_name='s', secret_key_to_env={'k': 'K'})",
            "t = step(x=n)\nkubernetes.set_image_pull_secrets(t, ['regcred'])",
            "t = step(x=n)\nt.set_env_variable('RUN', f'{dsl.PIPELINE_JOB_NAME_PLACEHOLDER}-x')",
            "i = dsl.importer(artifact_uri='gs://b/d', artifact_class=dsl.Dataset)\nuse(data=i.output)",
            "pass",
        ],
    )
    def test_pipeline_body_shapes(self, body):
        summary, err = check_pipeline_code(_pipeline(body))
        assert err is None, err and err.details

    def test_aliased_imports_and_constants(self):
        code = textwrap.dedent(
            '''
            """Module docstring."""
            from kfp.dsl import component as comp, pipeline, Output, Model, InputPath

            IMAGE = "python:3.11-slim"
            PACKAGES = ["torch==2.5.1"]

            @comp(base_image=IMAGE, packages_to_install=PACKAGES)
            def train(lr: float, model: Output[Model], cfg: dict = {"a": 1}, tags: list = []):
                pass

            @comp
            def read(path: InputPath("Dataset")) -> str:
                return ""

            @pipeline(name="aliased", description="d")
            def p(lr: float = 0.1):
                train(lr=lr)
            '''
        )
        summary, err = check_pipeline_code(code)
        assert err is None, err and err.details
        assert summary.components[0]["base_image"] == "python:3.11-slim"
        assert summary.components[0]["packages_to_install"] == ["torch==2.5.1"]
        assert summary.pipeline_name == "aliased"

    def test_component_bodies_are_not_restricted_but_are_scanned(self):
        code = (
            HEADER.replace(
                "def step(x: int, data: Output[Dataset]):\n    pass",
                "def step(x: int, data: Output[Dataset]):\n"
                "    import subprocess\n"
                "    subprocess.run(['torchrun', 'train.py'], check=True)",
            )
            + "\n@dsl.pipeline\ndef p():\n    step(x=1)\n"
        )

        summary, err = check_pipeline_code(code)

        assert err is None
        assert summary.safety_warnings and "subprocess.run" in summary.safety_warnings[0]


class TestRejectedTopLevel:
    @pytest.mark.parametrize(
        ("code", "message"),
        [
            ("import os\n", "use 'from kfp import dsl'"),
            ("import kfp\n", "use 'from kfp import dsl'"),
            ("from os import system\n", "import from 'os' is not allowed"),
            ("from kfp import compiler\n", "'compiler' cannot be imported from 'kfp'"),
            ("from kfp.dsl import component_factory\n", "cannot be imported from 'kfp.dsl'"),
            ("from kfp.dsl import notebook_component\n", "cannot be imported from 'kfp.dsl'"),
            ("from __future__ import annotations\n", "import from '__future__'"),
            ("from . import x\n", "is not allowed at the top level"),
            ("print('x')\n", "Expr is not allowed at the top level"),
            ("x = open('/etc/passwd').read()\n", "expected a literal value"),
            ("x = 'a' * 10**9\n", "expected a literal value"),
            ("x, y = 1, 2\n", "plain names"),
            ("if True:\n    pass\n", "If is not allowed at the top level"),
            ("class A:\n    pass\n", "ClassDef is not allowed"),
            ("async def f():\n    pass\n", "AsyncFunctionDef is not allowed"),
            ("def f():\n    pass\n", "needs exactly one decorator"),
            ("x = 1\n'not a docstring'\n", "only a module docstring"),
        ],
    )
    def test_rejected(self, code, message):
        assert message in _error(HEADER + code + "\n@dsl.pipeline\ndef p():\n    pass\n")

    def test_shadowing_an_import_is_rejected(self):
        assert "defined twice" in _error("from kfp import dsl\ndsl = 1\n")

    @pytest.mark.parametrize(
        ("decorators", "message"),
        [
            ("@dsl.component\n@dsl.component", "exactly one decorator"),
            ("@dsl.container_component", "must be a @dsl.component or a @dsl.pipeline"),
            ("@staticmethod", "must be a @dsl.component or a @dsl.pipeline"),
            ("@dsl.component(output_component_file='/tmp/x.yaml')", "'output_component_file'"),
            ("@dsl.component(target_image='x')", "'target_image' is not supported"),
            ("@dsl.component(additional_funcs=[print])", "'additional_funcs'"),
            ("@dsl.component(embedded_artifact_path='/etc')", "'embedded_artifact_path'"),
            ("@dsl.component(base_image=open('x').read())", "expected a literal value"),
            ("@dsl.component('python:3.11')", "as keywords"),
        ],
    )
    def test_component_decorators(self, decorators, message):
        code = (
            HEADER
            + f"{decorators}\ndef extra(x: int):\n    pass\n\n@dsl.pipeline\ndef p():\n    pass\n"
        )
        assert message in _error(code)

    @pytest.mark.parametrize(
        ("signature", "message"),
        [
            ("x: 'int'", "type annotation"),
            ("x: open('x')", "type annotation"),
            ("x: Output[open('x')]", "type annotation"),
            ("x: int = len('abc')", "expected a literal value"),
            ("*args", "only take named parameters"),
            ("**kwargs", "only take named parameters"),
            ("x: int, /", "only take named parameters"),
        ],
    )
    def test_component_signatures(self, signature, message):
        code = (
            HEADER
            + f"@dsl.component\ndef extra({signature}):\n    pass\n\n@dsl.pipeline\ndef p():\n    pass\n"
        )
        assert message in _error(code)

    def test_return_annotation_is_checked(self):
        code = HEADER + "@dsl.component\ndef extra() -> __import__('os'):\n    pass\n"
        assert "type annotation" in _error(code + "\n@dsl.pipeline\ndef p():\n    pass\n")

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (HEADER, "found 0"),
            (
                HEADER + "@dsl.pipeline\ndef a():\n    pass\n\n@dsl.pipeline\ndef b():\n    pass\n",
                "found 2",
            ),
            (HEADER + "@dsl.pipeline(func=print)\ndef a():\n    pass\n", "'func' is not supported"),
        ],
    )
    def test_pipeline_count_and_options(self, code, message):
        assert message in _error(code)


class TestRejectedPipelineBody:
    """The pipeline body runs on the server at compile time: only graph wiring passes."""

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("open('/etc/passwd')", "calling 'open' is not allowed"),
            ("print(n)", "calling 'print' is not allowed"),
            ("exec('1')", "calling 'exec' is not allowed"),
            ("__import__('os').system('id')", "is not allowed"),
            ("getattr(n, 'x')", "calling 'getattr' is not allowed"),
            ("step.python_func(1, None)", "calling 'step.python_func' is not allowed"),
            ("step.execute(x=1)", "calling 'step.execute' is not allowed"),
            ("step.after(n)", "call component 'step' instead of referencing it"),
            ("t = step(x=n)\nt.component_spec", "Expr is not allowed"),
            ("t = step(x=n)\nx = t.component_spec", "attribute 'component_spec' is not allowed"),
            ("t = step(x=n)\nx = t._task_spec", "private attribute"),
            ("t = step(x=n)\nx = t.__class__", "private attribute"),
            ("x = step", "call component 'step' instead"),
            ("x = dsl.component(step)", "calling 'dsl.component' is not allowed"),
            ("x = dsl.os", "attribute 'os' is not allowed"),
            ("x = kubernetes.common", "attribute 'common' is not allowed"),
            ("for i in range(3):\n    step(x=i)", "For is not allowed"),
            ("while True:\n    pass", "While is not allowed"),
            ("if n:\n    step(x=n)", "If is not allowed"),
            ("import os", "Import is not allowed"),
            ("def f():\n    pass", "FunctionDef is not allowed"),
            ("x = [step(x=i) for i in [1, 2]]", "ListComp is not allowed"),
            ("x = lambda: 1", "Lambda is not allowed"),
            ("x = 'a' * 100000", "BinOp is not allowed"),
            ("x = 2 ** 100", "BinOp is not allowed"),
            ("x = f'{n:>100000}'", "format specs"),
            ("step(*[1])", "*args are not allowed"),
            ("step(**{'x': 1})", "**kwargs are not allowed"),
            ("x = {**{}}", "'**' in a dict"),
            ("x = (y := 1)", "NamedExpr is not allowed"),
            ("a, b = 1, 2", "one plain name"),
            ("x = undefined_name", "unknown name 'undefined_name'"),
            ("with open('x') as f:\n    pass", "only 'with dsl.If"),
            ("with dsl.ParallelFor(items=[1]) as (a, b):\n    pass", "one plain name"),
            ("dsl = 1", "shadows an import"),
            ("t = step(x=n)\nt.set_caching_options(open('x'))", "calling 'open' is not allowed"),
            ("global n", "Global is not allowed"),
            ("try:\n    pass\nexcept Exception:\n    pass", "Try is not allowed"),
            ("raise ValueError()", "Raise is not allowed"),
            ("x = 'a'\nx = x + x", "can only hold a task, a task output or a literal"),
            ("s = f'{n}-{n}'", "can only hold a task, a task output or a literal"),
            ("y = n", "can only hold a task, a task output or a literal"),
            ("y = -n", "can only hold a task, a task output or a literal"),
        ],
    )
    def test_rejected(self, body, message):
        assert message in _error(_pipeline(body))

    def test_doubling_a_value_is_rejected_before_compiling(self):
        # Review repro: ~500 bytes of `x = x + x` compiled to a 4 MB spec.
        body = "x = 'aaaa'\n" + "x = x + x\n" * 20 + "step(x=x)"
        assert "can only hold a task" in _error(_pipeline(body))

    @pytest.mark.parametrize(
        "body",
        [
            "items = [1, 2, 3]\nwith dsl.ParallelFor(items=items) as i:\n    step(x=i)",
            "data = step(x=n).outputs['data']\nuse(data=data)",
            "t = step(x=n)\nout = t.output\nuse(data=out)",
            "label = 'run'\nt = step(x=n)\nt.set_display_name(label + '-a')",
        ],
    )
    def test_tasks_outputs_and_literals_can_be_assigned(self, body):
        summary, err = check_pipeline_code(_pipeline(body))
        assert err is None, err and err.details

    def test_errors_name_the_line(self):
        err = _error(_pipeline("open('x')"))
        line = HEADER.count("\n") + 4
        assert f"line {line}:" in err

    def test_several_problems_are_listed(self):
        code = "import os\nimport sys\n" + PIPELINE_CODE
        _, err = check_pipeline_code(code)
        assert len(err.details["problems"]) == 2
        assert err.hint.startswith("Read pipelines://guides/authoring")


class TestInputLimits:
    @pytest.mark.parametrize("code", [None, "", "   ", 42])
    def test_empty_or_wrong_type(self, code):
        _, err = check_pipeline_code(code)
        assert err.error == "pipeline_code is required"

    def test_too_large(self):
        _, err = check_pipeline_code("#" * (_compile.MAX_PIPELINE_CODE_CHARS + 1))
        assert "too large" in err.error

    def test_syntax_error_names_the_line(self):
        _, err = check_pipeline_code("from kfp import dsl\n\ndef broken(:\n")
        assert "syntax error at line 3" in err.error

    def test_null_byte_is_rejected(self):
        _, err = check_pipeline_code("x = 1\x00")
        assert err.error_code == "VALIDATION_ERROR"

    def test_deep_nesting_is_rejected_without_crashing(self):
        _, err = check_pipeline_code("x = " + "[" * 200 + "]" * 200 + "\n")
        assert err is not None


def _fake_run(returncode=0, stderr="", calls=None):
    def run(args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, returncode, stdout="", stderr=stderr)

    return run


class TestCompileProcess:
    def test_runs_an_isolated_interpreter_with_a_stripped_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KUBECONFIG", "/secret/kubeconfig")
        monkeypatch.setenv("KUBEFLOW_MCP_PIPELINES_TOKEN", "secret-token")
        calls = []
        monkeypatch.setattr(_compile.subprocess, "run", _fake_run(returncode=2, calls=calls))

        compile_pipeline_code(PIPELINE_CODE, str(tmp_path))

        [(args, kwargs)] = calls
        assert args[:2] == [sys.executable, "-I"]
        assert args[2].endswith("_compile_worker.py")
        assert "KUBECONFIG" not in kwargs["env"]
        assert "KUBEFLOW_MCP_PIPELINES_TOKEN" not in kwargs["env"]
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == _compile.COMPILE_TIMEOUT_SECONDS
        assert kwargs["cwd"] == str(tmp_path)
        assert (tmp_path / "pipeline_source.py").read_text(encoding="utf-8") == PIPELINE_CODE
        assert kwargs["env"][_compile.COMPILE_MEMORY_ENV] == str(_compile.MAX_COMPILE_MEMORY_BYTES)
        assert kwargs["env"][_compile.COMPILE_OUTPUT_ENV] == str(_compile.MAX_SPEC_BYTES)

    def test_worker_and_server_agree_on_limit_names(self):
        assert _compile_worker.MEMORY_ENV == _compile.COMPILE_MEMORY_ENV
        assert _compile_worker.OUTPUT_ENV == _compile.COMPILE_OUTPUT_ENV

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows has no resource module")
    def test_limits_are_skipped_without_resource_module(self, monkeypatch):
        monkeypatch.setenv(_compile.COMPILE_MEMORY_ENV, "1000")
        assert _compile_worker.apply_limits() == []

    def test_timeout_is_reported(self, tmp_path, monkeypatch):
        def slow(args, **kwargs):
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        monkeypatch.setattr(_compile.subprocess, "run", slow)
        _, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path), timeout=5)
        assert err.error_code == "VALIDATION_ERROR"
        assert "longer than 5s" in err.error

    def test_missing_kfp_explains_how_to_install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _compile.subprocess, "run", _fake_run(3, "kfp is not installed: No module named 'kfp'")
        )
        _, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))
        assert err.error_code == "SDK_ERROR"
        assert "pipelines" in err.hint

    def test_compile_error_is_reported_and_bounded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _compile.subprocess, "run", _fake_run(2, "line 7: TypeError: bad\n" + "x" * 5000)
        )
        _, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))
        assert err.error_code == "VALIDATION_ERROR"
        assert len(err.error) < _compile.MAX_COMPILE_ERROR_CHARS + 100

    def test_interpreter_that_cannot_start(self, tmp_path, monkeypatch):
        def broken(args, **kwargs):
            raise OSError("no such file")

        monkeypatch.setattr(_compile.subprocess, "run", broken)
        _, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))
        assert err.error_code == "SDK_ERROR"

    def test_oversized_output_is_rejected(self, tmp_path, monkeypatch):
        (tmp_path / "pipeline.yaml").write_text("a: 1\n" * 100, encoding="utf-8")
        monkeypatch.setattr(_compile.subprocess, "run", _fake_run(0))
        monkeypatch.setattr(_compile, "MAX_SPEC_BYTES", 10)
        _, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))
        assert "too large" in err.error


needs_kfp = pytest.mark.skipif(
    importlib.util.find_spec("kfp") is None, reason="needs the pipelines extra (kfp)"
)


@needs_kfp
class TestRealCompile:
    """Compile with the installed kfp, in the child interpreter, as create_pipeline does."""

    def test_sample_compiles_with_the_usual_header(self, tmp_path):
        compiled, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))

        assert err is None, err
        text = Path(compiled.path).read_text(encoding="utf-8")
        assert text.startswith("# PIPELINE DEFINITION\n# Name: churn-prediction\n")
        assert compiled.documents[0]["pipelineInfo"]["name"] == "churn-prediction"

    def test_platform_document_is_kept(self, tmp_path):
        code = _pipeline(
            "t = step(x=n)\nkubernetes.use_secret_as_env(t, secret_name='s', secret_key_to_env={'k': 'K'})"
        )
        compiled, err = compile_pipeline_code(code, str(tmp_path))

        assert err is None, err
        assert len(compiled.documents) == 2
        assert "platforms" in compiled.documents[1]

    def test_importer_pipeline_compiles_and_passes_inspection(self, tmp_path):
        # Review repro: an importer step has no container, which the image check rejected.
        code = _pipeline(
            "i = dsl.importer(artifact_uri='gs://bucket/data.csv', artifact_class=dsl.Dataset)\n"
            "use(data=i.output)"
        )
        compiled, err = compile_pipeline_code(code, str(tmp_path))
        assert err is None, err

        summary, err = inspect_pipeline_spec(compiled.documents)

        assert err is None, err
        importer = next(s for s in summary["steps"] if s["kind"] == "importer")
        assert importer["image"] is None
        assert "gs://bucket/data.csv" in str(importer["artifact_uri"])
        assert summary["images"] == ["python:3.11-slim"]

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="OS limits need POSIX (resource module)")
    def test_oversized_output_is_stopped_by_the_file_size_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_compile, "MAX_SPEC_BYTES", 1024)

        started = time.monotonic()
        compiled, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))

        assert compiled is None
        assert err.error_code == "VALIDATION_ERROR"
        assert "File too large" in err.error
        assert time.monotonic() - started < _compile.COMPILE_TIMEOUT_SECONDS
        assert not (tmp_path / "pipeline.yaml").exists() or (
            (tmp_path / "pipeline.yaml").stat().st_size <= 1024
        )

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="OS limits need POSIX (resource module)")
    def test_memory_limit_stops_the_worker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_compile, "MAX_COMPILE_MEMORY_BYTES", 64 * 1024 * 1024)

        started = time.monotonic()
        compiled, err = compile_pipeline_code(PIPELINE_CODE, str(tmp_path))

        assert compiled is None
        assert err is not None
        assert time.monotonic() - started < _compile.COMPILE_TIMEOUT_SECONDS

    def test_kfp_errors_come_back_with_the_line(self, tmp_path):
        code = PIPELINE_CODE.replace("load(rows=rows)", "load(rowz=rows)")
        _, err = compile_pipeline_code(code, str(tmp_path))

        assert err.error_code == "VALIDATION_ERROR"
        assert "rowz" in err.error
        assert "line " in err.error


@needs_kfp
class TestWorkerInProcess:
    def test_compiles(self, tmp_path):
        source = tmp_path / "p.py"
        source.write_text(PIPELINE_CODE, encoding="utf-8")
        output = tmp_path / "p.yaml"

        assert _compile_worker.main(["w", str(source), str(output)]) == 0
        assert yaml.safe_load(output.read_text(encoding="utf-8").split("---")[0])["pipelineInfo"]

    def test_reports_failures(self, tmp_path, capsys):
        source = tmp_path / "p.py"
        source.write_text("from kfp import dsl\nx = 1 / 0\n", encoding="utf-8")

        assert _compile_worker.main(["w", str(source), str(tmp_path / "o.yaml")]) == 2
        assert "line 2: ZeroDivisionError" in capsys.readouterr().err

    def test_needs_exactly_one_pipeline(self, tmp_path, capsys):
        source = tmp_path / "p.py"
        source.write_text("from kfp import dsl\n", encoding="utf-8")

        assert _compile_worker.main(["w", str(source), str(tmp_path / "o.yaml")]) == 2
        assert "found 0" in capsys.readouterr().err

    def test_usage(self, capsys):
        assert _compile_worker.main(["w"]) == 2


def _spec(executors=None, parameters=None):
    return [
        {
            "pipelineInfo": {"name": "p"},
            "root": {"inputDefinitions": {"parameters": parameters or {}}},
            "deploymentSpec": {
                "executors": executors
                if executors is not None
                else {"exec-a": {"container": {"image": "python:3.11"}}}
            },
        }
    ]


class TestInspectSpec:
    def test_summary(self):
        summary, err = inspect_pipeline_spec(
            _spec(
                executors={
                    "exec-train": {
                        "container": {
                            "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
                            "resources": {"accelerator": {"resourceCount": "2"}},
                        }
                    }
                },
                parameters={
                    "epochs": {
                        "parameterType": "NUMBER_INTEGER",
                        "defaultValue": 1.0,
                        "isOptional": True,
                    }
                },
            )
        )
        assert err is None
        assert summary["steps"] == [
            {
                "executor": "exec-train",
                "kind": "container",
                "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
                "accelerators": 2,
            }
        ]
        assert summary["parameters"]["epochs"] == {
            "type": "NUMBER_INTEGER",
            "default": 1.0,
            "optional": True,
        }
        assert summary["has_platform_config"] is False

    @pytest.mark.parametrize("documents", [[], [None], [{"pipelineInfo": {}}], ["text"]])
    def test_not_a_spec(self, documents):
        _, err = inspect_pipeline_spec(documents)
        assert "did not produce a KFP pipeline spec" in err.error

    @pytest.mark.parametrize(
        ("container", "message"),
        [
            ({}, "has no container image"),
            ({"image": "Bad Image"}, "image of step 'exec-a'"),
            (
                {"image": "python:3.11", "resources": {"accelerator": {"count": "17"}}},
                "17 accelerators",
            ),
            (
                {"image": "python:3.11", "resources": {"accelerator": {"count": "lots"}}},
                "-1 accelerators",
            ),
        ],
    )
    def test_bad_steps(self, container, message):
        _, err = inspect_pipeline_spec(_spec(executors={"exec-a": {"container": container}}))
        assert message in err.error

    def test_importer_steps_need_no_image(self):
        summary, err = inspect_pipeline_spec(
            _spec(
                executors={
                    "exec-importer": {"importer": {"artifactUri": {"constant": "gs://b/d.csv"}}},
                    "exec-use": {"container": {"image": "python:3.11"}},
                }
            )
        )
        assert err is None
        assert summary["steps"][0] == {
            "executor": "exec-importer",
            "kind": "importer",
            "image": None,
            "accelerators": 0,
            "artifact_uri": {"constant": "gs://b/d.csv"},
        }
        assert summary["images"] == ["python:3.11"]

    def test_too_many_steps(self):
        executors = {
            f"exec-{i}": {"container": {"image": "python:3.11"}}
            for i in range(_compile.MAX_EXECUTORS + 1)
        }
        _, err = inspect_pipeline_spec(_spec(executors=executors))
        assert "too many steps" in err.error
