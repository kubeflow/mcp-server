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

"""Compile one KFP pipeline source file to pipeline YAML, in its own process.

Run as ``python -I _compile_worker.py SOURCE OUTPUT``. ``create_pipeline`` starts it
with an isolated interpreter, a stripped environment and a timeout, so pipeline code
never runs inside the MCP server process: KFP's compiler keeps global state, an
in-process compile cannot be timed out, and anything printed would land on the MCP
stdio channel. This file must stay standalone (it is not imported with ``-I``).

Exit codes: 0 compiled, 2 the pipeline code failed, 3 ``kfp`` is not installed.
Errors go to stderr, one line, prefixed with the source line when there is one.
"""

import importlib.util
import os
import sys
import traceback

EXIT_OK = 0
EXIT_PIPELINE_ERROR = 2
EXIT_KFP_MISSING = 3
_MODULE_NAME = "kubeflow_mcp_user_pipeline"


def _user_line(exc: BaseException, source: str) -> int | None:
    """Return the last line of the user's file in the traceback, if any."""
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if os.path.normcase(os.path.abspath(frame.filename)) == source:
            line = frame.lineno
    return line


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: _compile_worker.py SOURCE OUTPUT", file=sys.stderr)
        return EXIT_PIPELINE_ERROR
    source = os.path.normcase(os.path.abspath(argv[1]))
    output = argv[2]

    try:
        from kfp import compiler
        from kfp.dsl import graph_component
    except ImportError as e:
        print(f"kfp is not installed: {e}", file=sys.stderr)
        return EXIT_KFP_MISSING

    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipelines = [
            value
            for value in vars(module).values()
            if isinstance(value, graph_component.GraphComponent)
        ]
        if len(pipelines) != 1:
            print(
                f"expected exactly one @dsl.pipeline function, found {len(pipelines)}",
                file=sys.stderr,
            )
            return EXIT_PIPELINE_ERROR
        compiler.Compiler().compile(pipeline_func=pipelines[0], package_path=output)
    except Exception as e:  # noqa: BLE001 - every failure is reported to the caller
        line = _user_line(e, source)
        prefix = f"line {line}: " if line else ""
        message = " ".join(str(e).split()) or type(e).__name__
        print(f"{prefix}{type(e).__name__}: {message}", file=sys.stderr)
        return EXIT_PIPELINE_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv))
