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

"""Tests for server instruction assembly across client modules.

Regression guard: every loaded module's INSTRUCTION_SECTIONS must actually
surface to the agent — not just the trainer's. A hardcoded section-order list
previously dropped the optimizer's namespaced sections.
"""

import importlib

from kubeflow_mcp.core.server import _build_server_instructions, _sections_for_persona


def _modules():
    return {
        "trainer": importlib.import_module("kubeflow_mcp.trainer"),
        "optimizer": importlib.import_module("kubeflow_mcp.optimizer"),
    }


def test_optimizer_sections_surface_for_admin():
    sections = _sections_for_persona("platform-admin")
    assert "optimizer_planning" in sections
    assert "optimizer_optimization" in sections
    assert "optimizer_monitoring" in sections


def test_trainer_sections_still_ordered_first():
    sections = _sections_for_persona("platform-admin")
    # Known trainer sections keep their canonical position ahead of optimizer's.
    assert sections.index("planning") < sections.index("optimizer_planning")


def test_admin_instructions_include_optimizer_guidance():
    instr = _build_server_instructions(_modules(), "platform-admin", "full")
    assert "OPTIMIZER PLANNING" in instr
    assert "OPTIMIZER MONITORING" in instr
    assert "OPTIMIZER TOOL SELECTION" in instr
    # Trainer guidance is unaffected.
    assert "PLANNING" in instr


def test_readonly_excludes_optimization_section():
    """readonly persona has no create_* tools, so the optimization-phase
    instruction section must not be injected."""
    instr = _build_server_instructions(_modules(), "readonly", "full")
    assert "OPTIMIZER TOOL SELECTION" not in instr
    # But read-only monitoring/planning guidance should still appear.
    assert "OPTIMIZER MONITORING" in instr


def _optimizer_tools_named(instructions):
    from kubeflow_mcp.optimizer import TOOLS

    return {t.__name__ for t in TOOLS if f"{t.__name__}(" in instructions}


def test_readonly_is_not_told_to_call_optimizer_tools_it_lacks():
    from kubeflow_mcp.core.policy import get_allowed_tools

    instr = _build_server_instructions(_modules(), "readonly", "full")
    assert "OPTIMIZER LIFECYCLE" not in instr
    assert _optimizer_tools_named(instr) <= get_allowed_tools("readonly")


def test_lifecycle_guidance_reaches_personas_with_lifecycle_tools():
    """data-scientist gets the section for wait and delete. Full-tier sections
    are chosen per phase, not per tool (the same granularity as the trainer's
    guidance), so it also sees the two ml-engineer tools that share a phase
    with tools it has, and nothing else it lacks."""
    from kubeflow_mcp.core.policy import get_allowed_tools

    instr = _build_server_instructions(_modules(), "data-scientist", "full")
    assert "OPTIMIZER LIFECYCLE" in instr
    assert _optimizer_tools_named(instr) - get_allowed_tools("data-scientist") == {
        "update_hpo_experiment",
        "create_hpo_experiment_from_spec",
    }
    ml = _build_server_instructions(_modules(), "ml-engineer", "full")
    assert _optimizer_tools_named(ml) <= get_allowed_tools("ml-engineer")
