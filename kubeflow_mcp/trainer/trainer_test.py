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
"""Tests for trainer module exports and get_tools."""

from kubeflow_mcp.trainer import MODULE_INFO, TOOL_CATEGORIES, TOOLS, get_tools


class TestModuleInfo:
    def test_status_implemented(self):
        assert MODULE_INFO["status"] == "implemented"

    def test_has_required_keys(self):
        assert "name" in MODULE_INFO
        assert "description" in MODULE_INFO
        assert "sdk_client" in MODULE_INFO

    def test_name_is_trainer(self):
        assert MODULE_INFO["name"] == "trainer"


class TestToolsList:
    def test_tools_is_non_empty(self):
        assert len(TOOLS) > 0

    def test_all_tools_are_callable(self):
        for tool in TOOLS:
            assert callable(tool), f"{tool} is not callable"

    def test_all_tools_have_names(self):
        for tool in TOOLS:
            assert hasattr(tool, "__name__")

    def test_no_duplicate_tools(self):
        names = [t.__name__ for t in TOOLS]
        assert len(names) == len(set(names))

    def test_contains_expected_tools(self):
        names = {t.__name__ for t in TOOLS}
        expected = {
            "get_cluster_resources",
            "estimate_resources",
            "fine_tune",
            "run_custom_training",
            "run_container_training",
            "list_training_jobs",
            "get_training_job",
            "list_runtimes",
            "get_runtime",
            "get_runtime_packages",
            "get_training_logs",
            "get_training_events",
            "wait_for_training",
            "delete_training_job",
            "suspend_training_job",
            "resume_training_job",
        }
        assert expected.issubset(names)


class TestToolCategories:
    def test_has_expected_categories(self):
        expected = {"core", "planning", "training", "discovery", "monitoring", "lifecycle"}
        assert set(TOOL_CATEGORIES.keys()) == expected

    def test_all_category_tools_are_in_tools(self):
        tool_names = {t.__name__ for t in TOOLS}
        for cat, cat_tools in TOOL_CATEGORIES.items():
            for tool in cat_tools:
                assert tool.__name__ in tool_names, f"{tool.__name__} in '{cat}' not in TOOLS"

    def test_core_is_subset_of_all(self):
        core_names = {t.__name__ for t in TOOL_CATEGORIES["core"]}
        all_names = {t.__name__ for t in TOOLS}
        assert core_names.issubset(all_names)


class TestGetTools:
    def test_none_returns_all(self):
        result = get_tools(categories=None)
        assert result is TOOLS

    def test_single_category(self):
        result = get_tools(categories=["planning"])
        names = {t.__name__ for t in result}
        assert "get_cluster_resources" in names
        assert "estimate_resources" in names

    def test_multiple_categories(self):
        result = get_tools(categories=["planning", "monitoring"])
        names = {t.__name__ for t in result}
        assert "get_cluster_resources" in names
        assert "get_training_logs" in names

    def test_deduplication(self):
        result = get_tools(categories=["core", "planning"])
        names = [t.__name__ for t in result]
        assert names.count("get_cluster_resources") == 1

    def test_unknown_category_returns_empty(self):
        result = get_tools(categories=["nonexistent"])
        assert result == []

    def test_empty_categories_returns_empty(self):
        result = get_tools(categories=[])
        assert result == []
