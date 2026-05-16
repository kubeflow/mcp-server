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

"""Tests for _make_train_func: user-defined train() detection and wrapping."""

import pytest

from kubeflow_mcp.trainer.api.training import _make_train_func


class TestMakeTrainFuncWithUserDefinedTrain:
    """When the user already defines ``def train()`` at module level,
    _make_train_func must use that function directly instead of wrapping
    it inside another ``def train(): ...`` (which would silently skip
    execution).
    """

    def test_user_train_is_called(self):
        """A user-defined train() at module level is returned and callable."""
        script = """
def train():
    return "user code ran"
"""
        func = _make_train_func(script)
        assert func() == "user code ran"

    def test_user_train_with_print_output(self):
        """The function executes and produces expected side effects."""
        script = """
results = []
def train():
    for i in range(3):
        results.append(i)
    return results
"""
        func = _make_train_func(script)
        assert func() == [0, 1, 2]

    def test_user_train_imports_at_module_level(self):
        """Module-level imports work inside a user-defined train script."""
        script = """
import math as maths
def train():
    return maths.pi
"""
        func = _make_train_func(script)
        assert func() == pytest.approx(3.14159, rel=1e-4)

    def test_user_train_with_non_train_code(self):
        """Module-level code outside train() also executes on import."""
        script = """
flag = "set"
def train():
    return flag
"""
        func = _make_train_func(script)
        assert func() == "set"

    def test_nested_train_not_detected(self):
        """A ``def train(self):`` inside a class is not a top-level function
        and must still be wrapped."""
        script = """
class MyClass:
    def train(self):
        return 42
"""
        func = _make_train_func(script)
        # The user's train() is a method, not top-level.  Wrapping should
        # produce a callable train() that instantiates the class.
        func()
        # If we got here without error, the wrapper worked fine.
        # The wrapped function just defines the class but doesn't call the method,
        # so we don't assert a specific return value.

    def test_user_train_with_func_args_match(self):
        """When func_args matches the user's train() signature, the function
        is returned directly and can be called with those kwargs."""
        script = """
def train(model_dir, batch_size=32):
    return f"model={model_dir} batch={batch_size}"
"""
        func = _make_train_func(script, func_args={"model_dir": None, "batch_size": None})
        result = func(model_dir="/models", batch_size=64)
        assert result == "model=/models batch=64"

    def test_user_train_with_func_args_default_values(self):
        """func_args defaults (None) are passed through; the SDK calls
        func(**func_args) with the actual values."""
        script = """
def train(data_dir):
    return f"data={data_dir}"
"""
        func = _make_train_func(script, func_args={"data_dir": None})
        # The SDK will replace None with the actual value at call time.
        assert func(data_dir="/datasets/imagenet") == "data=/datasets/imagenet"

    def test_user_train_missing_func_args_raises(self):
        """If func_args specifies parameters not in the user's train()
        signature, raise ValueError with a clear message."""
        script = """
def train(model_dir):
    return "done"
"""
        with pytest.raises(ValueError, match="func_args requires parameters"):
            _make_train_func(script, func_args={"unknown_param": None})

        with pytest.raises(ValueError, match="unknown_param"):
            _make_train_func(script, func_args={"unknown_param": None})

    def test_user_train_syntax_error_raises(self):
        """If the script has a syntax error, _make_train_func raises
        SyntaxError (both direct and wrapping paths will fail to compile)."""
        script = "def train(:  # broken syntax"
        with pytest.raises(SyntaxError):
            _make_train_func(script)

    def test_original_behavior_preserved_for_script_without_def_train(self):
        """Scripts without ``def train()`` still get wrapped as before."""
        script = "x = 1 + 1"
        func = _make_train_func(script)
        func()  # Should not raise
        # x=2 is set in the wrapper's namespace but not returned.
        # Just verifying no exception.

    def test_original_behavior_with_func_args(self):
        """Scripts without def train + func_args still get the wrapper
        with the requested parameter signature."""
        script = "print(f'training with {data_dir}')"
        func = _make_train_func(script, func_args={"data_dir": None})
        # Should accept the keyword argument.
        func(data_dir="/data")
        # No exception = success
