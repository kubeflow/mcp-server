import pytest
from unittest.mock import patch, mock_open
from kubeflow_mcp.trainer.api.training import _make_train_func

def test_make_train_func_simple():
    script = "print('hello')"
    with patch("kubeflow_mcp.trainer.api.training.uuid") as mock_uuid, \
         patch("builtins.open", mock_open()) as mock_file:
        func = _make_train_func(script)
        assert callable(func)
        assert func.__name__ == "train"

def test_make_train_func_with_existing_train():
    script = "def train():\n    return 42\n"
    func = _make_train_func(script)
    assert func() == 42

def test_make_train_func_missing_args():
    script = "def train():\n    return 42\n"
    with pytest.raises(ValueError, match="is missing parameters required by func_args: param1"):
        _make_train_func(script, func_args={"param1": 1})

def test_make_train_func_matching_args():
    script = "def train(param1, param2=None):\n    return param1\n"
    func = _make_train_func(script, func_args={"param1": 1})
    assert func(42) == 42

def test_make_train_func_with_kwargs():
    script = "def train(**kwargs):\n    return kwargs.get('param1')\n"
    func = _make_train_func(script, func_args={"param1": 1})
    assert func(param1=42) == 42
