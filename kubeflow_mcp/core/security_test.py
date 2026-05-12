"""Tests for security validation."""

from unittest.mock import patch

from kubeflow_mcp.core.security import (
    check_namespace_allowed,
    is_safe_python_code,
    mask_sensitive_data,
    truncate_log_output,
    validate_k8s_name,
    validate_namespace,
    validate_resource_limits,
    validate_training_bounds,
)


def test_validate_k8s_name_valid():
    assert validate_k8s_name("my-training-job") is None
    assert validate_k8s_name("job123") is None
    assert validate_k8s_name("a") is None


def test_validate_k8s_name_empty():
    err = validate_k8s_name("")
    assert err is not None
    assert "empty" in err.error.lower()


def test_validate_k8s_name_invalid():
    err = validate_k8s_name("My-Job")
    assert err is not None
    assert "lowercase" in err.error.lower()

    err = validate_k8s_name("job_with_underscore")
    assert err is not None


def test_validate_k8s_name_too_long():
    err = validate_k8s_name("a" * 64)
    assert err is not None
    assert "too long" in err.error.lower()


def test_is_safe_python_code_safe():
    safe, _ = is_safe_python_code("x = 1 + 2")
    assert safe

    safe, _ = is_safe_python_code("import torch\nmodel = torch.nn.Linear(10, 5)")
    assert safe


def test_is_safe_python_code_dangerous():
    safe, reason = is_safe_python_code("import os\nos.system('rm -rf /')")
    assert not safe
    assert "os" in reason.lower()

    safe, reason = is_safe_python_code("eval('bad')")
    assert not safe


def test_validate_resource_limits_valid():
    assert validate_resource_limits("100m", "256Mi", 1) is None
    assert validate_resource_limits("2", "1Gi", 0) is None
    assert validate_resource_limits("0.5", "1Gi", 0) is None
    assert validate_resource_limits("1.5", None, None) is None
    assert validate_resource_limits(None, None, None) is None


def test_validate_resource_limits_invalid():
    err = validate_resource_limits("invalid", None, None)
    assert err is not None

    err = validate_resource_limits(None, "256MB", None)
    assert err is not None

    err = validate_resource_limits(None, None, -1)
    assert err is not None


def test_mask_sensitive_data():
    data = {"user": "admin", "password": "secret123", "access_token": "abc"}
    masked = mask_sensitive_data(data)
    assert masked["user"] == "admin"
    assert masked["password"] == "***"
    assert masked["access_token"] == "***"


def test_mask_sensitive_data_nested():
    data = {"config": {"api_key": "secret", "url": "http://example.com"}}
    masked = mask_sensitive_data(data)
    assert masked["config"]["api_key"] == "***"
    assert masked["config"]["url"] == "http://example.com"


# ─── validate_namespace ────────────────────────────────────────────────────────


def test_validate_namespace_valid():
    assert validate_namespace("default") is None
    assert validate_namespace("ml-team") is None


def test_validate_namespace_invalid():
    err = validate_namespace("MY NAMESPACE")
    assert err is not None
    assert err.error_code == "VALIDATION_ERROR"


# ─── check_namespace_allowed ──────────────────────────────────────────────────


def test_check_namespace_allowed_no_policy():
    """When no namespace policy is set, all namespaces are allowed."""
    with patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=None):
        assert check_namespace_allowed("any-namespace") is None
        assert check_namespace_allowed(None) is None


def test_check_namespace_allowed_explicit_allowed():
    with patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["ml", "prod"]):
        assert check_namespace_allowed("ml") is None
        assert check_namespace_allowed("prod") is None


def test_check_namespace_allowed_explicit_denied():
    with patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["ml"]):
        err = check_namespace_allowed("default")
        assert err is not None
        assert err.error_code == "PERMISSION_DENIED"
        assert "default" in err.error


def test_check_namespace_allowed_none_resolves_effective():
    """None namespace resolves via get_trainer_effective_namespace."""
    with (
        patch("kubeflow_mcp.core.policy.get_allowed_namespaces", return_value=["ml"]),
        patch(
            "kubeflow_mcp.common.utils.get_trainer_effective_namespace",
            return_value="default",
        ),
    ):
        err = check_namespace_allowed(None)
        assert err is not None
        assert "default" in err.error


# ─── validate_training_bounds ─────────────────────────────────────────────────


def test_validate_training_bounds_all_valid():
    assert validate_training_bounds(batch_size=4, epochs=3, num_nodes=2, gpu_per_node=1) is None


def test_validate_training_bounds_batch_size_too_large():
    err = validate_training_bounds(batch_size=99999)
    assert err is not None
    assert "batch_size" in err.error


def test_validate_training_bounds_epochs_zero():
    err = validate_training_bounds(epochs=0)
    assert err is not None
    assert "epochs" in err.error


def test_validate_training_bounds_lora_dropout_out_of_range():
    err = validate_training_bounds(lora_dropout=1.5)
    assert err is not None
    assert "lora_dropout" in err.error


def test_validate_training_bounds_empty_script():
    err = validate_training_bounds(script="   ")
    assert err is not None
    assert "Script" in err.error


def test_validate_training_bounds_none_values_ok():
    """All None inputs are valid (no constraint to check)."""
    assert validate_training_bounds() is None


# ─── truncate_log_output ──────────────────────────────────────────────────────


def test_truncate_log_output_short_string_unchanged():
    text = "hello world"
    assert truncate_log_output(text) == text


def test_truncate_log_output_truncates_long_string():
    long_text = "x" * 20000
    result = truncate_log_output(long_text, max_length=100)
    assert len(result) < len(long_text)
    assert "truncated" in result.lower() or len(result) <= 200


def test_truncate_log_output_custom_max():
    text = "a" * 500
    result = truncate_log_output(text, max_length=100)
    assert len(result) <= 200


# ─── mask_sensitive_data list recursion ───────────────────────────────────────


def test_mask_sensitive_data_list_of_dicts():
    data = {"configs": [{"hf_token": "tok1"}, {"url": "http://x"}]}
    masked = mask_sensitive_data(data)
    assert masked["configs"][0]["hf_token"] == "***"
    assert masked["configs"][1]["url"] == "http://x"


def test_mask_sensitive_data_safe_keys_not_masked():
    data = {"public_key": "pk-abc", "key_name": "my-key"}
    masked = mask_sensitive_data(data)
    assert masked["public_key"] == "pk-abc"
    assert masked["key_name"] == "my-key"
