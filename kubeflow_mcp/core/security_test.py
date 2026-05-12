"""Tests for security validation."""

from kubeflow_mcp.core.security import (
    is_safe_python_code,
    mask_sensitive_data,
    validate_k8s_name,
    validate_resource_limits,
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
    data = {"user": "admin", "password": "secret123", "token": "abc"}
    masked = mask_sensitive_data(data)
    assert masked["user"] == "admin"
    assert masked["password"] == "***"
    assert masked["token"] == "***"


def test_mask_sensitive_data_nested():
    data = {"config": {"api_key": "secret", "url": "http://example.com"}}
    masked = mask_sensitive_data(data)
    assert masked["config"]["api_key"] == "***"
    assert masked["config"]["url"] == "http://example.com"
