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

"""Bounded pod-log transport tests (no cluster or eager SDK log iterator)."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.response import HTTPResponse

from kubeflow_mcp.trainer.api import monitoring


class CountingBody(BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.bytes_read = 0
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        assert 0 < size <= monitoring.LOG_READ_CHUNK_BYTES, "Unbounded HTTP read"
        self.requests.append(size)
        data = super().read(size)
        self.bytes_read += len(data)
        return data


def response_for(data: bytes) -> tuple[HTTPResponse, CountingBody, MagicMock]:
    body = CountingBody(data)
    connection = MagicMock()
    pool = MagicMock()
    response = HTTPResponse(body=body, preload_content=False, connection=connection, pool=pool)
    return response, body, pool


@pytest.mark.parametrize("count", [9999, 10000, 10001, 1000000])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_line_cap_reads_bounded_prefix_and_closes(count, trailing_newline):
    data = b"x\n" * count
    if not trailing_newline:
        data = data[:-1]
    response, body, pool = response_for(data)
    connection = response.connection
    lines, truncated = monitoring._read_bounded_log_lines(response)
    assert lines == ["x"] * min(count, monitoring.MAX_LOG_BUFFER_LINES)
    assert truncated is (count > monitoring.MAX_LOG_BUFFER_LINES)
    assert body.bytes_read <= 2 * monitoring.MAX_LOG_BUFFER_LINES + monitoring.LOG_READ_CHUNK_BYTES
    assert body.closed
    connection.close.assert_called_once()
    pool._put_conn.assert_called_once_with(connection)


@pytest.mark.parametrize("data", [b"", b"a", b"a\n", b"\n", b"a\r\nb\rc\n\n"])
@pytest.mark.parametrize("chunk_size", [1, 2, 7])
def test_chunk_boundaries_match_splitlines(data, chunk_size, monkeypatch):
    monkeypatch.setattr(monitoring, "LOG_READ_CHUNK_BYTES", chunk_size)
    response, body, pool = response_for(data)
    lines, truncated = monitoring._read_bounded_log_lines(response)
    assert lines == data.decode().splitlines()
    assert not truncated
    assert body.closed
    pool._put_conn.assert_called_once()


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7])
def test_utf8_and_unicode_line_endings(chunk_size, monkeypatch):
    monkeypatch.setattr(monitoring, "LOG_READ_CHUNK_BYTES", chunk_size)
    text = "학습 🚀\r\n다음\u2028끝\x85last"
    response, _, _ = response_for(text.encode())
    assert monitoring._read_bounded_log_lines(response) == (text.splitlines(), False)


@pytest.mark.parametrize(
    "data, expected, truncated",
    [
        (b"one\ntwo\n", ["one", "two"], False),
        (b"one\ntwo\nx", ["one", "two"], True),
        (b"one\nlong-line-without-a-newline", ["one"], True),
        (b"x" * 100000, [], True),
        ("one\n🚀🚀".encode(), ["one"], True),
    ],
)
def test_byte_cap_does_not_advance_over_partial_lines(data, expected, truncated, monkeypatch):
    monkeypatch.setattr(monitoring, "MAX_LOG_BUFFER_BYTES", 8)
    response, body, pool = response_for(data)
    assert monitoring._read_bounded_log_lines(response) == (expected, truncated)
    assert body.bytes_read <= 9
    assert body.closed
    pool._put_conn.assert_called_once()


@pytest.mark.parametrize("error", [OSError("connection reset"), KeyboardInterrupt()])
def test_read_failure_closes_and_releases(error):
    response, body, pool = response_for(b"logs")
    connection = response.connection
    with patch.object(response, "read", side_effect=error):
        with pytest.raises(type(error)):
            monitoring._read_bounded_log_lines(response)
    assert body.closed
    connection.close.assert_called_once()
    pool._put_conn.assert_called_once_with(connection)


def test_release_is_attempted_even_if_close_raises():
    response, _, pool = response_for(b"")
    with patch.object(response, "close", side_effect=OSError("close failed")):
        with pytest.raises(OSError, match="close failed"):
            monitoring._read_bounded_log_lines(response)
    pool._put_conn.assert_called_once()


@pytest.mark.parametrize(
    "step, container",
    [("node-1", "node"), ("launcher-0", "launcher"), ("model-initializer", "model-initializer")],
)
def test_cursor_uses_non_preloaded_pod_response_not_sdk_logs(step, container):
    client = MagicMock()
    client.get_job_logs.side_effect = AssertionError("must not eagerly load SDK logs")
    client.get_job.return_value = SimpleNamespace(
        steps=[
            SimpleNamespace(name="other-step", pod_name="other-pod", status="Running"),
            SimpleNamespace(name=step, pod_name="target-pod", status="Running"),
        ]
    )
    response, body, pool = response_for(b"first\nsecond\n")
    with (
        patch.object(monitoring, "get_trainer_client_for_namespace", return_value=client),
        patch.object(monitoring, "get_core_v1_api") as core,
        patch.object(monitoring, "check_namespace_allowed", return_value=None),
    ):
        core.return_value.read_namespaced_pod_log.return_value = response
        result = monitoring.get_training_logs("job", step=step, namespace="training", since_line=0)
    assert result["success"]
    assert result["data"]["logs"] == "first\nsecond"
    client.get_job.assert_called_once_with(name="job")
    client.get_job_logs.assert_not_called()
    core.return_value.read_namespaced_pod_log.assert_called_once_with(
        name="target-pod",
        namespace="training",
        container=container,
        follow=False,
        _preload_content=False,
        _request_timeout=monitoring.K8S_TIMEOUT,
    )
    assert body.closed
    pool._put_conn.assert_called_once()


@pytest.mark.parametrize("status", ["Pending", "Running"])
def test_pending_or_missing_step_does_not_read_logs(status):
    client = MagicMock()
    client.get_job.return_value = SimpleNamespace(
        steps=[
            SimpleNamespace(name="node-0", pod_name="pod", status=status),
        ]
    )
    with patch.object(monitoring, "get_core_v1_api") as core:
        step = "node-0" if status == "Pending" else "node-1"
        assert monitoring._collect_cursor_log_lines(client, "job", step, "ns") == ([], False)
    core.assert_not_called()


@pytest.mark.parametrize(
    "error, code",
    [
        (ApiException(status=404), "RESOURCE_NOT_FOUND"),
        (OSError("read timeout"), "SDK_ERROR"),
    ],
)
def test_cursor_transport_errors_use_existing_tool_errors(error, code):
    client = MagicMock()
    client.get_job.return_value = SimpleNamespace(
        steps=[
            SimpleNamespace(name="node-0", pod_name="pod", status="Running"),
        ]
    )
    with (
        patch.object(monitoring, "get_trainer_client_for_namespace", return_value=client),
        patch.object(monitoring, "get_core_v1_api") as core,
        patch.object(monitoring, "check_namespace_allowed", return_value=None),
    ):
        core.return_value.read_namespaced_pod_log.side_effect = error
        result = monitoring.get_training_logs("job", namespace="ns", since_line=0)
    assert not result["success"]
    assert result["error_code"] == code


def test_line_cap_counts_incomplete_utf8_as_more_data(monkeypatch):
    monkeypatch.setattr(monitoring, "MAX_LOG_BUFFER_LINES", 1)
    response, _, _ = response_for(b"first\n\xe2")
    assert monitoring._read_bounded_log_lines(response) == (["first"], True)


def test_previous_logs_are_bounded_and_cannot_be_used_as_cursor(monkeypatch):
    monkeypatch.setattr(monitoring, "MAX_LOG_BUFFER_BYTES", 8)
    client = MagicMock()
    client.get_job.side_effect = [
        SimpleNamespace(steps=[]),
        SimpleNamespace(steps=[SimpleNamespace(name="node-0", pod_name="pod")]),
    ]
    response, body, pool = response_for(b"x" * 100000)
    with (
        patch.object(monitoring, "get_trainer_client_for_namespace", return_value=client),
        patch.object(monitoring, "get_core_v1_api") as core,
        patch.object(monitoring, "check_namespace_allowed", return_value=None),
    ):
        core.return_value.read_namespaced_pod_log.return_value = response
        result = monitoring.get_training_logs("job", namespace="ns", since_line=0)
    assert not result["success"]
    assert result["error_code"] == "VALIDATION_ERROR"
    assert body.bytes_read == 9
    assert body.closed
    pool._put_conn.assert_called_once()
    core.return_value.read_namespaced_pod_log.assert_called_once_with(
        name="pod",
        namespace="ns",
        container="node",
        previous=True,
        tail_lines=1000,
        _preload_content=False,
        _request_timeout=monitoring.K8S_TIMEOUT,
    )
