from __future__ import annotations

import json
from email.message import Message
from urllib import error, request

import pytest

from scripts.review_agent.anthropic_compatible_client import AnthropicCompatibleClient


class DummyResponse:
    def __init__(self, payload: object) -> None:
        if isinstance(payload, bytes):
            self._payload = payload
        elif isinstance(payload, str):
            self._payload = payload.encode("utf-8")
        else:
            self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def close(self) -> None:
        return None


class DummyOpener:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.requests: list[request.Request] = []
        self.timeouts: list[int] = []

    def open(self, req: request.Request, timeout: int = 180) -> DummyResponse:
        self.requests.append(req)
        self.timeouts.append(timeout)
        return DummyResponse(self._payload)


class SequencedOpener:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.requests: list[request.Request] = []
        self.timeouts: list[int] = []

    def open(self, req: request.Request, timeout: int = 180) -> DummyResponse:
        self.requests.append(req)
        self.timeouts.append(timeout)
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return DummyResponse(event)


def _http_error(status: int, body: str, *, retry_after: str | None = None) -> error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return error.HTTPError(
        url="https://gateway.example/anthropic/v1/messages",
        code=status,
        msg="error",
        hdrs=headers,
        fp=DummyResponse(body),
    )


def test_constructor_accepts_positional_or_keyword_parameters() -> None:
    positional_client = AnthropicCompatibleClient(
        "secret",
        "claude-sonnet",
        "https://gateway.example/anthropic/",
        "2023-06-01",
    )
    keyword_client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic/",
        api_version="2023-06-01",
    )

    assert positional_client.messages_url == "https://gateway.example/anthropic/v1/messages"
    assert keyword_client.messages_url == "https://gateway.example/anthropic/v1/messages"


def test_messages_request_uses_base_url_headers_and_model() -> None:
    opener = DummyOpener(
        {
            "content": [
                {
                    "type": "text",
                    "text": '{"comments": []}',
                }
            ]
        }
    )
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic/",
        api_version="2023-06-01",
        opener=opener,
    )

    result = client.create_review(prompt="Return JSON")

    assert result == '{"comments": []}'
    assert client.messages_url == "https://gateway.example/anthropic/v1/messages"
    assert opener.timeouts == [120]
    req = opener.requests[0]
    assert req.full_url == "https://gateway.example/anthropic/v1/messages"
    assert req.get_method() == "POST"
    assert req.headers["Content-type"] == "application/json"
    assert req.headers["X-api-key"] == "secret"
    assert req.headers["Anthropic-version"] == "2023-06-01"
    assert json.loads(req.data.decode("utf-8")) == {
        "model": "claude-sonnet",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Return JSON"}],
    }


def test_client_raises_on_missing_text_content() -> None:
    opener = DummyOpener({"content": [{"type": "image"}]})
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    with pytest.raises(
        RuntimeError,
        match="Anthropic-compatible response did not contain text content",
    ):
        client.create_review(prompt="Return JSON")


def test_client_joins_multiple_text_blocks_and_skips_empty_prefix() -> None:
    opener = DummyOpener(
        {
            "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": '{"comments": []}'},
            ]
        }
    )
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    result = client.create_review(prompt="Return JSON")

    assert result == '{"comments": []}'


def test_client_retries_on_rate_limit_and_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("scripts.review_agent.anthropic_compatible_client.time.sleep", sleep_calls.append)
    opener = SequencedOpener(
        [
            _http_error(429, '{"error":"rate limited"}', retry_after="3"),
            {"content": [{"type": "text", "text": '{"comments": []}'}]},
        ]
    )
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    result = client.create_review(prompt="Return JSON")

    assert result == '{"comments": []}'
    assert len(opener.requests) == 2
    assert opener.timeouts == [120, 120]
    assert sleep_calls == [3.0]


def test_client_does_not_retry_non_retryable_http_errors() -> None:
    opener = SequencedOpener([_http_error(401, '{"error":"unauthorized"}')])
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    with pytest.raises(RuntimeError, match="HTTP 401"):
        client.create_review(prompt="Return JSON")

    assert len(opener.requests) == 1


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ("not-json", "Anthropic-compatible API returned malformed JSON"),
        (b"\x80", "Anthropic-compatible API returned malformed JSON"),
        (["not", "an", "object"], "Anthropic-compatible API returned malformed JSON"),
        ({"content": "not-a-list"}, "Anthropic-compatible response did not contain text content"),
    ],
)
def test_client_raises_runtime_error_on_malformed_successful_responses(
    payload: object,
    expected_message: str,
) -> None:
    opener = DummyOpener(payload)
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        client.create_review(prompt="Return JSON")
