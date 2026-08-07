from __future__ import annotations

from email.message import Message
from urllib import error, parse

import pytest

from scripts.review_agent.gitlab_client import GitlabMergeRequestClient, request_text


class DummyResponse:
    def __init__(self, text: str) -> None:
        self._text = text.encode("utf-8")

    def read(self) -> bytes:
        return self._text

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def close(self) -> None:
        return None


class SequencedOpener:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.timeouts: list[int] = []

    def open(self, req: object, timeout: int = 180) -> DummyResponse:
        self.timeouts.append(timeout)
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return DummyResponse(str(event))


def _http_error(status: int, body: str, *, retry_after: str | None = None) -> error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return error.HTTPError(
        url="https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42/notes",
        code=status,
        msg="error",
        hdrs=headers,
        fp=DummyResponse(body),
    )


def test_merge_request_api_url_quotes_project_id_once() -> None:
    client = GitlabMergeRequestClient(
        api_base="https://gitlab.example/api/v4/",
        project_id="group/subgroup/project%2Fname",
        merge_request_id="42",
        base_sha="base",
        head_sha="head",
        token="token",
    )

    assert client.merge_request_api_url == (
        "https://gitlab.example/api/v4/projects/"
        "group%2Fsubgroup%2Fproject%2Fname/merge_requests/42"
    )


def test_post_inline_comment_posts_discussion_form_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_request_json(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None = None,
        data: bytes | None = None,
        opener: object | None = None,
    ) -> object:
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "payload": payload,
                "data": data,
                "opener": opener,
            }
        )
        if method == "GET":
            return {"diff_refs": {"start_sha": "start-from-api"}}
        return {}

    monkeypatch.setattr("scripts.review_agent.gitlab_client.request_json", fake_request_json)
    client = GitlabMergeRequestClient(
        api_base="https://gitlab.example/api/v4",
        project_id="group/project",
        merge_request_id="42",
        base_sha="base",
        head_sha="head",
        token="token",
    )

    publish_url = client.post_inline_comment(
        body="Inline body",
        relative_file_path="src/app.py",
        line=12,
    )

    assert publish_url == (
        "https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42/discussions"
    )
    assert calls[0] == {
        "method": "GET",
        "url": "https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42",
        "headers": {
            "Content-Type": "application/json",
            "PRIVATE-TOKEN": "token",
        },
        "payload": None,
        "data": None,
        "opener": None,
    }
    assert calls[1]["method"] == "POST"
    assert calls[1]["url"] == publish_url
    assert calls[1]["headers"] == {
        "Content-Type": "application/x-www-form-urlencoded",
        "PRIVATE-TOKEN": "token",
    }
    assert calls[1]["payload"] is None
    assert calls[1]["opener"] is None
    assert parse.parse_qs(calls[1]["data"].decode("utf-8")) == {
        "body": ["Inline body"],
        "position[position_type]": ["text"],
        "position[base_sha]": ["base"],
        "position[start_sha]": ["start-from-api"],
        "position[head_sha]": ["head"],
        "position[new_path]": ["src/app.py"],
        "position[new_line]": ["12"],
    }


def test_post_note_posts_body_field(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_request_json(
        method: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object] | None = None,
        data: bytes | None = None,
        opener: object | None = None,
    ) -> object:
        calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "payload": payload,
                "data": data,
                "opener": opener,
            }
        )
        return {}

    monkeypatch.setattr("scripts.review_agent.gitlab_client.request_json", fake_request_json)
    client = GitlabMergeRequestClient(
        api_base="https://gitlab.example/api/v4",
        project_id="group/project",
        merge_request_id="42",
        base_sha="base",
        head_sha="head",
        token="token",
    )

    publish_url = client.post_note("Top-level note")

    assert publish_url == "https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42/notes"
    assert calls == [
        {
            "method": "POST",
            "url": publish_url,
            "headers": {
                "Content-Type": "application/x-www-form-urlencoded",
                "PRIVATE-TOKEN": "token",
            },
            "payload": None,
            "data": b"body=Top-level+note",
            "opener": None,
        }
    ]


def test_request_text_retries_on_retryable_gitlab_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("scripts.review_agent.gitlab_client.time.sleep", sleep_calls.append)
    opener = SequencedOpener(
        [
            _http_error(502, "bad gateway"),
            "ok",
        ]
    )

    result = request_text(
        method="POST",
        url="https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42/notes",
        headers={"PRIVATE-TOKEN": "token"},
        opener=opener,
    )

    assert result == "ok"
    assert opener.timeouts == [30, 30]
    assert sleep_calls == [1.0]
