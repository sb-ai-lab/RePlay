from __future__ import annotations

from urllib import parse

import pytest

from scripts.review_agent.gitlab_client import GitlabMergeRequestClient


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
