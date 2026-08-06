from __future__ import annotations

from pathlib import Path

import pytest

from scripts.review_agent.publish_runner import MergeRequestPublishService, ReviewCommentsPublisher
from scripts.review_agent.schema import CodeLocation, LineRange, ReviewComment, ReviewResult


class RecordingGitlabClient:
    def __init__(self, *, inline_error: Exception | None = None) -> None:
        self.inline_error = inline_error
        self.inline_calls: list[dict[str, object]] = []
        self.note_calls: list[str] = []

    def post_inline_comment(self, *, body: str, relative_file_path: str, line: int) -> str:
        if self.inline_error is not None:
            raise self.inline_error
        self.inline_calls.append(
            {
                "body": body,
                "relative_file_path": relative_file_path,
                "line": line,
            }
        )
        return "inline-url"

    def post_note(self, body: str) -> str:
        self.note_calls.append(body)
        return "note-url"


class ServiceGitlabClient(RecordingGitlabClient):
    instances: list["ServiceGitlabClient"] = []

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _comment(*, start: int = 3, end: int = 7) -> ReviewComment:
    return ReviewComment(
        title="Null check is missing",
        body="The new branch dereferences `payload` before validating it.",
        confidence_score=0.85,
        priority=1,
        code_location=CodeLocation(
            relative_file_path="src/module.py",
            line_range=LineRange(start=start, end=end),
        ),
    )


def test_publish_all_posts_inline_discussions() -> None:
    client = RecordingGitlabClient()
    publisher = ReviewCommentsPublisher(
        gitlab_client=client,
        comments=[_comment().model_dump()],
    )

    stats = publisher.publish_all()

    assert stats == {"inline": 1, "fallback_notes": 0, "errors": 0}
    assert client.note_calls == []
    assert client.inline_calls == [
        {
            "body": "### [P1][Confidence: 85%] Null check is missing\n\n"
            "The new branch dereferences `payload` before validating it.\n\n"
            "- Location: src/module.py:3-7",
            "relative_file_path": "src/module.py",
            "line": 7,
        }
    ]


def test_publish_all_falls_back_to_note_when_inline_publish_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GITLAB_API_TOKEN", "secret123")
    client = RecordingGitlabClient(
        inline_error=RuntimeError(
            "HTTP 400 for https://gitlab.example/api/v4/projects/group%2Fproject/merge_requests/42/discussions: "
            "PRIVATE-TOKEN=secret123"
        )
    )
    publisher = ReviewCommentsPublisher(
        gitlab_client=client,
        comments=[_comment().model_dump()],
    )

    stats = publisher.publish_all()

    assert stats == {"inline": 0, "fallback_notes": 1, "errors": 0}
    assert client.inline_calls == []
    assert client.note_calls == [
        "### [P1][Confidence: 85%] Null check is missing\n\n"
        "The new branch dereferences `payload` before validating it.\n\n"
        "- Location: src/module.py:3-7\n\n"
        "_Inline publish fallback was used._"
    ]
    output = capsys.readouterr().out
    assert "[review-agent][publish] inline discussion failed" in output
    assert "path=src/module.py, line=7" in output
    assert "<redacted-url>" in output
    assert "PRIVATE-TOKEN=<redacted>" in output
    assert "secret123" not in output


def test_publish_service_posts_success_note_when_review_has_no_comments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ServiceGitlabClient.instances.clear()
    monkeypatch.setattr("scripts.review_agent.publish_runner.GitlabMergeRequestClient", ServiceGitlabClient)
    review_path = tmp_path / "review.json"
    review_path.write_text(ReviewResult(comments=[]).model_dump_json(), encoding="utf-8")

    service = MergeRequestPublishService(
        api_base="https://gitlab.example/api/v4",
        project_id="group/project",
        merge_request_id="42",
        base_sha="base",
        head_sha="head",
        review_path=review_path,
        gitlab_api_token="token",
    )

    exit_code = service.run()

    assert exit_code == 0
    assert len(ServiceGitlabClient.instances) == 1
    assert ServiceGitlabClient.instances[0].kwargs == {
        "api_base": "https://gitlab.example/api/v4",
        "project_id": "group/project",
        "merge_request_id": "42",
        "base_sha": "base",
        "head_sha": "head",
        "token": "token",
    }
    assert ServiceGitlabClient.instances[0].note_calls == [
        "**The review is completed. No problems found**"
    ]
