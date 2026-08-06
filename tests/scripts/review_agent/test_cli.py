from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from scripts.review_agent import cli


def test_review_parser_accepts_review_arguments() -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "review",
            "--base-sha",
            "abc123",
            "--output-path",
            "review.json",
        ]
    )

    assert args.command == "review"
    assert args.base_sha == "abc123"
    assert args.output_path == "review.json"


def test_publish_parser_accepts_publish_arguments() -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "publish",
            "--api-base",
            "https://gitlab.example/api/v4",
            "--project-id",
            "group/project",
            "--merge-request-id",
            "42",
            "--base-sha",
            "base",
            "--head-sha",
            "head",
            "--review-path",
            "review.json",
        ]
    )

    assert args.command == "publish"
    assert args.api_base == "https://gitlab.example/api/v4"
    assert args.project_id == "group/project"
    assert args.merge_request_id == "42"
    assert args.base_sha == "base"
    assert args.head_sha == "head"
    assert args.review_path == "review.json"


def test_main_dispatches_review_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyService:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def run(self) -> int:
            calls.append(("run", None))
            return 0

    review_runner_module = types.ModuleType("scripts.review_agent.review_runner")
    review_runner_module.MergeRequestReviewService = DummyService
    monkeypatch.setitem(sys.modules, "scripts.review_agent.review_runner", review_runner_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "test-model")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.review_agent.cli",
            "review",
            "--base-sha",
            "abc123",
            "--output-path",
            str(tmp_path / "review.json"),
        ],
    )

    assert cli.main() == 0
    assert calls[0][0] == "init"
    assert calls[0][1] == {
        "base_sha": "abc123",
        "output_path": tmp_path / "review.json",
        "api_key": "test-api-key",
        "model": "test-model",
        "base_url": "https://api.anthropic.com",
        "api_version": "2023-06-01",
    }
    assert calls[1] == ("run", None)


def test_main_dispatches_publish_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyService:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def run(self) -> int:
            calls.append(("run", None))
            return 0

    publish_runner_module = types.ModuleType("scripts.review_agent.publish_runner")
    publish_runner_module.MergeRequestPublishService = DummyService
    monkeypatch.setitem(sys.modules, "scripts.review_agent.publish_runner", publish_runner_module)
    monkeypatch.setenv("GITLAB_API_TOKEN", "gitlab-token")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.review_agent.cli",
            "publish",
            "--api-base",
            "https://gitlab.example/api/v4",
            "--project-id",
            "group/project",
            "--merge-request-id",
            "42",
            "--base-sha",
            "base",
            "--head-sha",
            "head",
            "--review-path",
            str(tmp_path / "review.json"),
        ],
    )

    assert cli.main() == 0
    assert calls[0][0] == "init"
    assert calls[0][1] == {
        "api_base": "https://gitlab.example/api/v4",
        "project_id": "group/project",
        "merge_request_id": "42",
        "base_sha": "base",
        "head_sha": "head",
        "review_path": tmp_path / "review.json",
        "gitlab_api_token": "gitlab-token",
    }
    assert calls[1] == ("run", None)


def test_require_env_returns_default_when_value_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)

    assert cli.require_env("MISSING_ENV_VAR", default="fallback") == "fallback"


def test_require_env_raises_when_value_missing_and_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_ENV_VAR"):
        cli.require_env("MISSING_ENV_VAR")
