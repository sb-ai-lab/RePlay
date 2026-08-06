from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.review_agent import cli
from scripts.review_agent.common import require_env
from scripts.review_agent.schema import CodeLocation, LineRange, ReviewComment, ReviewResult


def _find_forbidden_name_references(root: Path, forbidden_substring: str) -> list[Path]:
    lowered_substring = forbidden_substring.casefold()
    offenders: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md"}:
            continue

        relative_path = path.relative_to(root).as_posix().casefold()
        file_contents = path.read_text(encoding="utf-8").casefold()

        if lowered_substring in relative_path or lowered_substring in file_contents:
            offenders.append(path)

    return offenders


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


def test_main_review_preserves_empty_optional_env_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_VERSION", "")
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
        "base_url": "",
        "api_version": "",
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


def test_main_publish_preserves_empty_required_env_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setenv("GITLAB_API_TOKEN", "")
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
        "gitlab_api_token": "",
    }
    assert calls[1] == ("run", None)


def test_require_env_returns_default_when_value_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)

    assert require_env("MISSING_ENV_VAR", default="fallback") == "fallback"


def test_require_env_returns_existing_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMPTY_ENV_VAR", "")

    assert require_env("EMPTY_ENV_VAR", default="fallback") == ""


def test_require_env_raises_when_value_missing_and_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_ENV_VAR", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_ENV_VAR"):
        require_env("MISSING_ENV_VAR")


def test_line_range_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="line_range.end must be >= line_range.start"):
        LineRange(start=3, end=2)


def test_review_result_accepts_structured_comment() -> None:
    result = ReviewResult(
        comments=[
            ReviewComment(
                title="Missing guard",
                body="The code path can raise on null input.",
                confidence_score=0.8,
                priority=2,
                code_location=CodeLocation(
                    relative_file_path="src/module.py",
                    line_range=LineRange(start=10, end=12),
                ),
            )
        ]
    )

    assert result.comments[0].code_location.relative_file_path == "src/module.py"


@pytest.mark.parametrize(
    ("payload", "error_field"),
    [
        (
            {
                "comments": [
                    {
                        "title": "Malformed line range",
                        "body": "start was emitted as a string.",
                        "confidence_score": 0.8,
                        "priority": 2,
                        "code_location": {
                            "relative_file_path": "src/module.py",
                            "line_range": {"start": "1", "end": 2},
                        },
                    }
                ]
            },
            "start",
        ),
        (
            {
                "comments": [
                    {
                        "title": "Malformed priority",
                        "body": "priority was emitted as a boolean.",
                        "confidence_score": 0.8,
                        "priority": True,
                        "code_location": {
                            "relative_file_path": "src/module.py",
                            "line_range": {"start": 1, "end": 2},
                        },
                    }
                ]
            },
            "priority",
        ),
        (
            {
                "comments": [
                    {
                        "title": "Malformed confidence",
                        "body": "confidence was emitted as a string.",
                        "confidence_score": "0.8",
                        "priority": 2,
                        "code_location": {
                            "relative_file_path": "src/module.py",
                            "line_range": {"start": 1, "end": 2},
                        },
                    }
                ]
            },
            "confidence_score",
        ),
    ],
)
def test_review_result_rejects_coerced_numeric_payloads(payload: dict[str, object], error_field: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ReviewResult.model_validate(payload)

    assert error_field in str(exc_info.value)


def test_forbidden_name_guard_detects_codex_in_reviewer_files(tmp_path: Path) -> None:
    package_root = tmp_path / "review_agent"
    package_root.mkdir()
    offender = package_root / "review_runner.py"
    offender.write_text('"""Legacy codex reference."""\n', encoding="utf-8")
    (package_root / "review_prompt.md").write_text("clean prompt\n", encoding="utf-8")

    assert _find_forbidden_name_references(package_root, "codex") == [offender]


def test_forbidden_name_guard_detects_codex_in_reviewer_paths(tmp_path: Path) -> None:
    package_root = tmp_path / "review_agent"
    package_root.mkdir()
    direct_offender = package_root / "codex_adapter.py"
    nested_offender = package_root / "codex" / "review_runner.py"
    nested_offender.parent.mkdir()
    direct_offender.write_text('"""Clean reviewer module."""\n', encoding="utf-8")
    nested_offender.write_text('"""Another clean reviewer module."""\n', encoding="utf-8")
    (package_root / "review_prompt.md").write_text("clean prompt\n", encoding="utf-8")

    assert _find_forbidden_name_references(package_root, "codex") == sorted(
        [direct_offender, nested_offender]
    )


def test_reviewer_package_files_do_not_contain_forbidden_codex_name() -> None:
    reviewer_root = Path(__file__).resolve().parents[3] / "scripts" / "review_agent"

    assert _find_forbidden_name_references(reviewer_root, "codex") == []
