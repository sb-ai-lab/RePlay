from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts.review_agent.review_runner import MergeRequestReviewService
from scripts.review_agent.schema import ReviewResult


class RecordingClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def create_review(self, *, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


def test_run_writes_validated_json_and_includes_changed_file_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_cmd(
        command: list[str],
        *,
        stream_stdout: bool = True,
        tee_output: bool = False,
        **_: object,
    ) -> CompletedProcess[str]:
        calls.append({"command": command, "stream_stdout": stream_stdout, "tee_output": tee_output})
        return CompletedProcess(
            args=command,
            returncode=0,
            stdout="M\0scripts/review_agent/common.py\0A\0tests/scripts/review_agent/test_review_runner.py\0",
            stderr="",
        )

    monkeypatch.setattr("scripts.review_agent.review_runner.run_cmd", fake_run_cmd)
    client = RecordingClient(
        json.dumps(
            {
                "comments": [
                    {
                        "title": "Null check is missing",
                        "body": "The new code dereferences payload before validating it.",
                        "confidence_score": 0.91,
                        "priority": 1,
                        "code_location": {
                            "relative_file_path": "scripts/review_agent/common.py",
                            "line_range": {"start": 12, "end": 12},
                        },
                    }
                ]
            }
        )
    )
    output_path = tmp_path / "artifacts" / "review.json"
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=output_path,
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=client,
    )

    exit_code = service.run()

    assert exit_code == 0
    assert calls == [
        {
            "command": ["git", "--no-pager", "diff", "--name-status", "-z", "abc123"],
            "stream_stdout": False,
            "tee_output": True,
        }
    ]
    assert len(client.prompts) == 1
    assert "base_sha: abc123" in client.prompts[0]
    assert '"status": "M"' in client.prompts[0]
    assert '"path": "scripts/review_agent/common.py"' in client.prompts[0]
    assert '"status": "A"' in client.prompts[0]
    assert '"path": "tests/scripts/review_agent/test_review_runner.py"' in client.prompts[0]
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == ReviewResult.model_validate(
        json.loads(client.response_text)
    ).model_dump(mode="json")


def test_run_raises_json_decode_error_for_invalid_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run_cmd(command: list[str], *, stream_stdout: bool = True, **_: object) -> CompletedProcess[str]:
        return CompletedProcess(args=command, returncode=0, stdout="M\0file.py\0", stderr="")

    monkeypatch.setattr("scripts.review_agent.review_runner.run_cmd", fake_run_cmd)
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=tmp_path / "review.json",
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=RecordingClient("not json"),
    )

    with pytest.raises(RuntimeError, match="non-JSON response"):
        service.run()


def test_run_accepts_markdown_wrapped_json_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run_cmd(command: list[str], *, stream_stdout: bool = True, **_: object) -> CompletedProcess[str]:
        return CompletedProcess(args=command, returncode=0, stdout="M\0file.py\0", stderr="")

    monkeypatch.setattr("scripts.review_agent.review_runner.run_cmd", fake_run_cmd)
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=tmp_path / "review.json",
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=RecordingClient('Here is the result:\n```json\n{"comments": []}\n```'),
    )

    exit_code = service.run()

    assert exit_code == 0
    assert json.loads((tmp_path / "review.json").read_text(encoding="utf-8")) == {"comments": []}


def test_run_rejects_comment_for_untouched_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run_cmd(command: list[str], *, stream_stdout: bool = True, **_: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=command,
            returncode=0,
            stdout="M\0scripts/review_agent/review_runner.py\0",
            stderr="",
        )

    monkeypatch.setattr("scripts.review_agent.review_runner.run_cmd", fake_run_cmd)
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=tmp_path / "review.json",
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=RecordingClient(
            json.dumps(
                {
                    "comments": [
                        {
                            "title": "This should be rejected",
                            "body": "The reported file is not part of the diff.",
                            "confidence_score": 0.8,
                            "priority": 1,
                            "code_location": {
                                "relative_file_path": "scripts/review_agent/common.py",
                                "line_range": {"start": 3, "end": 3},
                            },
                        }
                    ]
                }
            )
        ),
    )

    with pytest.raises(ValueError, match="untouched file"):
        service.run()


def test_run_serializes_changed_files_as_structured_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    odd_path = "odd`\nname.py"

    def fake_run_cmd(command: list[str], *, stream_stdout: bool = True, **_: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"M\0{odd_path}\0",
            stderr="",
        )

    monkeypatch.setattr("scripts.review_agent.review_runner.run_cmd", fake_run_cmd)
    client = RecordingClient(
        json.dumps(
            {
                "comments": [
                    {
                        "title": "Odd path is still reviewable",
                        "body": "Structured serialization should preserve this filename safely.",
                        "confidence_score": 0.7,
                        "priority": 2,
                        "code_location": {
                            "relative_file_path": odd_path,
                            "line_range": {"start": 1, "end": 1},
                        },
                    }
                ]
            }
        )
    )
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=tmp_path / "review.json",
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=client,
    )

    service.run()

    assert len(client.prompts) == 1
    assert '"changed_files": [' in client.prompts[0]
    assert '"status": "M"' in client.prompts[0]
    assert json.dumps(odd_path) in client.prompts[0]


def test_parse_name_status_output_supports_rename_records() -> None:
    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=Path("review.json"),
        api_key="test-key",
        model="claude-test",
        base_url="https://llm.example",
        api_version="2023-06-01",
        client=RecordingClient('{"comments": []}'),
    )

    changed_files = service._parse_name_status_output("R100\0old_name.py\0new_name.py\0")

    assert changed_files == [{"status": "R100", "old_path": "old_name.py", "path": "new_name.py"}]
