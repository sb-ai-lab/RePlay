"""Review mode implementation for review-agent CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.review_agent.anthropic_compatible_client import AnthropicCompatibleClient
from scripts.review_agent.common import run_cmd
from scripts.review_agent.schema import ReviewResult

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "review_prompt.md"
ChangedFileRecord = dict[str, str]


class MergeRequestReviewService:
    """Generate structured review comments for a merge request."""

    def __init__(
        self,
        *,
        base_sha: str,
        output_path: Path,
        api_key: str,
        model: str,
        base_url: str,
        api_version: str,
        client: AnthropicCompatibleClient | None = None,
    ) -> None:
        self._base_sha = base_sha
        self._output_path = output_path
        self._client = client or AnthropicCompatibleClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_version=api_version,
        )

    def _load_changed_files(self) -> list[ChangedFileRecord]:
        completed = run_cmd(
            ["git", "--no-pager", "diff", "--name-status", "-z", self._base_sha],
            stream_stdout=False,
        )
        return self._parse_name_status_output(completed.stdout)

    def _parse_name_status_output(self, raw_output: str) -> list[ChangedFileRecord]:
        entries = [entry for entry in raw_output.split("\0") if entry]
        changed_files: list[ChangedFileRecord] = []
        index = 0
        while index < len(entries):
            entry = entries[index]
            status, _, path = entry.partition("\t")
            if not status or not path:
                raise ValueError("Unable to parse git diff --name-status -z output")
            record: ChangedFileRecord = {"status": status}
            if status.startswith(("R", "C")):
                if index + 1 >= len(entries):
                    raise ValueError("Incomplete rename/copy record in git diff output")
                record["old_path"] = path
                record["path"] = entries[index + 1]
                index += 2
            else:
                record["path"] = path
                index += 1
            changed_files.append(record)
        return changed_files

    def _build_prompt(self, changed_files: list[ChangedFileRecord]) -> str:
        prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
        changed_files_payload: dict[str, Any] = {
            "changed_files": changed_files,
        }
        review_context = (
            "Review context:\n"
            f"- base_sha: {self._base_sha}\n"
            "- changed files (JSON encoded from `git --no-pager diff --name-status -z base_sha`):\n"
            f"{json.dumps(changed_files_payload, indent=2)}\n\n"
            "Use the local repository state and git history as supporting context.\n"
            "Focus findings on the changed files listed above.\n"
        )
        return f"{prompt_template}\n\n{review_context}"

    def _validate_changed_file_paths(
        self, result: ReviewResult, changed_files: list[ChangedFileRecord]
    ) -> None:
        allowed_paths = {
            path
            for record in changed_files
            for path in (record.get("path"), record.get("old_path"))
            if path
        }
        untouched_paths = sorted(
            {
                comment.code_location.relative_file_path
                for comment in result.comments
                if comment.code_location.relative_file_path not in allowed_paths
            }
        )
        if untouched_paths:
            joined_paths = ", ".join(untouched_paths)
            raise ValueError(f"Review result references untouched file paths: {joined_paths}")

    def run(self) -> int:
        changed_files = self._load_changed_files()
        prompt = self._build_prompt(changed_files)
        raw_response = self._client.create_review(prompt=prompt)
        payload = json.loads(raw_response)
        result = ReviewResult.model_validate(payload)
        self._validate_changed_file_paths(result, changed_files)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return 0
