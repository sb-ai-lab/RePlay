"""Review mode implementation for review-agent CLI."""

from __future__ import annotations

import json
import re
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
            tee_output=True,
        )
        return self._parse_name_status_output(completed.stdout)

    def _parse_name_status_output(self, raw_output: str) -> list[ChangedFileRecord]:
        entries = [entry for entry in raw_output.split("\0") if entry]
        changed_files: list[ChangedFileRecord] = []
        index = 0
        while index < len(entries):
            status = entries[index]
            if not status:
                message = "Unable to parse git diff --name-status -z output"
                raise ValueError(message)
            record: ChangedFileRecord = {"status": status}
            if status.startswith(("R", "C")):
                if index + 2 >= len(entries):
                    message = "Incomplete rename/copy record in git diff output"
                    raise ValueError(message)
                record["old_path"] = entries[index + 1]
                record["path"] = entries[index + 2]
                index += 3
            else:
                if index + 1 >= len(entries):
                    message = "Incomplete file record in git diff output"
                    raise ValueError(message)
                record["path"] = entries[index + 1]
                index += 2
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

    def _validate_changed_file_paths(self, result: ReviewResult, changed_files: list[ChangedFileRecord]) -> None:
        allowed_paths = {
            path for record in changed_files for path in (record.get("path"), record.get("old_path")) if path
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
            message = f"Review result references untouched file paths: {joined_paths}"
            raise ValueError(message)

    @staticmethod
    def _response_preview(text: str, *, limit: int = 200) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."

    @classmethod
    def _parse_response_payload(cls, raw_response: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        stripped = raw_response.strip()
        candidates = [stripped]

        fenced_blocks = re.findall(
            r"```(?:json)?\s*(.*?)```",
            raw_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload, end_index = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                pass
            else:
                if candidate[end_index:].strip():
                    continue
                if isinstance(payload, dict):
                    return payload

        for match in re.finditer(r"{", raw_response):
            try:
                payload, _ = decoder.raw_decode(raw_response[match.start() :].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        preview = cls._response_preview(raw_response)
        message = f"Review model returned non-JSON response: {preview}"
        raise RuntimeError(message)

    def run(self) -> int:
        changed_files = self._load_changed_files()
        prompt = self._build_prompt(changed_files)
        raw_response = self._client.create_review(prompt=prompt)
        payload = self._parse_response_payload(raw_response)
        result = ReviewResult.model_validate(payload)
        self._validate_changed_file_paths(result, changed_files)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return 0
