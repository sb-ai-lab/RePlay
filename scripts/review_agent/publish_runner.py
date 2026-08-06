"""Publish mode implementation for review-agent CLI."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from scripts.review_agent.gitlab_client import GitlabMergeRequestClient
from scripts.review_agent.schema import ReviewResult


class ReviewCommentsPublisher:
    """Publish comments one by one to GitLab MR discussions."""

    def __init__(
        self,
        *,
        gitlab_client: GitlabMergeRequestClient,
        comments: list[dict[str, Any]],
    ) -> None:
        self._gitlab_client = gitlab_client
        self._comments = comments

    @staticmethod
    def _sanitize_error_message(error: Exception) -> str:
        """Return a scrubbed error message safe for CI logs."""
        text = f"{type(error).__name__}: {error}"
        text = re.sub(r"https?://[^\s]+", "<redacted-url>", text)

        secret_pattern = (
            r"(?i)\b(authorization|private[-_ ]?token|token|api[-_ ]?key|password|secret)\b\s*[:=]\s*([^\s,;]+)"
        )
        text = re.sub(secret_pattern, r"\1=<redacted>", text)

        for env_name in ("GITLAB_API_TOKEN", "OPENAI_API_KEY", "CI_JOB_TOKEN"):
            value = os.getenv(env_name, "")
            if value:
                text = text.replace(value, "<redacted>")

        return text

    @staticmethod
    def _to_discussion_body(comment: dict[str, Any]) -> tuple[str, str, int]:
        title = str(comment["title"])
        body = str(comment["body"])
        priority = int(comment["priority"])
        confidence = float(comment["confidence_score"])
        location = comment["code_location"]
        path = str(location["relative_file_path"])
        line_range = location["line_range"]
        start_line = int(line_range["start"])
        end_line = int(line_range["end"])
        line_label = str(start_line) if start_line == end_line else f"{start_line}-{end_line}"
        confidence_percent = round(confidence * 100)

        discussion_body = (
            f"### [P{priority}][Confidence: {confidence_percent}%] {title}\n\n{body}\n\n- Location: {path}:{line_label}"
        )
        return discussion_body, path, end_line

    def publish_all(self) -> dict[str, int]:
        """Publish all comments and return publish statistics."""
        inline_count = 0
        fallback_note_count = 0
        errors = 0

        for comment in self._comments:
            body, path, end_line = self._to_discussion_body(comment)

            try:
                self._gitlab_client.post_inline_comment(
                    body=body,
                    relative_file_path=path,
                    line=end_line,
                )
                inline_count += 1
                continue
            except Exception as exc:
                safe_error = self._sanitize_error_message(exc)
                sys.stdout.write(
                    "[review-agent][publish] inline discussion failed, "
                    f"path={path}, line={end_line}, error={safe_error}"
                    "\n"
                )
                fallback_body = f"{body}\n\n_Inline publish fallback was used._"

            try:
                self._gitlab_client.post_note(fallback_body)
                fallback_note_count += 1
            except Exception:
                errors += 1

        if not self._comments:
            self._gitlab_client.post_note("**The review is completed. No problems found**")

        return {
            "inline": inline_count,
            "fallback_notes": fallback_note_count,
            "errors": errors,
        }


class MergeRequestPublishService:
    """Load review output and publish comments to GitLab MR."""

    def __init__(
        self,
        *,
        api_base: str,
        project_id: str,
        merge_request_id: str,
        base_sha: str,
        head_sha: str,
        review_path: Path,
        gitlab_api_token: str,
    ) -> None:
        self._api_base = api_base
        self._project_id = project_id
        self._merge_request_id = merge_request_id
        self._base_sha = base_sha
        self._head_sha = head_sha
        self._review_path = review_path
        self._gitlab_api_token = gitlab_api_token
        self._gitlab_client = GitlabMergeRequestClient(
            api_base=self._api_base,
            project_id=self._project_id,
            merge_request_id=self._merge_request_id,
            base_sha=self._base_sha,
            head_sha=self._head_sha,
            token=self._gitlab_api_token,
        )

    def run(self) -> int:
        """Load review output and publish comments to GitLab."""
        if not self._review_path.exists():
            message = f"Review file does not exist: {self._review_path}"
            raise RuntimeError(message)

        result = ReviewResult.model_validate_json(self._review_path.read_text(encoding="utf-8"))
        comments = [comment.model_dump() for comment in result.comments]
        publisher = ReviewCommentsPublisher(gitlab_client=self._gitlab_client, comments=comments)
        stats = publisher.publish_all()
        return 1 if stats["errors"] else 0
