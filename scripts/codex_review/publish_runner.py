"""Publish mode implementation for codex review CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.codex_review.gitlab_client import GitlabMergeRequestClient
from scripts.codex_review.schema import ReviewResult


class ReviewCommentsPublisher:
    """Publish comments one by one to GitLab MR discussions."""

    def __init__(
        self,
        *,
        gitlab_client: GitlabMergeRequestClient,
        comments: list[dict[str, Any]],
    ) -> None:
        """Initialize comment publisher for one merge request.

        Args:
            gitlab_client: Configured GitLab MR client instance.
            comments: Structured review comments to publish.
        """
        self._gitlab_client = gitlab_client
        self._comments = comments

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
            f"### [P{priority}][Confidence: {confidence_percent}%] {title}\n\n"
            f"{body}\n\n"
            f"- Location: {path}:{line_label}"
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
                fallback_body = f"{body}\n\n_Inline publish fallback was used. Error: {exc}_"

            try:
                self._gitlab_client.post_note(fallback_body)
                fallback_note_count += 1
            except Exception:
                errors += 1

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
        """Initialize the merge request publish service.

        Args:
            api_base: GitLab API base URL.
            project_id: GitLab project ID or path.
            merge_request_id: Target merge request IID.
            base_sha: Base commit SHA for inline positions.
            head_sha: Head commit SHA for inline positions.
            review_path: Path to structured review JSON file.
            gitlab_api_token: GitLab API token used for publishing.
        """
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
        if not comments:
            return 0

        publisher = ReviewCommentsPublisher(gitlab_client=self._gitlab_client, comments=comments)
        stats = publisher.publish_all()
        return 1 if stats["errors"] else 0
