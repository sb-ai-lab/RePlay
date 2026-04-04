"""Publish mode implementation for codex review CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.codex_review.gitlab_api import GitlabMergeRequestClient
from scripts.codex_review.schema import read_review_result


class ReviewCommentsProcessor:
    """Build GitLab-friendly discussion bodies from structured comments."""

    @staticmethod
    def to_discussion_body(comment: dict[str, Any]) -> tuple[str, str, int]:
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
            f"- Location: {path}:{line_label}\n\n"
            f"{body}"
        )
        return discussion_body, path, end_line


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
        self._processor = ReviewCommentsProcessor()

    def publish_all(self) -> dict[str, int]:
        inline_count = 0
        fallback_note_count = 0
        errors = 0

        for comment in self._comments:
            body, path, end_line = self._processor.to_discussion_body(comment)

            try:
                self._gitlab_client.post_inline_comment(
                    body=body,
                    relative_file_path=path,
                    line=end_line,
                )
                inline_count += 1
                continue
            except Exception as exc:
                fallback_body = (
                    f"{body}\n\n"
                    f"_Inline publish fallback was used. Error: {exc}_"
                )

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


def run_publish(*, gitlab_client: GitlabMergeRequestClient, review_path: Path) -> int:
    if not review_path.exists():
        message = f"Review file does not exist: {review_path}"
        raise RuntimeError(message)

    result = read_review_result(review_path)
    comments = [comment.model_dump() for comment in result.comments]
    if not comments:
        return 0

    publisher = ReviewCommentsPublisher(gitlab_client=gitlab_client, comments=comments)
    stats = publisher.publish_all()
    return 1 if stats["errors"] else 0
