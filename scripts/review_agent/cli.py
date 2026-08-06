"""CLI entrypoint for review agent workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def require_env(name: str, default: str | None = None) -> str:
    """Return an environment variable value or a provided default."""
    value = os.environ.get(name)
    if value:
        return value
    if default is not None:
        return default
    message = f"Missing required environment variable: {name}"
    raise RuntimeError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.review_agent.cli",
        description="Review-agent merge request CLI with review and publish modes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser(
        "review",
        help="Run review generation and save structured comments.",
    )
    review_parser.add_argument(
        "--base-sha",
        required=True,
        help="Base commit SHA of the merge request diff.",
    )
    review_parser.add_argument(
        "--output-path",
        required=True,
        help="Path to JSON output file produced by review mode.",
    )

    publish_parser = subparsers.add_parser(
        "publish",
        help="Publish comments from structured JSON to GitLab MR.",
    )
    publish_parser.add_argument(
        "--api-base",
        required=True,
        help="GitLab API base URL.",
    )
    publish_parser.add_argument(
        "--project-id",
        required=True,
        help="GitLab project ID or project path.",
    )
    publish_parser.add_argument(
        "--merge-request-id",
        required=True,
        help="Merge request IID.",
    )
    publish_parser.add_argument(
        "--base-sha",
        required=True,
        help="Base SHA for GitLab inline discussion position.",
    )
    publish_parser.add_argument(
        "--head-sha",
        required=True,
        help="Head SHA for GitLab inline discussion position.",
    )
    publish_parser.add_argument(
        "--review-path",
        required=True,
        help="Path to structured JSON review result produced by review mode.",
    )

    return parser


def main() -> int:
    """Run the review-agent CLI and dispatch the selected subcommand."""
    args = _build_parser().parse_args()

    if args.command == "review":
        from scripts.review_agent.review_runner import MergeRequestReviewService

        service = MergeRequestReviewService(
            base_sha=args.base_sha,
            output_path=Path(args.output_path),
            api_key=require_env("ANTHROPIC_API_KEY"),
            model=require_env("ANTHROPIC_MODEL"),
            base_url=require_env("ANTHROPIC_BASE_URL", default="https://api.anthropic.com"),
            api_version=require_env("ANTHROPIC_API_VERSION", default="2023-06-01"),
        )
        return service.run()

    if args.command == "publish":
        from scripts.review_agent.publish_runner import MergeRequestPublishService

        service = MergeRequestPublishService(
            api_base=args.api_base,
            project_id=args.project_id,
            merge_request_id=args.merge_request_id,
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            review_path=Path(args.review_path),
            gitlab_api_token=require_env("GITLAB_API_TOKEN"),
        )
        return service.run()

    message = f"Unsupported command: {args.command}"
    raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
