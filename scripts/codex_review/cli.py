"""CLI entrypoint for codex review workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.codex_review.common import require_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.codex_review.cli",
        description="Codex-based merge request review CLI with review and publish modes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review", help="Run codex-cli review and save structured comments.")
    review_parser.add_argument(
        "--base-sha",
        required=True,
        help="Base commit SHA of the merge request diff.",
    )
    review_parser.add_argument(
        "--codex-sandbox-mode",
        required=True,
        help="Codex sandbox mode passed to codex exec, e.g. read-only or workspace-write.",
    )
    review_parser.add_argument(
        "--codex-proxy",
        required=True,
        help="Proxy URL used by codex-cli for model requests.",
    )
    review_parser.add_argument(
        "--output-path",
        required=True,
        help="Path to JSON output file produced by review mode.",
    )

    publish_parser = subparsers.add_parser("publish", help="Publish comments from structured JSON to GitLab MR.")
    publish_parser.add_argument(
        "--api-base",
        required=True,
        help="GitLab API base URL, e.g. https://gitlab.example/api/v4.",
    )
    publish_parser.add_argument(
        "--project-id",
        required=True,
        help="GitLab project ID or project path (raw or URL-encoded).",
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
    """Run the codex-review CLI and dispatch the selected subcommand."""
    args = _build_parser().parse_args()
    if args.command == "review":
        from scripts.codex_review.review_runner import MergeRequestReviewService

        service = MergeRequestReviewService(
            base_sha=args.base_sha,
            codex_sandbox_mode=args.codex_sandbox_mode,
            codex_proxy=args.codex_proxy,
            output_path=Path(args.output_path),
        )
        return service.run()

    if args.command == "publish":
        from scripts.codex_review.publish_runner import MergeRequestPublishService

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
