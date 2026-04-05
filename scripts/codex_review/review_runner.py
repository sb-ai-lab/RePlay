"""Review mode implementation for codex review CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.codex_review.codex_client import CodexClient
from scripts.codex_review.common import require_env, run_cmd
from scripts.codex_review.schema import ReviewResult

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "review_prompt.md"


class MergeRequestReviewService:
    """Generate review comments from git diff data via codex-cli."""

    def __init__(
        self,
        *,
        base_sha: str,
        codex_sandbox_mode: str,
        codex_proxy: str,
        output_path: Path,
    ) -> None:
        """Initialize the merge request review service.

        Args:
            base_sha: Base commit SHA used to collect changed files.
            codex_sandbox_mode: Codex sandbox mode for review execution.
            codex_proxy: Proxy URL used by Codex requests.
            output_path: Path where structured review output is written.
        """
        self._base_sha = base_sha
        self._output_path = output_path
        self._codex_client = CodexClient(
            model=require_env("OPENAI_MODEL"),
            sandbox_mode=codex_sandbox_mode,
            proxy=codex_proxy,
            api_key=require_env("OPENAI_API_KEY"),
        )

    def _load_changed_files(self) -> str:
        command = ["git", "--no-pager", "diff", "--name-status", self._base_sha]
        completed = run_cmd(command)
        return completed.stdout.strip()

    def _build_prompt(self, changed_files_output: str) -> str:
        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        changed_file_lines = changed_files_output or "(no changed files)"

        user_prompt = (
            "Review context:\n"
            f"- base_sha: {self._base_sha}\n"
            "- changed files (`git --no-pager diff --name-status base_sha`):\n"
            "```\n"
            f"{changed_file_lines}\n"
            "```\n\n"
            "Use the local repository state and git history to review these changes.\n"
            "Focus only on files listed above.\n"
        )
        return f"{system_prompt}\n\n{user_prompt}"

    def run(self) -> int:
        """Run review generation and write structured output to disk."""
        changed_files_output = self._load_changed_files()
        prompt = self._build_prompt(changed_files_output=changed_files_output)
        completed = self._codex_client.run_review_prompt(prompt)

        payload = json.loads(completed.stdout)
        result = ReviewResult.model_validate(payload)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return 0
