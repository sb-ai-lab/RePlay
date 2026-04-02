"""Review mode implementation for codex review CLI."""

from __future__ import annotations

from pathlib import Path

from scripts.codex_review.codex_client import CodexClient
from scripts.codex_review.common import env, read_text_file, require_env, run_cmd
from scripts.codex_review.schema import parse_review_result_text, write_review_result

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
        self._base_sha = base_sha
        self._output_path = output_path
        self._codex_client = CodexClient(
            model=env("OPENAI_MODEL", "gpt-5.2-codex"),
            sandbox_mode=codex_sandbox_mode,
            proxy=codex_proxy,
            api_key=require_env("OPENAI_API_KEY"),
        )

    def _load_changed_files(self) -> str:
        command = ["git", "--no-pager", "diff", "--name-status", self._base_sha]
        completed = run_cmd(command)
        return completed.stdout.strip()

    def _build_prompt(self, changed_files_output: str) -> str:
        system_prompt = read_text_file(PROMPT_PATH)
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
        changed_files_output = self._load_changed_files()
        prompt = self._build_prompt(changed_files_output=changed_files_output)
        completed = self._codex_client.run_review_prompt(prompt)

        result = parse_review_result_text(completed.stdout)
        write_review_result(path=self._output_path, result=result)
        return 0
