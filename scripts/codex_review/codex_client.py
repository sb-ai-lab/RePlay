"""Codex CLI integration for non-interactive review runs."""

from __future__ import annotations

import subprocess

from scripts.codex_review.common import run_cmd


class CodexClient:
    """Client for running codex-cli review commands."""

    def __init__(
        self,
        *,
        model: str,
        sandbox_mode: str,
        proxy: str,
        binary: str = "codex",
    ) -> None:
        """Initialize a Codex client instance.

        Args:
            model: Model name used for Codex execution.
            sandbox_mode: Codex sandbox mode passed to CLI.
            proxy: Proxy URL used for HTTP and HTTPS requests.
            binary: Codex executable name or path.
        """
        self._model = model
        self._sandbox_mode = sandbox_mode
        self._proxy = proxy
        self._binary = binary

    def ensure_available(self) -> subprocess.CompletedProcess[str]:
        """Check that Codex is installed and authenticated."""
        run_cmd(
            [self._binary, "--version"],
            stream_stdout=False,
        )

        return run_cmd(
            [self._binary, "login", "status"],
            stream_stdout=False,
        )

    def run_review_prompt(self, prompt: str) -> subprocess.CompletedProcess[str]:
        """Run a non-interactive Codex review command."""
        self.ensure_available()

        env_overrides = {
            "HTTP_PROXY": self._proxy,
            "HTTPS_PROXY": self._proxy,
        }

        command = [
            self._binary,
            "exec",
            "--sandbox",
            self._sandbox_mode,
            "--model",
            self._model,
            "-",
        ]

        return run_cmd(
            command,
            input_text=prompt,
            env_overrides=env_overrides,
            stream_stdout=False,
        )