"""Shared utilities for codex review CLI."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from typing import TextIO


def require_env(name: str) -> str:
    """Return a required environment variable value or raise an error.

    Args:
        name: Environment variable name.
    """
    value = os.getenv(name)
    if value is None:
        message = f"Missing required environment variable: {name}"
        raise RuntimeError(message)
    return value


def run_cmd(
    command: list[str],
    *,
    input_text: str | None = None,
    env_overrides: dict[str, str] | None = None,
    stream_stdout: bool = True,
    stream_stderr: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, optionally stream outputs, and return a completed process.

    Args:
        command: Command and arguments to execute.
        input_text: Optional stdin text passed to the process.
        env_overrides: Optional environment variables to override.
        stream_stdout: Whether to stream stdout to the console.
        stream_stderr: Whether to stream stderr to the console.
    """
    command_text = shlex.join(command)
    sys.stdout.write(f"\n[run_cmd] >>> {command_text}\n")
    sys.stdout.flush()

    process_env = os.environ.copy()
    if env_overrides:
        process_env.update(env_overrides)

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=process_env,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def stream_output(stream: TextIO | None, sink: TextIO, chunks: list[str], stream_enabled: bool) -> None:
        if stream is None:
            return
        try:
            for chunk in stream:
                chunks.append(chunk)
                if stream_enabled:
                    sink.write(chunk)
                    sink.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, sys.stdout, stdout_chunks, stream_stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, sys.stderr, stderr_chunks, stream_stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    if input_text is not None and process.stdin is not None:
        process.stdin.write(input_text)
        process.stdin.close()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)
    completed = subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout=stdout_text,
        stderr=stderr_text,
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=command,
            output=stdout_text,
            stderr=stderr_text,
        )
    return completed
