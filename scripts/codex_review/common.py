"""Shared utilities for codex review CLI."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, TextIO
from urllib import error, request


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def require_env(name: str) -> str:
    value = env(name)
    if not value:
        message = f"Missing required environment variable: {name}"
        raise RuntimeError(message)
    return value


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_cmd(
    command: list[str],
    *,
    input_text: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
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

    def stream_output(stream: TextIO | None, sink: TextIO, chunks: list[str]) -> None:
        if stream is None:
            return
        try:
            for chunk in stream:
                chunks.append(chunk)
                sink.write(chunk)
                sink.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, sys.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, sys.stderr, stderr_chunks),
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


def request_text(
    method: str,
    url: str,
    headers: dict[str, str],
    data: bytes | None = None,
    opener: request.OpenerDirector | None = None,
) -> str:
    req = request.Request(url=url, data=data, method=method, headers=headers)
    open_method = opener.open if opener is not None else request.urlopen
    try:
        with open_method(req, timeout=180) as response:
            return response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        message = f"HTTP {exc.code} for {url}: {details}"
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        message = f"Network error for {url}: {exc}"
        raise RuntimeError(message) from exc


def request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    data: bytes | None = None,
    opener: request.OpenerDirector | None = None,
) -> Any:
    raw_data = data
    if payload is not None:
        raw_data = json.dumps(payload).encode("utf-8")
    text = request_text(
        method=method,
        url=url,
        headers=headers,
        data=raw_data,
        opener=opener,
    )
    if not text.strip():
        return {}
    return json.loads(text)
