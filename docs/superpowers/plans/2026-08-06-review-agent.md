# Review Agent Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current legacy merge request reviewer with a neutral `review_agent` that uses an Anthropic-compatible HTTP API, preserves the `review`/`publish` CLI workflow, and keeps GitLab review publication behavior unchanged.

**Architecture:** Build a new `scripts/review_agent` package by porting the stable publish/schema/common logic from the legacy reviewer package, replacing only the review execution path with an Anthropic-compatible HTTP client. Update GitLab CI to call the new package and use `ANTHROPIC_*` variables, then remove the old reviewer package and all reviewer-related legacy branding in one coherent migration.

**Tech Stack:** Python 3, argparse, pydantic v2, urllib, pytest, GitLab CI YAML

---

## File Structure

### Files to create

- `scripts/review_agent/__init__.py`
- `scripts/review_agent/cli.py`
- `scripts/review_agent/common.py`
- `scripts/review_agent/schema.py`
- `scripts/review_agent/gitlab_client.py`
- `scripts/review_agent/publish_runner.py`
- `scripts/review_agent/review_runner.py`
- `scripts/review_agent/anthropic_compatible_client.py`
- `scripts/review_agent/prompts/review_prompt.md`
- `tests/scripts/review_agent/test_cli.py`
- `tests/scripts/review_agent/test_anthropic_compatible_client.py`
- `tests/scripts/review_agent/test_review_runner.py`
- `tests/scripts/review_agent/test_publish_runner.py`
- `tests/scripts/review_agent/test_gitlab_client.py`

### Files to modify

- `.gitlab/workflows/main.yml`

### Files to delete at the end

- `scripts/codex_review/__init__.py`
- `scripts/codex_review/cli.py`
- `scripts/codex_review/common.py`
- `scripts/codex_review/schema.py`
- `scripts/codex_review/gitlab_client.py`
- `scripts/codex_review/publish_runner.py`
- `scripts/codex_review/review_runner.py`
- `scripts/codex_review/codex_client.py`
- `scripts/codex_review/prompts/review_prompt.md`

### Responsibility map

- `cli.py`: parse arguments and dispatch `review` and `publish`
- `common.py`: required env lookup and subprocess helper for git diff only
- `schema.py`: structured review JSON contract
- `gitlab_client.py`: GitLab API transport and inline/note publishing
- `publish_runner.py`: publish orchestration and fallback behavior
- `anthropic_compatible_client.py`: HTTP client for Anthropic-compatible API endpoints
- `review_runner.py`: prompt assembly, changed-file collection, API call, validation, artifact write
- `review_prompt.md`: vendor-neutral review prompt that returns schema-compatible JSON

### Notes before starting

- Keep the JSON artifact shape identical unless a test proves that a cosmetic rename would break publish compatibility.
- Reuse the existing `urllib` transport style already present in the legacy reviewer GitLab client to avoid introducing `requests`.
- Ignore the untracked `.codex` path shown by `git status`; do not add it to commits.

### Task 1: Scaffold the new package and pin the CLI contract

**Files:**
- Create: `scripts/review_agent/__init__.py`
- Create: `scripts/review_agent/cli.py`
- Create: `tests/scripts/review_agent/test_cli.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.review_agent import cli


def test_review_parser_accepts_review_agent_arguments() -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "review",
            "--base-sha",
            "abc123",
            "--output-path",
            "review.json",
        ]
    )

    assert args.command == "review"
    assert args.base_sha == "abc123"
    assert args.output_path == "review.json"


def test_publish_parser_accepts_existing_contract_shape() -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "publish",
            "--api-base",
            "https://gitlab.example/api/v4",
            "--project-id",
            "group/project",
            "--merge-request-id",
            "42",
            "--base-sha",
            "base",
            "--head-sha",
            "head",
            "--review-path",
            "review.json",
        ]
    )

    assert args.command == "publish"
    assert args.review_path == "review.json"


def test_main_dispatches_review_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyService:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def run(self) -> int:
            calls.append(("run", None))
            return 0

    monkeypatch.setattr("scripts.review_agent.review_runner.MergeRequestReviewService", DummyService)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python",
            "-m",
            "scripts.review_agent.cli",
            "review",
            "--base-sha",
            "abc123",
            "--output-path",
            str(tmp_path / "review.json"),
        ],
    )

    assert cli.main() == 0
    assert calls[0][0] == "init"
    assert calls[1] == ("run", None)
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `pytest tests/scripts/review_agent/test_cli.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.review_agent'`

- [ ] **Step 3: Create the package marker**

```python
"""Review agent package for merge request review and publication."""
```

Save as `scripts/review_agent/__init__.py`.

- [ ] **Step 4: Implement the initial CLI module**

```python
"""CLI entrypoint for review agent workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.review_agent.common import require_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.review_agent.cli",
        description="Review-agent merge request CLI with review and publish modes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review", help="Run review generation and save structured comments.")
    review_parser.add_argument("--base-sha", required=True, help="Base commit SHA of the merge request diff.")
    review_parser.add_argument("--output-path", required=True, help="Path to JSON output file produced by review mode.")

    publish_parser = subparsers.add_parser("publish", help="Publish comments from structured JSON to GitLab MR.")
    publish_parser.add_argument("--api-base", required=True, help="GitLab API base URL.")
    publish_parser.add_argument("--project-id", required=True, help="GitLab project ID or project path.")
    publish_parser.add_argument("--merge-request-id", required=True, help="Merge request IID.")
    publish_parser.add_argument("--base-sha", required=True, help="Base SHA for GitLab inline discussion position.")
    publish_parser.add_argument("--head-sha", required=True, help="Head SHA for GitLab inline discussion position.")
    publish_parser.add_argument("--review-path", required=True, help="Path to structured JSON review result produced by review mode.")
    return parser


def main() -> int:
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

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the CLI tests again**

Run: `pytest tests/scripts/review_agent/test_cli.py -v`

Expected: FAIL with import errors for `scripts.review_agent.common` or `scripts.review_agent.review_runner`

- [ ] **Step 6: Commit the scaffolding**

```bash
git add scripts/review_agent/__init__.py scripts/review_agent/cli.py tests/scripts/review_agent/test_cli.py
git commit -m "feat: scaffold review agent cli"
```

### Task 2: Port shared common and schema utilities under neutral naming

**Files:**
- Create: `scripts/review_agent/common.py`
- Create: `scripts/review_agent/schema.py`
- Modify: `tests/scripts/review_agent/test_cli.py`

- [ ] **Step 1: Extend CLI tests with env helper expectations**

```python
def test_require_env_returns_default_when_value_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    from scripts.review_agent.common import require_env

    assert require_env("ANTHROPIC_BASE_URL", default="https://api.anthropic.com") == "https://api.anthropic.com"


def test_require_env_raises_when_value_missing_and_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from scripts.review_agent.common import require_env

    with pytest.raises(RuntimeError, match="Missing required environment variable: ANTHROPIC_API_KEY"):
        require_env("ANTHROPIC_API_KEY")
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/scripts/review_agent/test_cli.py -v`

Expected: FAIL because `scripts.review_agent.common` does not exist or `require_env` has no `default` support

- [ ] **Step 3: Implement `common.py`**

```python
"""Shared utilities for review agent CLI."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
from typing import TextIO


def require_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    if default is not None:
        return default
    raise RuntimeError(f"Missing required environment variable: {name}")


def run_cmd(
    command: list[str],
    *,
    input_text: str | None = None,
    env_overrides: dict[str, str] | None = None,
    stream_stdout: bool = True,
    stream_stderr: bool = True,
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

    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, sys.stdout, stdout_chunks, stream_stdout), daemon=True)
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, sys.stderr, stderr_chunks, stream_stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    if input_text is not None and process.stdin is not None:
        process.stdin.write(input_text)
        process.stdin.close()

    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    completed = subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed
```

- [ ] **Step 4: Implement `schema.py`**

```python
"""Pydantic schema for structured review output."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class LineRange(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LineRange":
        if self.end < self.start:
            raise ValueError("line_range.end must be >= line_range.start")
        return self


class CodeLocation(BaseModel):
    relative_file_path: str = Field(min_length=1)
    line_range: LineRange


class ReviewComment(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=0, le=3)
    code_location: CodeLocation


class ReviewResult(BaseModel):
    comments: list[ReviewComment]
```

- [ ] **Step 5: Run the CLI tests again**

Run: `pytest tests/scripts/review_agent/test_cli.py -v`

Expected: PASS for parser/env tests, with any remaining failures limited to missing review/publish runner modules

- [ ] **Step 6: Commit the shared port**

```bash
git add scripts/review_agent/common.py scripts/review_agent/schema.py tests/scripts/review_agent/test_cli.py
git commit -m "feat: port review agent common and schema"
```

### Task 3: Port GitLab client and publish runner with compatibility tests

**Files:**
- Create: `scripts/review_agent/gitlab_client.py`
- Create: `scripts/review_agent/publish_runner.py`
- Create: `tests/scripts/review_agent/test_gitlab_client.py`
- Create: `tests/scripts/review_agent/test_publish_runner.py`

- [ ] **Step 1: Write the failing GitLab client and publish tests**

```python
from __future__ import annotations

from pathlib import Path

from scripts.review_agent.publish_runner import MergeRequestPublishService, ReviewCommentsPublisher


class DummyGitlabClient:
    def __init__(self) -> None:
        self.inline_calls: list[tuple[str, str, int]] = []
        self.note_calls: list[str] = []
        self.fail_inline = False

    def post_inline_comment(self, *, body: str, relative_file_path: str, line: int) -> str:
        if self.fail_inline:
            raise RuntimeError("inline failure")
        self.inline_calls.append((body, relative_file_path, line))
        return "ok"

    def post_note(self, body: str) -> str:
        self.note_calls.append(body)
        return "ok"


def test_publish_all_uses_inline_comment_by_default() -> None:
    client = DummyGitlabClient()
    publisher = ReviewCommentsPublisher(
        gitlab_client=client,
        comments=[
            {
                "title": "Bug",
                "body": "Broken branch for empty input; return early.",
                "confidence_score": 0.95,
                "priority": 1,
                "code_location": {"relative_file_path": "pkg/file.py", "line_range": {"start": 10, "end": 12}},
            }
        ],
    )

    stats = publisher.publish_all()

    assert stats == {"inline": 1, "fallback_notes": 0, "errors": 0}
    assert client.inline_calls[0][1:] == ("pkg/file.py", 12)


def test_publish_all_falls_back_to_note_when_inline_fails() -> None:
    client = DummyGitlabClient()
    client.fail_inline = True
    publisher = ReviewCommentsPublisher(
        gitlab_client=client,
        comments=[
            {
                "title": "Bug",
                "body": "Broken branch for empty input; return early.",
                "confidence_score": 0.95,
                "priority": 1,
                "code_location": {"relative_file_path": "pkg/file.py", "line_range": {"start": 10, "end": 12}},
            }
        ],
    )

    stats = publisher.publish_all()

    assert stats == {"inline": 0, "fallback_notes": 1, "errors": 0}
    assert "_Inline publish fallback was used._" in client.note_calls[0]


def test_publish_service_posts_success_note_for_empty_review(tmp_path: Path, monkeypatch) -> None:
    review_path = tmp_path / "review.json"
    review_path.write_text('{"comments": []}', encoding="utf-8")

    client = DummyGitlabClient()
    monkeypatch.setattr("scripts.review_agent.publish_runner.GitlabMergeRequestClient", lambda **_: client)

    service = MergeRequestPublishService(
        api_base="https://gitlab.example/api/v4",
        project_id="group/project",
        merge_request_id="42",
        base_sha="base",
        head_sha="head",
        review_path=review_path,
        gitlab_api_token="token",
    )

    assert service.run() == 0
    assert client.note_calls == ["**The review is completed. No problems found**"]
```

- [ ] **Step 2: Run the publish tests to verify they fail**

Run: `pytest tests/scripts/review_agent/test_publish_runner.py -v`

Expected: FAIL because `scripts.review_agent.publish_runner` and `scripts.review_agent.gitlab_client` do not exist

- [ ] **Step 3: Port `gitlab_client.py` with neutral imports**

```python
"""GitLab Merge Request API client."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, parse, request


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
        raise RuntimeError(f"HTTP {exc.code} for {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    data: bytes | None = None,
    opener: request.OpenerDirector | None = None,
) -> Any:
    raw_data = data if payload is None else json.dumps(payload).encode("utf-8")
    text = request_text(method=method, url=url, headers=headers, data=raw_data, opener=opener)
    if not text.strip():
        return {}
    return json.loads(text)


class GitlabMergeRequestClient:
    def __init__(
        self,
        *,
        api_base: str,
        project_id: str,
        merge_request_id: str,
        base_sha: str,
        head_sha: str,
        token: str,
        start_sha: str | None = None,
    ) -> None:
        self._api_base = api_base
        self._project_id = project_id
        self._merge_request_id = merge_request_id
        self._base_sha = base_sha
        self._head_sha = head_sha
        self._token = token
        self._start_sha = start_sha

    @staticmethod
    def _quote_once(value: str) -> str:
        return parse.quote(parse.unquote(value), safe="")

    @property
    def merge_request_api_url(self) -> str:
        api_base = self._api_base.rstrip("/")
        project_id = self._quote_once(self._project_id)
        mr_id = self._quote_once(self._merge_request_id)
        return f"{api_base}/projects/{project_id}/merge_requests/{mr_id}"

    @property
    def start_sha(self) -> str:
        if self._start_sha:
            return self._start_sha
        self._start_sha = self._fetch_start_sha()
        return self._start_sha

    def _json_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "PRIVATE-TOKEN": self._token}

    def _form_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/x-www-form-urlencoded", "PRIVATE-TOKEN": self._token}

    def _fetch_start_sha(self) -> str:
        response = request_json(method="GET", url=self.merge_request_api_url, headers=self._json_headers())
        if not isinstance(response, dict):
            return self._base_sha
        diff_refs = response.get("diff_refs")
        if not isinstance(diff_refs, dict):
            return self._base_sha
        start_sha = diff_refs.get("start_sha")
        if isinstance(start_sha, str) and start_sha.strip():
            return start_sha.strip()
        return self._base_sha

    def _post_form(self, *, endpoint: str, payload: dict[str, str]) -> str:
        url = f"{self.merge_request_api_url}/{endpoint}"
        encoded_data = parse.urlencode(payload).encode("utf-8")
        request_json(method="POST", url=url, headers=self._form_headers(), data=encoded_data)
        return url

    def post_inline_comment(self, *, body: str, relative_file_path: str, line: int) -> str:
        payload = {
            "body": body,
            "position[position_type]": "text",
            "position[base_sha]": self._base_sha,
            "position[start_sha]": self.start_sha,
            "position[head_sha]": self._head_sha,
            "position[new_path]": relative_file_path,
            "position[new_line]": str(line),
        }
        return self._post_form(endpoint="discussions", payload=payload)

    def post_note(self, body: str) -> str:
        return self._post_form(endpoint="notes", payload={"body": body})
```

- [ ] **Step 4: Port `publish_runner.py` with neutral log prefixes and env redaction**

```python
"""Publish mode implementation for review agent CLI."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from scripts.review_agent.gitlab_client import GitlabMergeRequestClient
from scripts.review_agent.schema import ReviewResult


class ReviewCommentsPublisher:
    def __init__(self, *, gitlab_client: GitlabMergeRequestClient, comments: list[dict[str, Any]]) -> None:
        self._gitlab_client = gitlab_client
        self._comments = comments

    @staticmethod
    def _sanitize_error_message(error: Exception) -> str:
        text = f"{type(error).__name__}: {error}"
        text = re.sub(r"https?://[^\s]+", "<redacted-url>", text)
        secret_pattern = r"(?i)\b(authorization|private[-_ ]?token|token|api[-_ ]?key|password|secret)\b\s*[:=]\s*([^\s,;]+)"
        text = re.sub(secret_pattern, r"\1=<redacted>", text)
        for env_name in ("GITLAB_API_TOKEN", "ANTHROPIC_API_KEY", "CI_JOB_TOKEN"):
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
        discussion_body = f"### [P{priority}][Confidence: {confidence_percent}%] {title}\n\n{body}\n\n- Location: {path}:{line_label}"
        return discussion_body, path, end_line

    def publish_all(self) -> dict[str, int]:
        inline_count = 0
        fallback_note_count = 0
        errors = 0
        for comment in self._comments:
            body, path, end_line = self._to_discussion_body(comment)
            try:
                self._gitlab_client.post_inline_comment(body=body, relative_file_path=path, line=end_line)
                inline_count += 1
                continue
            except Exception as exc:
                safe_error = self._sanitize_error_message(exc)
                sys.stdout.write(
                    "[review-agent][publish] inline discussion failed, "
                    f"path={path}, line={end_line}, error={safe_error}\n"
                )
                fallback_body = f"{body}\n\n_Inline publish fallback was used._"
            try:
                self._gitlab_client.post_note(fallback_body)
                fallback_note_count += 1
            except Exception:
                errors += 1
        if not self._comments:
            self._gitlab_client.post_note("**The review is completed. No problems found**")
        return {"inline": inline_count, "fallback_notes": fallback_note_count, "errors": errors}


class MergeRequestPublishService:
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
        self._review_path = review_path
        self._gitlab_client = GitlabMergeRequestClient(
            api_base=api_base,
            project_id=project_id,
            merge_request_id=merge_request_id,
            base_sha=base_sha,
            head_sha=head_sha,
            token=gitlab_api_token,
        )

    def run(self) -> int:
        if not self._review_path.exists():
            raise RuntimeError(f"Review file does not exist: {self._review_path}")
        result = ReviewResult.model_validate_json(self._review_path.read_text(encoding="utf-8"))
        comments = [comment.model_dump() for comment in result.comments]
        stats = ReviewCommentsPublisher(gitlab_client=self._gitlab_client, comments=comments).publish_all()
        return 1 if stats["errors"] else 0
```

- [ ] **Step 5: Run the publish compatibility tests**

Run: `pytest tests/scripts/review_agent/test_publish_runner.py -v`

Expected: PASS

- [ ] **Step 6: Commit the publish port**

```bash
git add scripts/review_agent/gitlab_client.py scripts/review_agent/publish_runner.py tests/scripts/review_agent/test_publish_runner.py tests/scripts/review_agent/test_gitlab_client.py
git commit -m "feat: port review agent publish flow"
```

### Task 4: Build the Anthropic-compatible HTTP client with endpoint configurability

**Files:**
- Create: `scripts/review_agent/anthropic_compatible_client.py`
- Create: `tests/scripts/review_agent/test_anthropic_compatible_client.py`

- [ ] **Step 1: Write the failing client tests**

```python
from __future__ import annotations

import json
from urllib import request

import pytest

from scripts.review_agent.anthropic_compatible_client import AnthropicCompatibleClient


class DummyResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class DummyOpener:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests: list[request.Request] = []

    def open(self, req: request.Request, timeout: int = 180) -> DummyResponse:
        self.requests.append(req)
        return DummyResponse(self.payload)


def test_messages_request_uses_base_url_headers_and_model() -> None:
    opener = DummyOpener(
        {
            "content": [
                {
                    "type": "text",
                    "text": '{"comments": []}',
                }
            ]
        }
    )
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    result = client.create_review(prompt="Return JSON")

    assert result == '{"comments": []}'
    req = opener.requests[0]
    assert req.full_url == "https://gateway.example/anthropic/v1/messages"
    assert req.headers["x-api-key"] == "secret"
    assert req.headers["anthropic-version"] == "2023-06-01"


def test_client_raises_on_missing_text_content() -> None:
    opener = DummyOpener({"content": [{"type": "image"}]})
    client = AnthropicCompatibleClient(
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example/anthropic",
        api_version="2023-06-01",
        opener=opener,
    )

    with pytest.raises(RuntimeError, match="Anthropic-compatible response did not contain text content"):
        client.create_review(prompt="Return JSON")
```

- [ ] **Step 2: Run the client tests to verify they fail**

Run: `pytest tests/scripts/review_agent/test_anthropic_compatible_client.py -v`

Expected: FAIL because `scripts.review_agent.anthropic_compatible_client` does not exist

- [ ] **Step 3: Implement the HTTP client**

```python
"""Anthropic-compatible HTTP client for review generation."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class AnthropicCompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        api_version: str,
        opener: request.OpenerDirector | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._opener = opener

    @property
    def messages_url(self) -> str:
        return f"{self._base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._api_version,
        }

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

    def create_review(self, *, prompt: str) -> str:
        data = json.dumps(self._payload(prompt)).encode("utf-8")
        req = request.Request(url=self.messages_url, data=data, method="POST", headers=self._headers())
        open_method = self._opener.open if self._opener is not None else request.urlopen
        try:
            with open_method(req, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic-compatible API HTTP {exc.code} for {self.messages_url}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Anthropic-compatible API network error for {self.messages_url}: {exc}") from exc

        content = payload.get("content")
        if not isinstance(content, list):
            raise RuntimeError("Anthropic-compatible response did not contain content list")
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                return item["text"]
        raise RuntimeError("Anthropic-compatible response did not contain text content")
```

- [ ] **Step 4: Run the client tests again**

Run: `pytest tests/scripts/review_agent/test_anthropic_compatible_client.py -v`

Expected: PASS

- [ ] **Step 5: Commit the review transport**

```bash
git add scripts/review_agent/anthropic_compatible_client.py tests/scripts/review_agent/test_anthropic_compatible_client.py
git commit -m "feat: add anthropic compatible review client"
```

### Task 5: Implement the review runner and prompt assembly with schema validation

**Files:**
- Create: `scripts/review_agent/review_runner.py`
- Create: `scripts/review_agent/prompts/review_prompt.md`
- Create: `tests/scripts/review_agent/test_review_runner.py`

- [ ] **Step 1: Write the failing review-runner tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from scripts.review_agent.review_runner import MergeRequestReviewService


class DummyReviewClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.prompts: list[str] = []

    def create_review(self, *, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


def test_review_runner_writes_validated_json(tmp_path: Path) -> None:
    output_path = tmp_path / "review.json"
    client = DummyReviewClient(
        json.dumps(
            {
                "comments": [
                    {
                        "title": "Guard empty input",
                        "body": "The new branch dereferences an empty list. Return early when no items are present.",
                        "confidence_score": 0.96,
                        "priority": 1,
                        "code_location": {
                            "relative_file_path": "pkg/file.py",
                            "line_range": {"start": 7, "end": 7},
                        },
                    }
                ]
            }
        )
    )

    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=output_path,
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example",
        api_version="2023-06-01",
        client=client,
    )

    service._load_changed_files = lambda: "M\tpkg/file.py"
    service.run()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comments"][0]["title"] == "Guard empty input"
    assert "pkg/file.py" in client.prompts[0]


def test_review_runner_rejects_invalid_json(tmp_path: Path) -> None:
    output_path = tmp_path / "review.json"
    client = DummyReviewClient("not-json")

    service = MergeRequestReviewService(
        base_sha="abc123",
        output_path=output_path,
        api_key="secret",
        model="claude-sonnet",
        base_url="https://gateway.example",
        api_version="2023-06-01",
        client=client,
    )

    service._load_changed_files = lambda: "M\tpkg/file.py"

    with pytest.raises(json.JSONDecodeError):
        service.run()
```

- [ ] **Step 2: Run the review-runner tests to verify they fail**

Run: `pytest tests/scripts/review_agent/test_review_runner.py -v`

Expected: FAIL because `scripts.review_agent.review_runner` and the prompt file do not exist

- [ ] **Step 3: Implement the review prompt**

```markdown
You are a strict and senior-level code reviewer for a GitLab merge request.

Analyze the changed files listed in the review context. Use local repository state and git history only when needed to confirm a concrete issue.

Focus on:
1. bugs and logic errors,
2. security issues,
3. risky behavior changes and regressions,
4. critical dependency or build configuration mistakes,
5. important missing tests.

# Output language
- Output MUST be in English.

# Output format
Return output as JSON only.
- Do not add Markdown.
- Do not add explanations outside JSON.
- Return exactly one JSON object with the top-level field `comments`.

Schema:
{
  "comments": [
    {
      "title": "Short issue title, max 80 chars",
      "body": "Actionable explanation and concrete fix suggestion",
      "confidence_score": 0.0,
      "priority": 0,
      "code_location": {
        "relative_file_path": "path/from/repo/root.py",
        "line_range": {
          "start": 10,
          "end": 12
        }
      }
    }
  ]
}

# Rules
- Return only important findings.
- Do not invent issues.
- Do not repeat the same issue in multiple comments.
- If confidence is low or the finding is not actionable, omit it.
- If no important issues exist, return `{"comments": []}`.
```

- [ ] **Step 4: Implement the review runner**

```python
"""Review mode implementation for review agent CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.review_agent.anthropic_compatible_client import AnthropicCompatibleClient
from scripts.review_agent.common import run_cmd
from scripts.review_agent.schema import ReviewResult

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "review_prompt.md"


class MergeRequestReviewService:
    def __init__(
        self,
        *,
        base_sha: str,
        output_path: Path,
        api_key: str,
        model: str,
        base_url: str,
        api_version: str,
        client: AnthropicCompatibleClient | None = None,
    ) -> None:
        self._base_sha = base_sha
        self._output_path = output_path
        self._client = client or AnthropicCompatibleClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            api_version=api_version,
        )

    def _load_changed_files(self) -> str:
        completed = run_cmd(["git", "--no-pager", "diff", "--name-status", self._base_sha], stream_stdout=False)
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
        changed_files_output = self._load_changed_files()
        prompt = self._build_prompt(changed_files_output=changed_files_output)
        response_text = self._client.create_review(prompt=prompt)
        payload = json.loads(response_text)
        result = ReviewResult.model_validate(payload)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return 0
```

- [ ] **Step 5: Run the review-runner tests**

Run: `pytest tests/scripts/review_agent/test_review_runner.py -v`

Expected: PASS

- [ ] **Step 6: Commit the review orchestration**

```bash
git add scripts/review_agent/review_runner.py scripts/review_agent/prompts/review_prompt.md tests/scripts/review_agent/test_review_runner.py
git commit -m "feat: implement review agent review flow"
```

### Task 6: Migrate the GitLab CI job to the new package and variables

**Files:**
- Modify: `.gitlab/workflows/main.yml`

- [ ] **Step 1: Write a failing CI-focused grep check**

```bash
grep -nE 'legacy reviewer job|legacy auth blob|legacy reviewer env vars|old model variable|legacy login check|legacy reviewer package' .gitlab/workflows/main.yml
```

Expected: multiple matches in the current legacy reviewer job block.

- [ ] **Step 2: Replace the old install helper block and reviewer job definition**

Update the YAML around the existing reviewer section to this shape:

```yaml
.install_review_agent_deps: &install_review_agent_deps
  - pip install -q "pydantic>=2,<3"

review-agent:
  stage: code_quality
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event" && $CI_MERGE_REQUEST_SOURCE_PROJECT_ID == $CI_PROJECT_ID
      when: manual
    - when: never

  # Required CI variables:
  # - ANTHROPIC_API_KEY
  # - GITLAB_API_TOKEN
  #
  # Optional CI variables:
  # - ANTHROPIC_BASE_URL
  # - ANTHROPIC_API_VERSION

  variables:
    ANTHROPIC_MODEL: "claude-sonnet-4-20250514"
    ANTHROPIC_API_VERSION: "2023-06-01"
    REVIEW_AGENT_OUTPUT_FILE: "review-agent-output.json"

  before_script:
    - *install_review_agent_deps
    - |
      set -euo pipefail

      if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "Missing CI variable: ANTHROPIC_API_KEY"
        exit 1
      fi
  script:
    - mkdir -p "$(dirname "$REVIEW_AGENT_OUTPUT_FILE")"
    - BASE_SHA="${CI_MERGE_REQUEST_DIFF_BASE_SHA:-${CI_COMMIT_BEFORE_SHA:-$CI_COMMIT_SHA}}"
    - |
      python -m scripts.review_agent.cli review \
        --base-sha "$BASE_SHA" \
        --output-path "$REVIEW_AGENT_OUTPUT_FILE"
    - |
      python -m scripts.review_agent.cli publish \
        --api-base "$CI_API_V4_URL" \
        --project-id "$CI_PROJECT_ID" \
        --merge-request-id "$CI_MERGE_REQUEST_IID" \
        --base-sha "$BASE_SHA" \
        --head-sha "$CI_COMMIT_SHA" \
        --review-path "$REVIEW_AGENT_OUTPUT_FILE"
  artifacts:
    when: always
    paths:
      - $REVIEW_AGENT_OUTPUT_FILE
    expire_in: 1 day
  allow_failure: true
  tags:
    - ${RUNNER}
```

- [ ] **Step 3: Run the grep check again**

Run: `grep -nE 'review-agent|ANTHROPIC_API_KEY|ANTHROPIC_MODEL|scripts\.review_agent' .gitlab/workflows/main.yml`

Expected: no output

- [ ] **Step 4: Commit the CI migration**

```bash
git add .gitlab/workflows/main.yml
git commit -m "ci: migrate reviewer job to review agent"
```

### Task 7: Remove the old package and add regression checks for forbidden naming

**Files:**
- Delete: `scripts/codex_review/__init__.py`
- Delete: `scripts/codex_review/cli.py`
- Delete: `scripts/codex_review/common.py`
- Delete: `scripts/codex_review/schema.py`
- Delete: `scripts/codex_review/gitlab_client.py`
- Delete: `scripts/codex_review/publish_runner.py`
- Delete: `scripts/codex_review/review_runner.py`
- Delete: `scripts/codex_review/codex_client.py`
- Delete: `scripts/codex_review/prompts/review_prompt.md`
- Modify: `tests/scripts/review_agent/test_cli.py`

- [ ] **Step 1: Add a forbidden-naming regression test**

```python
from __future__ import annotations

from pathlib import Path


def test_reviewer_package_contains_no_codex_references() -> None:
    review_agent_root = Path("scripts/review_agent")
    forbidden = "codex"
    text_files = sorted(path for path in review_agent_root.rglob("*") if path.is_file() and path.suffix in {".py", ".md"})

    for path in text_files:
        content = path.read_text(encoding="utf-8").lower()
        assert forbidden not in content, f"forbidden '{forbidden}' found in {path}"
```

- [ ] **Step 2: Run the regression test to verify it fails before cleanup**

Run: `pytest tests/scripts/review_agent/test_cli.py::test_reviewer_package_contains_no_codex_references -v`

Expected: FAIL if any copied code or prompt still contains `codex`

- [ ] **Step 3: Delete the old package and clean any copied strings**

Use these commands:

```bash
rm -f scripts/codex_review/__init__.py \
      scripts/codex_review/cli.py \
      scripts/codex_review/common.py \
      scripts/codex_review/schema.py \
      scripts/codex_review/gitlab_client.py \
      scripts/codex_review/publish_runner.py \
      scripts/codex_review/review_runner.py \
      scripts/codex_review/codex_client.py \
      scripts/codex_review/prompts/review_prompt.md
find scripts/review_agent -type f \( -name '*.py' -o -name '*.md' \) -print0 | xargs -0 grep -nHi 'codex' || true
```

- [ ] **Step 4: Run the forbidden-naming regression test again**

Run: `pytest tests/scripts/review_agent/test_cli.py::test_reviewer_package_contains_no_codex_references -v`

Expected: PASS

- [ ] **Step 5: Commit the cleanup**

```bash
git add -A scripts/review_agent scripts/codex_review tests/scripts/review_agent
git commit -m "refactor: remove codex reviewer package"
```

### Task 8: Run verification commands for the complete migration

**Files:**
- Modify: `tests/scripts/review_agent/test_cli.py`
- Modify: `tests/scripts/review_agent/test_anthropic_compatible_client.py`
- Modify: `tests/scripts/review_agent/test_review_runner.py`
- Modify: `tests/scripts/review_agent/test_publish_runner.py`
- Modify: `tests/scripts/review_agent/test_gitlab_client.py`
- Modify: `.gitlab/workflows/main.yml`
- Modify: `scripts/review_agent/*.py`

- [ ] **Step 1: Run the focused reviewer test suite**

Run: `pytest tests/scripts/review_agent -v`

Expected: all review-agent tests PASS

- [ ] **Step 2: Run a repository-wide grep for forbidden reviewer naming**

Run: `grep -RIn --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.codex -E 'scripts\.review_agent|review-agent|ANTHROPIC_API_KEY|REVIEW_AGENT_OUTPUT_FILE' .`

Expected: no output

- [ ] **Step 3: Run a focused syntax check for the new package**

Run: `python -m compileall scripts/review_agent`

Expected: output showing successful compilation of the new review-agent modules

- [ ] **Step 4: Review git diff before the final commit**

Run: `git diff --stat HEAD~6..HEAD`

Expected: changes limited to the new `scripts/review_agent` package, its tests, CI YAML, and removal of `scripts/codex_review`

- [ ] **Step 5: Create the final integration commit**

```bash
git add scripts/review_agent tests/scripts/review_agent .gitlab/workflows/main.yml scripts/codex_review
git commit -m "feat: migrate merge request reviewer to review agent"
```

## Self-Review

### Spec coverage

- Neutral package naming: covered by Tasks 1, 7, and 8
- Anthropic-compatible API backend: covered by Task 4
- `z.ai` via `ANTHROPIC_BASE_URL`: covered by Task 4 and Task 6
- Preserve `review`/`publish` workflow: covered by Tasks 1, 3, 5, and 6
- Preserve publish semantics and JSON contract: covered by Tasks 2, 3, and 5
- Remove legacy reviewer authentication and CI assumptions: covered by Tasks 6, 7, and 8
- Migration completed as one coherent change: covered by Task 8 final verification

### Placeholder scan

- No `TODO`, `TBD`, or deferred implementation markers remain
- Every task contains exact files, concrete code, and explicit commands

### Type consistency

- `MergeRequestReviewService` constructor shape is consistent between CLI and tests
- `AnthropicCompatibleClient.create_review(prompt=...)` is used consistently in tests and runner
- `ReviewResult` schema contract is stable across runner and publish service
