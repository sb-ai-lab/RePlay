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
    """Send an HTTP request and return the response body as text.

    Args:
        method: HTTP method (for example, GET or POST).
        url: Target request URL.
        headers: HTTP headers for the request.
        data: Optional raw request body.
        opener: Optional custom urllib opener.
    """
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
    """Send an HTTP request and parse the response body as JSON.

    Args:
        method: HTTP method (for example, GET or POST).
        url: Target request URL.
        headers: HTTP headers for the request.
        payload: Optional JSON-serializable payload.
        data: Optional raw request body.
        opener: Optional custom urllib opener.
    """
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


class GitlabMergeRequestClient:
    """Client for reading MR changes and publishing comments."""

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
        """Initialize a GitLab merge request API client.

        Args:
            api_base: GitLab API base URL.
            project_id: GitLab project ID or path.
            merge_request_id: Target merge request IID.
            base_sha: Base commit SHA for inline positions.
            head_sha: Head commit SHA for inline positions.
            token: GitLab API token.
            start_sha: Optional start SHA for diff positions.
        """
        self._api_base = api_base
        self._project_id = project_id
        self._merge_request_id = merge_request_id
        self._base_sha = base_sha
        self._head_sha = head_sha
        self._token = token
        self._start_sha = start_sha

    @property
    def merge_request_api_url(self) -> str:
        """Return the encoded API URL for the target merge request."""
        api_base = self._api_base.rstrip("/")
        project_id = self._quote_once(self._project_id)
        mr_id = self._quote_once(self._merge_request_id)
        return f"{api_base}/projects/{project_id}/merge_requests/{mr_id}"

    @staticmethod
    def _quote_once(value: str) -> str:
        return parse.quote(parse.unquote(value), safe="")

    @property
    def start_sha(self) -> str:
        """Return cached start_sha or fetch it from GitLab diff refs."""
        if self._start_sha:
            return self._start_sha
        self._start_sha = self._fetch_start_sha()
        return self._start_sha

    def _json_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "PRIVATE-TOKEN": self._token,
        }

    def _form_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/x-www-form-urlencoded",
            "PRIVATE-TOKEN": self._token,
        }

    def _fetch_start_sha(self) -> str:
        response = request_json(
            method="GET",
            url=self.merge_request_api_url,
            headers=self._json_headers(),
        )
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
        """Publish an inline MR discussion at the given file and line.

        Args:
            body: Comment body text.
            relative_file_path: Repository-relative file path for inline position.
            line: Line number in the new file version.
        """
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
        """Publish a regular merge request note.

        Args:
            body: Note body text.
        """
        return self._post_form(endpoint="notes", payload={"body": body})
