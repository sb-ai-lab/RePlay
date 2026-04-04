"""GitLab Merge Request API client."""

from __future__ import annotations

from urllib import parse

from scripts.codex_review.common import request_json


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
        self._api_base = api_base
        self._project_id = project_id
        self._merge_request_id = merge_request_id
        self._base_sha = base_sha
        self._head_sha = head_sha
        self._token = token
        self._start_sha = start_sha

    @property
    def merge_request_api_url(self) -> str:
        api_base = self._api_base.rstrip("/")
        project_id = self._quote_once(self._project_id)
        mr_id = self._quote_once(self._merge_request_id)
        return f"{api_base}/projects/{project_id}/merge_requests/{mr_id}"

    @staticmethod
    def _quote_once(value: str) -> str:
        return parse.quote(parse.unquote(value), safe="")

    @property
    def start_sha(self) -> str:
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

    def post_inline_comment(self, *, body: str, relative_file_path: str, line: int) -> str:
        url = f"{self.merge_request_api_url}/discussions"
        payload = {
            "body": body,
            "position[position_type]": "text",
            "position[base_sha]": self._base_sha,
            "position[start_sha]": self.start_sha,
            "position[head_sha]": self._head_sha,
            "position[old_path]": relative_file_path,
            "position[new_path]": relative_file_path,
            "position[new_line]": str(line),
        }
        encoded_data = parse.urlencode(payload).encode("utf-8")
        request_json(method="POST", url=url, headers=self._form_headers(), data=encoded_data)
        return url

    def post_note(self, body: str) -> str:
        url = f"{self.merge_request_api_url}/notes"
        encoded_data = parse.urlencode({"body": body}).encode("utf-8")
        request_json(method="POST", url=url, headers=self._form_headers(), data=encoded_data)
        return url
