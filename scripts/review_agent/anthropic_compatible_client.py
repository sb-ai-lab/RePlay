"""Anthropic-compatible HTTP client for review generation."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request


def _sanitize_error_text(text: str) -> str:
    return " ".join(text.split())


class AnthropicCompatibleClient:
    """Client for creating reviews through an Anthropic-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        api_version: str,
        opener: request.OpenerDirector | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._api_version = api_version
        self._opener = opener

    @property
    def messages_url(self) -> str:
        return f"{self._base_url.rstrip('/')}/v1/messages"

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
        req = request.Request(
            url=self.messages_url,
            data=json.dumps(self._payload(prompt)).encode("utf-8"),
            method="POST",
            headers=self._headers(),
        )
        open_method = self._opener.open if self._opener is not None else request.urlopen
        try:
            with open_method(req, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = _sanitize_error_text(exc.read().decode("utf-8", errors="replace"))
            raise RuntimeError(
                f"Anthropic-compatible API HTTP {exc.code} for {self.messages_url}: {details}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Anthropic-compatible API network error for {self.messages_url}: "
                f"{_sanitize_error_text(str(exc.reason))}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Anthropic-compatible API returned malformed JSON") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Anthropic-compatible API returned malformed JSON")

        content = payload.get("content")
        if not isinstance(content, list):
            raise RuntimeError("Anthropic-compatible response did not contain text content")

        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                return item["text"]

        raise RuntimeError("Anthropic-compatible response did not contain text content")
