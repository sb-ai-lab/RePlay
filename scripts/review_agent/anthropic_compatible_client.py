"""Anthropic-compatible HTTP client for review generation."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request

LLM_REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRY_ATTEMPTS = 3


def _sanitize_error_text(text: str) -> str:
    return " ".join(text.split())


def _retry_after_seconds(headers: Any, *, attempt: int) -> float:
    retry_after: str | None = None
    if headers is not None:
        retry_after = headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return float(2 ** (attempt - 1))


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
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                with open_method(req, timeout=LLM_REQUEST_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                details = _sanitize_error_text(exc.read().decode("utf-8", errors="replace"))
                should_retry = exc.code == 429 or 500 <= exc.code < 600
                if should_retry and attempt < MAX_RETRY_ATTEMPTS:
                    time.sleep(_retry_after_seconds(exc.headers, attempt=attempt))
                    continue
                raise RuntimeError(
                    f"Anthropic-compatible API HTTP {exc.code} for {self.messages_url}: {details}"
                ) from exc
            except error.URLError as exc:
                if attempt < MAX_RETRY_ATTEMPTS:
                    time.sleep(float(2 ** (attempt - 1)))
                    continue
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

        text_chunks: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                text_chunks.append(item["text"])

        combined_text = "".join(text_chunks)
        if combined_text.strip():
            return combined_text

        raise RuntimeError("Anthropic-compatible response did not contain text content")
