from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMError(Exception):
    """Raised when the configured LLM API cannot return a usable response."""


class OpenAICompatibleClient:
    """Small adapter for an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
        ).encode()
        request = Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except HTTPError as error:
            raise LLMError(f"LLM API returned HTTP {error.code}") from error
        except URLError as error:
            raise LLMError(f"LLM API request failed: {error.reason}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise LLMError(f"LLM API response could not be read: {error}") from error

        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("LLM API returned an unexpected response format") from error

        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls", []),
        }

