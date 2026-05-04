from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .models import LlmConfig, OpenRouterConfig


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, llm: LlmConfig, openrouter: OpenRouterConfig) -> None:
        self.llm = llm
        self.openrouter = openrouter

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set.")

        body = {
            "model": self.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.openrouter.site_url,
                "X-Title": self.openrouter.app_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response: {response_data}") from exc

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"LLM returned non-JSON content: {content}") from exc
