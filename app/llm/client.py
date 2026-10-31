"""Chat client for OpenAI-compatible endpoints (Groq, OpenAI, OpenRouter, Ollama, ...)."""

import logging
import random
import threading
import time
from collections.abc import Callable
from typing import Protocol

import httpx

from app.core.config import Settings
from app.llm.errors import LLMError

logger = logging.getLogger(__name__)

Message = dict[str, str]
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 20.0


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[Message], *, json_mode: bool = True) -> str:
        """Send a chat completion request and return the assistant text."""
        ...


class OpenAICompatibleClient:
    """Minimal /chat/completions client with timeouts, retries and a concurrency cap."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_concurrency: int = 2,
        json_mode: bool = True,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._max_retries = max_retries
        self._json_mode = json_mode
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._http = http or httpx.Client(timeout=timeout)
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleClient":
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        return cls(
            base_url=settings.resolved_llm_base_url,
            model=settings.resolved_llm_model,
            api_key=key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            max_concurrency=settings.llm_max_concurrency,
            json_mode=settings.llm_json_mode,
        )

    def chat(self, messages: list[Message], *, json_mode: bool = True) -> str:
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0}
        if json_mode and self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = "unknown error"
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                with self._slots:
                    response = self._http.post(self._url, json=payload, headers=self._headers)
            except httpx.TransportError as exc:  # timeouts, connection errors
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    logger.debug(
                        "LLM call ok (model=%s, %.1fs)", self.model, time.perf_counter() - started
                    )
                    return _content(response)
                last_error = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    # Never include the request or headers here: they hold resume text and the key.
                    raise LLMError(f"LLM request rejected ({last_error})")
                retry_after = _retry_after(response)
            if attempt < self._max_retries:
                delay = min(MAX_BACKOFF_SECONDS, 0.5 * 2**attempt + random.uniform(0, 0.25))
                if last_error.startswith("HTTP") and retry_after is not None:
                    delay = min(MAX_BACKOFF_SECONDS, retry_after)
                logger.warning("LLM call failed (%s); retrying in %.1fs", last_error, delay)
                self._sleep(delay)
        raise LLMError(f"LLM request failed after {self._max_retries + 1} attempt(s): {last_error}")


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _content(response: httpx.Response) -> str:
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError("LLM response was not a chat completion") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM returned an empty message")
    return content
