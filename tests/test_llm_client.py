import json
import threading
import time

import httpx
import pytest

from app.llm.client import OpenAICompatibleClient
from app.llm.errors import LLMError
from tests.conftest import make_settings

MESSAGES = [{"role": "user", "content": "hello"}]


def reply(content: str = '{"ok": true}') -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def make_client(handler, **kwargs) -> tuple[OpenAICompatibleClient, list[float]]:
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        base_url="https://llm.test/v1/",
        model="m",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleeps.append,
        **kwargs,
    )
    return client, sleeps


def test_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return reply()

    client, _ = make_client(handler, api_key="secret")
    assert client.chat(MESSAGES) == '{"ok": true}'
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_no_auth_header_without_key_and_json_mode_can_be_off():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return reply()

    client, _ = make_client(handler, json_mode=False)
    client.chat(MESSAGES)
    assert seen["auth"] is None
    assert "response_format" not in seen["body"]


def test_retries_transient_status_with_backoff_then_succeeds():
    statuses = iter([429, 503])

    def handler(request):
        status = next(statuses, 200)
        return reply() if status == 200 else httpx.Response(status)

    client, sleeps = make_client(handler, max_retries=2)
    assert client.chat(MESSAGES)
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]


def test_retry_after_header_is_honoured():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "7"}) if len(calls) == 1 else reply()

    client, sleeps = make_client(handler)
    client.chat(MESSAGES)
    assert sleeps == [7.0]


def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    client, _ = make_client(handler, max_retries=2)
    with pytest.raises(LLMError, match="3 attempt"):
        client.chat(MESSAGES)
    assert len(calls) == 3


def test_network_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectTimeout("slow")
        return reply()

    client, sleeps = make_client(handler)
    assert client.chat(MESSAGES)
    assert len(sleeps) == 1


def test_client_errors_are_not_retried_and_do_not_leak_secrets():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key sk-secret"})

    client, sleeps = make_client(handler, api_key="sk-secret")
    with pytest.raises(LLMError) as info:
        client.chat([{"role": "user", "content": "private resume text"}])
    assert len(calls) == 1 and not sleeps
    assert "sk-secret" not in str(info.value)
    assert "private resume text" not in str(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
    ],
)
def test_malformed_or_empty_responses_raise(response):
    client, _ = make_client(lambda r: response)
    with pytest.raises(LLMError):
        client.chat(MESSAGES)


def test_concurrency_is_capped():
    active = 0
    peak = 0
    lock = threading.Lock()

    def handler(request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return reply()

    client, _ = make_client(handler, max_concurrency=2)
    threads = [threading.Thread(target=client.chat, args=(MESSAGES,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2


def test_from_settings_uses_provider_presets():
    groq = OpenAICompatibleClient.from_settings(make_settings(llm_provider="groq", llm_api_key="k"))
    assert groq.model == "llama-3.1-8b-instant"
    assert groq._url == "https://api.groq.com/openai/v1/chat/completions"
    ollama = OpenAICompatibleClient.from_settings(make_settings(llm_provider="ollama"))
    assert ollama._url == "http://localhost:11434/v1/chat/completions"
    assert ollama._headers == {}
    custom = OpenAICompatibleClient.from_settings(
        make_settings(llm_provider="custom", llm_base_url="http://x/v1", llm_model="mine")
    )
    assert custom.model == "mine"


def test_llm_enabled_rules():
    assert make_settings(llm_provider="none").llm_enabled is False
    assert make_settings(llm_provider="groq").llm_enabled is False  # no key
    assert make_settings(llm_provider="groq", llm_api_key="  ").llm_enabled is False
    assert make_settings(llm_provider="groq", llm_api_key="k").llm_enabled is True
    assert make_settings(llm_provider="ollama").llm_enabled is True
    assert make_settings(llm_provider="custom").llm_enabled is False
    assert make_settings(llm_provider="custom", llm_base_url="http://x", llm_model="m").llm_enabled
