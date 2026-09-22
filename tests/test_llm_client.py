import pytest

from aiboke.config import LlmConfig
from aiboke.llm_client import LlmClient, LlmError


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("FakeSession 收到多余的请求")
        return self._responses.pop(0)


def _cfg():
    return LlmConfig(base_url="http://127.0.0.1:8080", model="qwen3.8-27b")


def _ok(text):
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def test_complete_returns_message_content():
    session = FakeSession([_ok("你好")])
    client = LlmClient(_cfg(), session=session)
    assert client.complete("sys", "user") == "你好"


def test_complete_hits_openai_compatible_endpoint():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("sys", "user")
    assert session.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"


def test_complete_sends_both_roles_and_disables_thinking():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("S", "U")
    messages = session.calls[0]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "S"}
    assert messages[1] == {"role": "user", "content": "U"}
    assert session.calls[0]["json"]["chat_template_kwargs"]["enable_thinking"] is False


def test_complete_passes_json_schema_when_given():
    schema = {"type": "object", "properties": {}}
    session = FakeSession([_ok("{}")])
    LlmClient(_cfg(), session=session).complete("s", "u", json_schema=schema)
    body = session.calls[0]["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_complete_omits_response_format_without_schema():
    session = FakeSession([_ok("x")])
    LlmClient(_cfg(), session=session).complete("s", "u")
    assert "response_format" not in session.calls[0]["json"]


def test_complete_raises_llm_error_on_transport_failure():
    class Boom:
        def post(self, *a, **k):
            raise ConnectionError("connection refused")

    with pytest.raises(LlmError, match="llama.cpp"):
        LlmClient(_cfg(), session=Boom()).complete("s", "u")


def test_complete_raises_llm_error_on_malformed_payload():
    session = FakeSession([FakeResponse({"unexpected": True})])
    with pytest.raises(LlmError, match="响应结构"):
        LlmClient(_cfg(), session=session).complete("s", "u")


def test_complete_rejects_empty_content():
    session = FakeSession([_ok("")])
    with pytest.raises(LlmError, match="空"):
        LlmClient(_cfg(), session=session).complete("s", "u")
