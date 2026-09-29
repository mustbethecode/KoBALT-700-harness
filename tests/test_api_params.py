"""API request-param tests: deterministic temperature default (CON-002).

Uses fake client objects capturing kwargs — no network, no API keys.
"""

from __future__ import annotations

from types import SimpleNamespace


def _openai_client(captured):
    def create(**params):
        captured.update(params)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _anthropic_client(captured):
    def create(**params):
        captured.update(params)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_openai_sends_temperature_zero_by_default():
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(model="m")
    captured: dict = {}
    out = backend._call_once(_openai_client(captured), [{"role": "user", "content": "hi"}])
    assert out.text == "ok"
    assert captured["temperature"] == 0.0


def test_openai_engine_opts_overrides_temperature():
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(model="m", engine_opts={"temperature": 0.7})
    captured: dict = {}
    backend._call_once(_openai_client(captured), [{"role": "user", "content": "hi"}])
    assert captured["temperature"] == 0.7


def test_anthropic_sends_temperature_zero_by_default():
    from kobalt_eval.backends.api_anthropic import AnthropicBackend

    backend = AnthropicBackend(model="m")
    captured: dict = {}
    out = backend._call_once(
        _anthropic_client(captured),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert out.text == "ok"
    assert captured["temperature"] == 0.0
    assert captured["system"] == "sys"  # system extraction still intact


def test_anthropic_engine_opts_overrides_temperature():
    from kobalt_eval.backends.api_anthropic import AnthropicBackend

    backend = AnthropicBackend(model="m", engine_opts={"temperature": 0.7})
    captured: dict = {}
    backend._call_once(_anthropic_client(captured), [{"role": "user", "content": "hi"}])
    assert captured["temperature"] == 0.7
