"""Usage/cost extraction and the backend usage contract (offline fakes).

Canonical usage object (all keys optional; unknown values omitted; no usage
known at all -> None):

    {"prompt_tokens": int, "completion_tokens": int, "reasoning_tokens": int,
     "cost": float, "cost_source": "provider"}

A cost is never emitted without provenance: `cost_source` is mandatory.
No real SDK, no network, no API keys.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeModel:
    """Mimics a pydantic-v2 SDK object (``model_dump()``)."""

    def __init__(self, **fields):
        self._fields = dict(fields)

    def model_dump(self):
        return dict(self._fields)


# --------------------------- build_usage ---------------------------


def test_build_usage_all_fields():
    from kobalt_eval.backends.base import build_usage

    assert build_usage(
        prompt_tokens=812,
        completion_tokens=1543,
        reasoning_tokens=640,
        cost=0.0031,
        cost_source="provider",
    ) == {
        "prompt_tokens": 812,
        "completion_tokens": 1543,
        "reasoning_tokens": 640,
        "cost": 0.0031,
        "cost_source": "provider",
    }


def test_build_usage_omits_unknown_fields():
    from kobalt_eval.backends.base import build_usage

    assert build_usage(prompt_tokens=5) == {"prompt_tokens": 5}
    assert build_usage(completion_tokens=0) == {"completion_tokens": 0}


def test_build_usage_returns_none_when_nothing_known():
    from kobalt_eval.backends.base import build_usage

    assert build_usage() is None
    assert build_usage(prompt_tokens=None, completion_tokens=None) is None


def test_build_usage_requires_cost_provenance():
    from kobalt_eval.backends.base import build_usage

    # A cost with no source is dropped: provenance is mandatory.
    assert build_usage(cost=1.5) is None
    assert build_usage(cost=1.5, cost_source="provider") == {
        "cost": 1.5,
        "cost_source": "provider",
    }
    assert build_usage(prompt_tokens=1, cost=None, cost_source="provider") == {
        "prompt_tokens": 1
    }


# --------------------------- OpenAI-compatible ---------------------------


def test_openai_usage_from_pydantic_like_response():
    from kobalt_eval.backends.api_openai import _usage_from_response

    resp = SimpleNamespace(
        usage=FakeModel(
            prompt_tokens=812,
            completion_tokens=1543,
            total_tokens=2355,
            prompt_tokens_details=FakeModel(cached_tokens=0),
            completion_tokens_details=FakeModel(reasoning_tokens=640),
            cost=0.0031,
            cost_details={"upstream_inference_cost": 0.02},
        )
    )
    assert _usage_from_response(resp) == {
        "prompt_tokens": 812,
        "completion_tokens": 1543,
        "reasoning_tokens": 640,
        "cost": 0.0031,
        "cost_source": "provider",
    }


def test_openai_usage_from_plain_dict():
    from kobalt_eval.backends.api_openai import _usage_from_response

    resp = SimpleNamespace(usage={"prompt_tokens": 10, "completion_tokens": 20})
    assert _usage_from_response(resp) == {"prompt_tokens": 10, "completion_tokens": 20}


def test_openai_usage_cost_only_is_kept_with_provenance():
    from kobalt_eval.backends.api_openai import _usage_from_response

    resp = SimpleNamespace(usage={"cost": 0.5})
    assert _usage_from_response(resp) == {"cost": 0.5, "cost_source": "provider"}


def test_openai_usage_absent_is_none():
    from kobalt_eval.backends.api_openai import _usage_from_response

    assert _usage_from_response(SimpleNamespace(usage=None)) is None
    assert _usage_from_response(SimpleNamespace()) is None


def test_openai_call_once_returns_generation_result():
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    def create(**params):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage={"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.001},
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = OpenAICompatibleBackend(model="m")._call_once(
        client, [{"role": "user", "content": "hi"}]
    )
    assert result.text == "ok"
    assert result.usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cost": 0.001,
        "cost_source": "provider",
    }


# --------------------------- Anthropic ---------------------------


def test_anthropic_usage_tokens_only():
    from kobalt_eval.backends.api_anthropic import _usage_from_response

    resp = SimpleNamespace(usage=FakeModel(input_tokens=100, output_tokens=50))
    assert _usage_from_response(resp) == {"prompt_tokens": 100, "completion_tokens": 50}


def test_anthropic_usage_absent_is_none():
    from kobalt_eval.backends.api_anthropic import _usage_from_response

    assert _usage_from_response(SimpleNamespace(usage=None)) is None
    assert _usage_from_response(SimpleNamespace()) is None


def test_anthropic_call_once_returns_generation_result():
    from kobalt_eval.backends.api_anthropic import AnthropicBackend

    def create(**params):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = AnthropicBackend(model="m")._call_once(
        client, [{"role": "user", "content": "hi"}]
    )
    assert result.text == "ok"
    assert result.usage == {"prompt_tokens": 11, "completion_tokens": 4}
