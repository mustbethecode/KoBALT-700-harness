"""OpenAI-compatible API backend (covers most providers + self-hosted servers).

Auth: API key from environment only (default ``OPENAI_API_KEY``, configurable
via ``backend.api_key_env``). The key is never logged or persisted (SEC-001).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from kobalt_eval.backends.base import (
    AuthenticationFailed,
    Backend,
    GenerationResult,
    _as_mapping,
    build_usage,
)

if TYPE_CHECKING:
    from kobalt_eval.config import RunConfig

_TRANSIENT_MARKERS = ("rate limit", "429", "500", "502", "503", "504", "timeout", "connection", "overloaded")


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "apiconnectionerror" in name or "ratelimiterror" in name or "internalservererror" in name:
        return True
    if "apierror" in name and any(code in msg for code in ("500", "502", "503", "504", "529")):
        return True
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _usage_from_response(resp: Any) -> dict[str, Any] | None:
    """Canonical usage from an OpenAI-compatible response, or None.

    OpenRouter includes ``usage.cost`` (USD credits) automatically and the
    OpenAI SDK surfaces unknown fields via pydantic extras; dumping the
    usage object covers both shapes.
    """
    data = _as_mapping(getattr(resp, "usage", None))
    if data is None:
        return None
    details = _as_mapping(data.get("completion_tokens_details")) or {}
    cost = data.get("cost")
    return build_usage(
        prompt_tokens=data.get("prompt_tokens"),
        completion_tokens=data.get("completion_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        cost=cost,
        cost_source="provider" if cost is not None else None,
    )


class OpenAICompatibleBackend(Backend):
    """Chat-completions backend via the ``openai`` SDK (lazy import)."""

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
        concurrency: int = 1,
        max_new_tokens: int = 2048,
        engine_opts: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency or 1)
        self.max_new_tokens = max_new_tokens
        self.engine_opts = dict(engine_opts or {})

    @classmethod
    def from_config(cls, config: "RunConfig") -> "OpenAICompatibleBackend":
        b = config.backend
        return cls(
            model=b.model,
            endpoint=b.endpoint,
            api_key_env=b.api_key_env or "OPENAI_API_KEY",
            max_retries=b.max_retries,
            concurrency=b.concurrency,
            max_new_tokens=config.generation.max_new_tokens,
            engine_opts=b.engine_opts,
        )

    def _client(self):
        try:
            from openai import OpenAI  # lazy: optional at import time
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required for the OpenAI-compatible backend. "
                "Install it with `uv sync` (core dependency)."
            ) from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key env var {self.api_key_env!r} is not set. "
                "Export it before running; keys are never passed via config files."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.endpoint:
            kwargs["base_url"] = self.endpoint
        return OpenAI(**kwargs)

    def _call_once(self, client, messages: list[dict]) -> GenerationResult:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            # Deterministic default (CON-002, matches upstream temperature=0.0).
            "temperature": 0.0,
        }
        # engine_opts may carry extra provider params (e.g. temperature); merge last.
        params.update(self.engine_opts)
        resp = client.chat.completions.create(**params)
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        return GenerationResult(text=text, usage=_usage_from_response(resp))

    def _as_auth_failed(self, exc: Exception) -> AuthenticationFailed | None:
        """Map SDK auth/permission errors to fatal AuthenticationFailed."""
        try:
            from openai import AuthenticationError, PermissionDeniedError  # lazy
        except ImportError:
            return None
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            status = getattr(exc, "status_code", None) or "?"
            where = self.endpoint or "the default OpenAI endpoint"
            return AuthenticationFailed(
                f"{status} from {where} — the API key from env var "
                f"{self.api_key_env} was rejected by the server. Verify the key value."
            )
        return None

    def _generate_single(self, messages: list[dict]) -> GenerationResult:
        client = self._client()
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(client, messages)
            except Exception as e:  # noqa: BLE001 - retry semantics need broad catch
                auth = self._as_auth_failed(e)
                if auth is not None:
                    raise auth from e
                last = e
                if attempt >= self.max_retries or not _is_transient(e):
                    raise
                time.sleep(2**attempt)
        assert last is not None
        raise last

    def generate(self, messages_list: list[list[dict]], config: "RunConfig") -> list[GenerationResult]:
        # Refresh per-run generation settings from config when provided.
        if config is not None and getattr(config, "generation", None) is not None:
            self.max_new_tokens = config.generation.max_new_tokens
        if len(messages_list) > 1 and self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                return list(pool.map(self._generate_single, messages_list))
        return [self._generate_single(m) for m in messages_list]
