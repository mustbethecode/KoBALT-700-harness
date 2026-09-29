"""Native Anthropic adapter (for provider-specific behavior).

Auth: ``ANTHROPIC_API_KEY`` by default (configurable via
``backend.api_key_env``). Never logged or persisted (SEC-001).
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

_TRANSIENT_MARKERS = ("rate limit", "429", "500", "502", "503", "504", "529", "timeout", "connection", "overloaded")


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "internalserver" in name or "apiconnection" in name:
        return True
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _usage_from_response(resp: Any) -> dict[str, Any] | None:
    """Canonical usage from an Anthropic response, or None.

    The Messages API reports input/output token counts; it does not report
    a dollar cost, so no cost key is emitted.
    """
    data = _as_mapping(getattr(resp, "usage", None))
    if data is None:
        return None
    return build_usage(
        prompt_tokens=data.get("input_tokens"),
        completion_tokens=data.get("output_tokens"),
    )


class AnthropicBackend(Backend):
    """Native Anthropic messages backend via the ``anthropic`` SDK (lazy import)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 3,
        concurrency: int = 1,
        max_new_tokens: int = 2048,
        engine_opts: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency or 1)
        self.max_new_tokens = max_new_tokens
        self.engine_opts = dict(engine_opts or {})

    @classmethod
    def from_config(cls, config: "RunConfig") -> "AnthropicBackend":
        b = config.backend
        return cls(
            model=b.model,
            api_key_env=b.api_key_env or "ANTHROPIC_API_KEY",
            max_retries=b.max_retries,
            concurrency=b.concurrency,
            max_new_tokens=config.generation.max_new_tokens,
            engine_opts=b.engine_opts,
        )

    def _client(self):
        try:
            import anthropic  # lazy
        except ImportError as e:
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic backend. "
                "Install it with `uv sync` (core dependency)."
            ) from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key env var {self.api_key_env!r} is not set. "
                "Export it before running; keys are never passed via config files."
            )
        return anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _split_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Split system message out (Anthropic takes it as a top-level param)."""
        system: str | None = None
        rest: list[dict] = []
        for m in messages:
            if m.get("role") == "system" and system is None:
                system = m.get("content", "")
            else:
                rest.append({"role": m["role"], "content": m.get("content", "")})
        return system, rest

    def _call_once(self, client, messages: list[dict]) -> GenerationResult:
        system, rest = self._split_messages(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "messages": rest,
            # Deterministic default (CON-002, matches upstream temperature=0.0).
            "temperature": 0.0,
        }
        if system:
            params["system"] = system
        params.update(self.engine_opts)
        resp = client.messages.create(**params)
        parts: list[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", "") == "text":
                parts.append(getattr(block, "text", ""))
        return GenerationResult(text="".join(parts), usage=_usage_from_response(resp))

    def _as_auth_failed(self, exc: Exception) -> AuthenticationFailed | None:
        """Map SDK auth/permission errors to fatal AuthenticationFailed."""
        try:
            from anthropic import AuthenticationError, PermissionDeniedError  # lazy
        except ImportError:
            return None
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            status = getattr(exc, "status_code", None) or "?"
            return AuthenticationFailed(
                f"{status} from the Anthropic API — the API key from env var "
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
        if config is not None and getattr(config, "generation", None) is not None:
            self.max_new_tokens = config.generation.max_new_tokens
        if len(messages_list) > 1 and self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                return list(pool.map(self._generate_single, messages_list))
        return [self._generate_single(m) for m in messages_list]
