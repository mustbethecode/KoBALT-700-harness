"""Backend abstract contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kobalt_eval.config import RunConfig


class AuthenticationFailed(RuntimeError):
    """Fatal auth failure: the server rejected our API key (401/403).

    Raised by API backends when the provider SDK reports an authentication
    or permission error. The runner treats this as fatal and aborts the run
    immediately (no error-record swallowing, no scoring) so a bad key can
    never masquerade as a 0%-accuracy completed run.
    """


@dataclass
class GenerationResult:
    """One generated output plus provider-reported usage, when available.

    ``usage`` is the canonical usage object built by :func:`build_usage`
    (or None when the provider reported nothing). Local backends report
    token counts with no cost; API backends report what the provider sent.
    """

    text: str
    usage: dict[str, Any] | None = None


def _as_mapping(obj: Any) -> dict[str, Any] | None:
    """Best-effort read of an SDK usage object as a plain mapping.

    Handles pydantic-style objects (``model_dump()``), plain dicts, and
    attribute objects, so a future SDK change degrades to "usage unknown"
    instead of crashing the run.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            data = dump()
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001 - defensive: never fail the run over usage
            pass
    try:
        data = vars(obj)
    except TypeError:
        return None
    return data if isinstance(data, dict) else None


def build_usage(
    *,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    reasoning_tokens: Any = None,
    cost: Any = None,
    cost_source: str | None = None,
) -> dict[str, Any] | None:
    """Canonical usage object, or None when nothing is known.

    Only known fields are emitted (records stay lean), so consumers can
    distinguish "no usage recorded" (None) from a genuine zero. A cost is
    emitted only with provenance — ``cost_source`` is mandatory whenever a
    cost is present, so a measured provider cost can never be confused with
    an estimate.
    """
    out: dict[str, Any] = {}
    for key, val in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("reasoning_tokens", reasoning_tokens),
    ):
        if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
            out[key] = val
    if (
        isinstance(cost, (int, float))
        and not isinstance(cost, bool)
        and cost_source
    ):
        out["cost"] = float(cost)
        out["cost_source"] = cost_source
    return out or None


class Backend(ABC):
    """Pluggable model runner.

    All backends share one contract so prompt/render/parse/scoring code is
    identical across families:

        generate(messages_list: list[list[dict]], config)
            -> list[GenerationResult]

    Each result carries the raw output text and, when the provider or
    engine reports it, token usage/cost (see :class:`GenerationResult`).
    """

    @abstractmethod
    def generate(
        self,
        messages_list: list[list[dict]],
        config: "RunConfig",
    ) -> list[GenerationResult]:
        """Generate one result per messages list, in order."""
        raise NotImplementedError

    def generate_one(self, messages: list[dict], config: "RunConfig") -> GenerationResult:
        """Convenience: generate for a single item."""
        return self.generate([messages], config)[0]
