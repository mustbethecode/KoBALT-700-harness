"""vLLM local backend (fast 700-item sweeps; all imports lazy)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kobalt_eval.backends.base import Backend, GenerationResult, build_usage

if TYPE_CHECKING:
    from kobalt_eval.config import RunConfig


class VLLMBackend(Backend):
    """Local inference via vLLM (lazy import; linux-only wheel)."""

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        engine_opts: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.engine_opts = dict(engine_opts or {})
        self._llm = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config: "RunConfig") -> "VLLMBackend":
        return cls(
            model=config.backend.model,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=config.generation.do_sample,
            engine_opts=config.backend.engine_opts,
        )

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM  # lazy
        except ImportError as e:
            raise RuntimeError(
                "The 'vllm' extra is required for the vLLM backend. "
                "Install it with `uv sync --extra vllm` (Linux only)."
            ) from e
        llm_kwargs: dict[str, Any] = {"model": self.model, "trust_remote_code": True}
        llm_kwargs.update(self.engine_opts.get("llm_kwargs", {}) if self.engine_opts else {})
        self._llm = LLM(**llm_kwargs)
        try:
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self._tokenizer = None

    def _render(self, messages: list[dict]) -> str:
        tok = self._tokenizer
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        # Fallback: simple role-prefixed concatenation.
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)

    def generate(self, messages_list: list[list[dict]], config: "RunConfig") -> list[GenerationResult]:
        self._ensure_loaded()
        try:
            from vllm import SamplingParams  # lazy
        except ImportError as e:
            raise RuntimeError("The 'vllm' extra is required for the vLLM backend.") from e
        do_sample = config.generation.do_sample if config is not None else self.do_sample
        max_new_tokens = config.generation.max_new_tokens if config is not None else self.max_new_tokens
        extra: dict[str, Any] = dict(self.engine_opts.get("sampling_kwargs", {}) if self.engine_opts else {})
        params = SamplingParams(
            temperature=extra.pop("temperature", 0.0 if not do_sample else 1.0),
            top_p=extra.pop("top_p", 1.0),
            max_tokens=extra.pop("max_tokens", max_new_tokens),
            **extra,
        )
        prompts = [self._render(m) for m in messages_list]
        outputs = self._llm.generate(prompts, params)
        results: list[GenerationResult] = []
        for out in outputs:
            try:
                text = out.outputs[0].text
            except (AttributeError, IndexError):
                text = ""
            results.append(GenerationResult(text=text, usage=_token_usage(out)))
        return results


def _token_usage(out: Any) -> dict[str, Any] | None:
    """Prompt/completion token counts from one vLLM RequestOutput.

    Local engines have no dollar cost, so no cost key is emitted. Counting
    is defensive: a vLLM version that omits token ids degrades to unknown.
    """
    prompt_ids = getattr(out, "prompt_token_ids", None)
    completion_ids = None
    try:
        completion_ids = out.outputs[0].token_ids
    except (AttributeError, IndexError):
        completion_ids = None
    return build_usage(
        prompt_tokens=len(prompt_ids) if prompt_ids is not None else None,
        completion_tokens=len(completion_ids) if completion_ids is not None else None,
    )
