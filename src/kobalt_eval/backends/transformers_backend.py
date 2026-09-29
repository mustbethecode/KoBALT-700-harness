"""HF transformers local backend (reproduces upstream scripts exactly).

Defaults: greedy (``do_sample=False``), ``max_new_tokens=2048``,
``torch_dtype="auto"``, ``device_map="auto"``, ``model.eval()``,
``eos_token_id`` from tokenizer when present, chat template via
``tokenizer.apply_chat_template(messages, add_generation_prompt=True,
tokenize=True, return_tensors="pt")``; decode with
``skip_special_tokens=True`` on the full output (upstream behavior).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kobalt_eval.backends.base import Backend, GenerationResult, build_usage

if TYPE_CHECKING:
    from kobalt_eval.config import RunConfig


class TransformersBackend(Backend):
    """Local inference via HuggingFace transformers (all imports lazy)."""

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
        self._tokenizer = None
        self._model = None

    @classmethod
    def from_config(cls, config: "RunConfig") -> "TransformersBackend":
        return cls(
            model=config.backend.model,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=config.generation.do_sample,
            engine_opts=config.backend.engine_opts,
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch  # lazy
            from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        except ImportError as e:
            raise RuntimeError(
                "The 'transformers' extra is required for the transformers backend. "
                "Install it with `uv sync --extra transformers`."
            ) from e
        load_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
        load_kwargs.update(self.engine_opts.get("model_kwargs", {}) if self.engine_opts else {})
        # Allow explicit torch_dtype override via engine_opts.
        if "torch_dtype" in (self.engine_opts or {}):
            dtype_name = self.engine_opts["torch_dtype"]
            load_kwargs["torch_dtype"] = getattr(torch, dtype_name, "auto")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, trust_remote_code=True, **load_kwargs
        )
        self._model.eval()

    def _generate_single(self, messages: list[dict], do_sample: bool, max_new_tokens: int) -> GenerationResult:
        self._ensure_loaded()
        tok = self._tokenizer
        mdl = self._model
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
        try:
            import torch  # lazy

            inputs = inputs.to(mdl.device)
        except Exception:
            pass
        gen_kwargs: dict[str, Any] = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
        }
        if getattr(tok, "eos_token_id", None) is not None:
            gen_kwargs["eos_token_id"] = tok.eos_token_id
        extra = dict(self.engine_opts.get("generate_kwargs", {}) if self.engine_opts else {})
        gen_kwargs.update(extra)
        outputs = mdl.generate(inputs, **gen_kwargs)
        # Upstream behavior: decode the full output (prompt + continuation).
        text = tok.decode(outputs[0], skip_special_tokens=True)
        usage = _token_usage(inputs, outputs)
        return GenerationResult(text=text, usage=usage)

    def generate(self, messages_list: list[list[dict]], config: "RunConfig") -> list[GenerationResult]:
        do_sample = config.generation.do_sample if config is not None else self.do_sample
        max_new_tokens = config.generation.max_new_tokens if config is not None else self.max_new_tokens
        return [self._generate_single(m, do_sample, max_new_tokens) for m in messages_list]


def _token_usage(inputs: Any, outputs: Any) -> dict[str, Any] | None:
    """Prompt/completion token counts from the tokenized input and output.

    Local engines have no dollar cost, so no cost key is emitted. Counting
    is defensive: a tensor-shape surprise degrades to "usage unknown".
    """
    try:
        prompt_tokens = int(inputs.shape[-1])
        completion_tokens = max(0, int(outputs.shape[-1]) - prompt_tokens)
    except Exception:  # noqa: BLE001 - defensive: never fail the run over usage
        return None
    return build_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
