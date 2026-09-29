"""GPU-gated tests for local engines (transformers / vLLM).

Default runs skip cleanly: construction tests skip via importorskip when
torch/transformers/vllm are absent (the CI case), and inference tests
additionally require KOBALT_TEST_GPU_LIVE=1 since they need weights,
a GPU, and (for downloads) network. Real inference is manual-only.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu


def _cfg_for(family, model="sshleifer/tiny-gpt2"):
    from kobalt_eval.config import default_run_config

    cfg = default_run_config()
    cfg.backend.family = family
    cfg.backend.model = model
    return cfg


def _sample_messages():
    from kobalt_eval.prompts import build_messages

    return build_messages("다음 중 옳은 것은? (A) 가 (B) 나")


def test_transformers_backend_constructs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kobalt_eval.backends.transformers_backend import TransformersBackend

    backend = TransformersBackend.from_config(_cfg_for("transformers"))
    assert backend.model == "sshleifer/tiny-gpt2"
    assert backend.max_new_tokens == 2048
    assert backend.do_sample is False


@pytest.mark.skipif(
    os.environ.get("KOBALT_TEST_GPU_LIVE") != "1",
    reason="manual-only: needs model weights + GPU; set KOBALT_TEST_GPU_LIVE=1",
)
def test_transformers_backend_inference():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from kobalt_eval.backends.transformers_backend import TransformersBackend

    cfg = _cfg_for("transformers")
    backend = TransformersBackend.from_config(cfg)
    results = backend.generate([_sample_messages()], cfg)
    assert len(results) == 1
    assert isinstance(results[0].text, str)
    assert results[0].usage is not None
    assert results[0].usage["prompt_tokens"] > 0
    assert results[0].usage["completion_tokens"] >= 0


def test_vllm_backend_constructs():
    pytest.importorskip("vllm")
    from kobalt_eval.backends.vllm_backend import VLLMBackend

    backend = VLLMBackend.from_config(_cfg_for("vllm"))
    assert backend.model == "sshleifer/tiny-gpt2"
    assert backend.max_new_tokens == 2048
    assert backend.do_sample is False


@pytest.mark.skipif(
    os.environ.get("KOBALT_TEST_GPU_LIVE") != "1",
    reason="manual-only: needs model weights + GPU; set KOBALT_TEST_GPU_LIVE=1",
)
def test_vllm_backend_inference():
    pytest.importorskip("vllm")
    from kobalt_eval.backends.vllm_backend import VLLMBackend

    cfg = _cfg_for("vllm")
    backend = VLLMBackend.from_config(cfg)
    results = backend.generate([_sample_messages()], cfg)
    assert len(results) == 1
    assert isinstance(results[0].text, str)
    assert results[0].usage is not None
    assert results[0].usage["prompt_tokens"] > 0
