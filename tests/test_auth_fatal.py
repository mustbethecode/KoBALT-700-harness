"""Auth-fatal tests: a rejected API key aborts the run instead of yielding 0%."""

from __future__ import annotations

import json

import pytest

from helpers import STUB_OUTPUT_CORRECT_H, make_items, write_run_config
from kobalt_eval import cli
from kobalt_eval.backends.base import AuthenticationFailed, Backend, GenerationResult
from kobalt_eval.config import default_run_config
from kobalt_eval.runner import run_eval


class AuthFailBackend(Backend):
    """Always raises: simulates a server rejecting the API key (401)."""

    def generate(self, messages_list, config=None):
        raise AuthenticationFailed(
            "401 from the test endpoint — the API key from env var TEST_KEY "
            "was rejected by the server. Verify the key value."
        )


class FailAfterFirstBackend(Backend):
    """First call returns a canned output; every later call raises."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages_list, config=None):
        out = []
        for _ in messages_list:
            self.calls += 1
            if self.calls == 1:
                out.append(GenerationResult(text=STUB_OUTPUT_CORRECT_H))
            else:
                raise AuthenticationFailed(
                    "401 from the test endpoint — the API key from env var TEST_KEY "
                    "was rejected by the server. Verify the key value."
                )
        return out


def _config(model="stub-model"):
    cfg = default_run_config()
    cfg.backend.model = model
    return cfg


def _prediction_lines(run_dir):
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_run_eval_auth_failure_on_first_item_writes_no_records(tmp_path, items):
    run_dir = tmp_path / "run-auth"
    with pytest.raises(AuthenticationFailed, match="rejected"):
        run_eval(_config(), run_dir, dataset_items=items, backend=AuthFailBackend())
    # Nothing attempted before the failure -> zero records, and no scoring.
    assert _prediction_lines(run_dir) == []
    assert not (run_dir / "results.json").exists()


def test_run_eval_auth_failure_keeps_only_prior_records(tmp_path, items):
    run_dir = tmp_path / "run-auth-partial"
    with pytest.raises(AuthenticationFailed, match="rejected"):
        run_eval(_config(), run_dir, dataset_items=items, backend=FailAfterFirstBackend())
    lines = _prediction_lines(run_dir)
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == items[0]["id"]
    assert rec["predicted_answer"] == "H"
    assert "error" not in rec  # no error record for the never-answered item
    assert not (run_dir / "results.json").exists()


def test_cli_auth_failure_exit_1_no_results(tmp_path, monkeypatch, capsys):
    def fake_load(limit=None, **kwargs):
        return list(make_items()[:limit] if limit is not None else make_items())

    monkeypatch.setattr("kobalt_eval.runner.load_dataset_items", fake_load)
    monkeypatch.setattr("kobalt_eval.runner.get_dataset_revision", lambda **kw: "fake-rev")
    monkeypatch.setattr("kobalt_eval.backends.create_backend", lambda config: AuthFailBackend())

    cfg_path = write_run_config(tmp_path / "run.yaml", family="api", model="bad-key-model")
    run_dir = tmp_path / "run-bad-key"
    rc = cli.main(["run", "--config", str(cfg_path), "--out", str(run_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "rejected" in err
    assert not (run_dir / "results.json").exists()
    # Abort on the first item -> zero prediction records.
    assert _prediction_lines(run_dir) == []


def test_openai_sdk_401_maps_to_auth_failed():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, request=req)
    sdk_err = openai.AuthenticationError("Incorrect API key provided", response=resp, body=None)
    backend = OpenAICompatibleBackend(model="m", api_key_env="MY_KEY")
    mapped = backend._as_auth_failed(sdk_err)
    assert isinstance(mapped, AuthenticationFailed)
    assert "MY_KEY" in str(mapped)
    assert "rejected" in str(mapped)


def test_openai_sdk_403_maps_to_auth_failed():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(403, request=req)
    sdk_err = openai.PermissionDeniedError("Forbidden", response=resp, body=None)
    backend = OpenAICompatibleBackend(model="m")
    assert isinstance(backend._as_auth_failed(sdk_err), AuthenticationFailed)


def test_anthropic_sdk_401_maps_to_auth_failed():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    from kobalt_eval.backends.api_anthropic import AnthropicBackend

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(401, request=req)
    sdk_err = anthropic.AuthenticationError("invalid api key", response=resp, body=None)
    backend = AnthropicBackend(model="m", api_key_env="MY_ANTH_KEY")
    mapped = backend._as_auth_failed(sdk_err)
    assert isinstance(mapped, AuthenticationFailed)
    assert "MY_ANTH_KEY" in str(mapped)


def test_non_auth_errors_do_not_map_to_auth_failed():
    pytest.importorskip("openai")
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(model="m")
    assert backend._as_auth_failed(RuntimeError("boom")) is None
    assert backend._as_auth_failed(ValueError("nope")) is None
