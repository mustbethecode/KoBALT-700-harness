"""Integration tests: run_eval with stub backends (no network/GPU)."""

from __future__ import annotations

import json

from helpers import (
    PREDICTION_FIELDS,
    STUB_OUTPUT_CORRECT_H,
    STUB_OUTPUT_MULTI,
    STUB_OUTPUT_NO_PHRASE,
    StubBackend,
)
from kobalt_eval.backends.base import Backend, GenerationResult
from kobalt_eval.config import default_run_config
from kobalt_eval.runner import run_eval
from kobalt_eval.scoring import score_run


def _config(model="stub-model"):
    cfg = default_run_config()
    cfg.backend.model = model
    return cfg


def _read_predictions(run_dir):
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_run_eval_produces_complete_run_dir(tmp_path, items):
    run_dir = tmp_path / "run-1"
    backend = StubBackend(
        [STUB_OUTPUT_CORRECT_H, STUB_OUTPUT_NO_PHRASE, STUB_OUTPUT_MULTI]
    )
    out = run_eval(_config(), run_dir, dataset_items=items, backend=backend)

    assert (out / "config.snapshot.yaml").exists()
    assert (out / "predictions.jsonl").exists()
    assert (out / "results.json").exists()
    assert (out / "run.log").exists()

    records = _read_predictions(out)
    assert len(records) == len(items) == 3
    for rec in records:
        assert set(rec.keys()) == PREDICTION_FIELDS
        assert isinstance(rec["latency_ms"], int) and rec["latency_ms"] >= 0

    by_id = {r["id"]: r for r in records}
    assert by_id["test-id-0001"]["predicted_answer"] == "H"
    assert by_id["test-id-0001"]["correct"] is True
    assert by_id["test-id-0002"]["predicted_answer"] is None
    assert by_id["test-id-0002"]["correct"] is False
    assert by_id["test-id-0003"]["predicted_answer"] == "A, C"
    assert by_id["test-id-0003"]["correct"] is False

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["num_items"] == 3
    assert results["num_correct"] == 1
    assert results["accuracy"] == 1 / 3


def test_usage_written_per_record_and_aggregated(tmp_path, items):
    run_dir = tmp_path / "run-usage"
    usages = [
        {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.01, "cost_source": "provider"},
        None,
        {"prompt_tokens": 30, "completion_tokens": 40, "reasoning_tokens": 5},
    ]
    backend = StubBackend(
        [STUB_OUTPUT_CORRECT_H, STUB_OUTPUT_NO_PHRASE, STUB_OUTPUT_MULTI],
        usages=usages,
    )
    run_eval(_config(), run_dir, dataset_items=items, backend=backend)

    records = _read_predictions(run_dir)
    assert records[0]["usage"] == usages[0]
    assert records[1]["usage"] is None
    assert records[2]["usage"] == usages[2]

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["usage"] == {
        "records_with_usage": 2,
        "prompt_tokens": 40,
        "completion_tokens": 60,
        "reasoning_tokens": 5,
        "total_cost": 0.01,
        "cost_source": "provider",
    }
    assert results["cost_per_correct"] == 0.01  # 1 correct item / 3 records


def test_resume_only_infers_missing_items(tmp_path, items):
    run_dir = tmp_path / "run-resume"
    cfg = _config()

    first = StubBackend([STUB_OUTPUT_CORRECT_H, STUB_OUTPUT_NO_PHRASE])
    run_eval(cfg, run_dir, limit=2, dataset_items=items, backend=first)
    assert len(_read_predictions(run_dir)) == 2

    second = StubBackend(["다시 풀어보니 정답은 B입니다."])
    run_eval(cfg, run_dir, resume=True, dataset_items=items, backend=second)

    # Only the one missing item was inferred by the second backend.
    assert second.calls == 1
    records = _read_predictions(run_dir)
    assert len(records) == 3
    ids = [r["id"] for r in records]
    assert sorted(ids) == sorted([it["id"] for it in items])
    assert len(set(ids)) == 3  # no duplicate ids
    assert {r["id"]: r for r in records}["test-id-0003"]["predicted_answer"] == "B"


def test_rescore_idempotent_on_run_dir(tmp_path, items):
    run_dir = tmp_path / "run-1"
    run_eval(
        _config(),
        run_dir,
        dataset_items=items,
        backend=StubBackend(
            [STUB_OUTPUT_CORRECT_H, STUB_OUTPUT_NO_PHRASE, STUB_OUTPUT_MULTI]
        ),
    )
    score_run(run_dir)
    first = (run_dir / "results.json").read_bytes()
    score_run(run_dir)
    assert (run_dir / "results.json").read_bytes() == first


class CrashAfterFirstBackend(Backend):
    """Simulates a crash/Ctrl-C: one good response, then KeyboardInterrupt.

    KeyboardInterrupt (BaseException) propagates out of run_eval instead of
    being swallowed into an error record — exactly like a real crash.
    """

    def __init__(self):
        self.calls = 0

    def generate(self, messages_list, config=None):
        out = []
        for _ in messages_list:
            self.calls += 1
            if self.calls == 1:
                out.append(GenerationResult(text=STUB_OUTPUT_CORRECT_H))
            else:
                raise KeyboardInterrupt("simulated crash")
        return out


class FailSecondItemBackend(Backend):
    """Item 1 succeeds, item 2 raises a plain (non-auth) error, item 3 succeeds."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages_list, config=None):
        out = []
        for _ in messages_list:
            self.calls += 1
            if self.calls == 1:
                out.append(GenerationResult(text=STUB_OUTPUT_CORRECT_H))
            elif self.calls == 2:
                raise RuntimeError("transient boom")
            else:
                out.append(GenerationResult(text=STUB_OUTPUT_MULTI))
        return out


def test_crash_mid_run_keeps_completed_predictions(tmp_path, items):
    """REQ-007: completed records are on disk even when the run raises."""
    import pytest

    run_dir = tmp_path / "run-crash"
    with pytest.raises(KeyboardInterrupt):
        run_eval(_config(), run_dir, dataset_items=items, backend=CrashAfterFirstBackend())
    records = _read_predictions(run_dir)
    assert len(records) == 1
    assert records[0]["id"] == items[0]["id"]
    assert records[0]["predicted_answer"] == "H"
    assert not (run_dir / "results.json").exists()  # crash -> no scoring


def test_crash_partial_file_is_resumable(tmp_path, items):
    """A crashed run's partial file is picked up by --resume with no dup ids."""
    import pytest

    run_dir = tmp_path / "run-crash-resume"
    with pytest.raises(KeyboardInterrupt):
        run_eval(_config(), run_dir, dataset_items=items, backend=CrashAfterFirstBackend())

    second = StubBackend([STUB_OUTPUT_NO_PHRASE, STUB_OUTPUT_MULTI])
    run_eval(_config(), run_dir, resume=True, dataset_items=items, backend=second)
    assert second.calls == 2  # only the 2 missing items re-inferred
    records = _read_predictions(run_dir)
    assert len(records) == 3
    ids = [r["id"] for r in records]
    assert len(set(ids)) == 3  # no duplicate ids
    assert (run_dir / "results.json").exists()


def test_per_item_error_records_written_incrementally(tmp_path, items):
    """Non-auth per-item failures still become error records, one line each."""
    run_dir = tmp_path / "run-err"
    run_eval(_config(), run_dir, dataset_items=items, backend=FailSecondItemBackend())
    records = _read_predictions(run_dir)
    assert len(records) == 3
    by_id = {r["id"]: r for r in records}
    assert by_id[items[0]["id"]]["predicted_answer"] == "H"
    assert by_id[items[1]["id"]]["predicted_answer"] is None
    assert by_id[items[1]["id"]]["usage"] is None
    assert "error" in by_id[items[1]["id"]]
    assert by_id[items[2]["id"]]["predicted_answer"] == "A, C"
