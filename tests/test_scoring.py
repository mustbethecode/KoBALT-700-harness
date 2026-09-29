"""Unit tests: scorer math against hand-computed fixtures + idempotency."""

from __future__ import annotations

import json

import pytest

from kobalt_eval.config import default_run_config, write_config_snapshot
from kobalt_eval.scoring import compute_metrics, score_run


def make_records():
    """5 records: 2 correct / 3 incorrect (one wrong letter, one null, one multi-letter)."""
    return [
        {
            "id": "r1", "model": "m", "backend": "api",
            "raw_output": "정답은 A입니다.", "predicted_answer": "A",
            "ground_truth": "A", "correct": True,
            "class": "Syntax", "subclass": "s1", "level": 1, "latency_ms": 5,
        },
        {
            "id": "r2", "model": "m", "backend": "api",
            "raw_output": "정답은 B입니다.", "predicted_answer": "B",
            "ground_truth": "A", "correct": False,
            "class": "Syntax", "subclass": "s1", "level": 1, "latency_ms": 5,
        },
        {
            "id": "r3", "model": "m", "backend": "api",
            "raw_output": "정답은 C입니다.", "predicted_answer": "C",
            "ground_truth": "C", "correct": True,
            "class": "Syntax", "subclass": "s2", "level": 2, "latency_ms": 5,
        },
        {
            "id": "r4", "model": "m", "backend": "api",
            "raw_output": "모르겠습니다.", "predicted_answer": None,
            "ground_truth": "D", "correct": False,
            "class": "Semantics", "subclass": "s3", "level": 2, "latency_ms": 5,
        },
        {
            "id": "r5", "model": "m", "backend": "api",
            "raw_output": "정답은 A입니다 ... 정답은 C입니다.",
            "predicted_answer": "A, C",
            "ground_truth": "E", "correct": False,
            "class": "Semantics", "subclass": "s3", "level": 3, "latency_ms": 5,
        },
    ]


def _rec(correct, cls="Syntax", sub="s1", level=1):
    return {"correct": correct, "class": cls, "subclass": sub, "level": level}


def test_overall_accuracy_math():
    m = compute_metrics(make_records())
    assert m["num_items"] == 5
    assert m["num_correct"] == 2
    assert m["accuracy"] == pytest.approx(2 / 5)


def test_by_class_math():
    m = compute_metrics(make_records())
    assert m["by_class"]["Syntax"] == {"n": 3, "correct": 2, "accuracy": pytest.approx(2 / 3)}
    assert m["by_class"]["Semantics"] == {"n": 2, "correct": 0, "accuracy": 0.0}


def test_by_subclass_math():
    m = compute_metrics(make_records())
    assert m["by_subclass"]["s1"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert m["by_subclass"]["s2"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert m["by_subclass"]["s3"] == {"n": 2, "correct": 0, "accuracy": 0.0}


def test_by_level_math():
    m = compute_metrics(make_records())
    assert m["by_level"]["1"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert m["by_level"]["2"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert m["by_level"]["3"] == {"n": 1, "correct": 0, "accuracy": 0.0}


def test_empty_records():
    m = compute_metrics([])
    assert m == {
        "num_items": 0,
        "num_correct": 0,
        "accuracy": 0.0,
        "by_class": {},
        "by_subclass": {},
        "by_level": {},
        "usage": None,
        "cost_per_correct": None,
    }


def test_usage_aggregates_tokens_and_cost_per_correct():
    records = [
        {
            "correct": True, "class": "Syntax", "subclass": "s1", "level": 1,
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 200,
                "reasoning_tokens": 50, "cost": 0.25, "cost_source": "provider",
            },
        },
        {
            "correct": False, "class": "Syntax", "subclass": "s1", "level": 1,
            "usage": {
                "prompt_tokens": 300, "completion_tokens": 400,
                "cost": 0.75, "cost_source": "provider",
            },
        },
    ]
    m = compute_metrics(records)
    assert m["usage"] == {
        "records_with_usage": 2,
        "prompt_tokens": 400,
        "completion_tokens": 600,
        "reasoning_tokens": 50,
        "total_cost": 1.0,
        "cost_source": "provider",
    }
    assert m["cost_per_correct"] == pytest.approx(1.0)  # 1 correct item


def test_usage_ignores_records_without_usage_and_null_cost():
    records = [
        {"correct": True, "class": "Syntax", "subclass": "s1", "level": 1, "usage": None},
        {
            "correct": False, "class": "Syntax", "subclass": "s1", "level": 1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        {"correct": True, "class": "Syntax", "subclass": "s1", "level": 1},
    ]
    m = compute_metrics(records)
    assert m["usage"] == {
        "records_with_usage": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": None,
        "total_cost": None,
        "cost_source": None,
    }
    assert m["cost_per_correct"] is None


def _write_run_dir(run_dir, records, with_snapshot=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if with_snapshot:
        cfg = default_run_config()
        cfg.backend.model = "m"
        write_config_snapshot(cfg, run_dir, dataset_revision="rev-1")


def test_score_run_writes_results_with_metadata(tmp_path):
    run_dir = tmp_path / "run-a"
    _write_run_dir(run_dir, make_records())
    results = score_run(run_dir)
    assert results["num_items"] == 5
    assert results["accuracy"] == pytest.approx(2 / 5)
    assert results["model"] == "m"
    assert results["backend"] == "api"
    assert results["dataset_revision"] == "rev-1"
    assert results["prompt_template_hash"]
    on_disk = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert on_disk == results


def test_rescore_idempotent_byte_identical(tmp_path):
    run_dir = tmp_path / "run-a"
    _write_run_dir(run_dir, make_records())
    score_run(run_dir)
    first = (run_dir / "results.json").read_bytes()
    score_run(run_dir)
    second = (run_dir / "results.json").read_bytes()
    assert first == second


def test_score_missing_predictions_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_run(tmp_path / "empty-dir")
