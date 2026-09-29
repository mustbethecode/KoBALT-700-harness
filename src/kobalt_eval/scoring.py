"""Score a run directory -> results.json.

Reads predictions.jsonl + config snapshot; computes overall accuracy plus
accuracy grouped by Class, Subclass, Level. Runnable repeatedly with
identical output and zero model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _group_stats(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for rec in records:
        k = str(rec.get(key))
        g = groups.setdefault(k, {"n": 0, "correct": 0})
        g["n"] += 1
        if rec.get("correct") is True:
            g["correct"] += 1
    for g in groups.values():
        g["accuracy"] = (g["correct"] / g["n"]) if g["n"] else 0.0
    return groups


def _usage_summary(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate per-record usage; None when no record carries usage.

    Records without usage (error records, older prediction files) are
    ignored entirely, so a partially-instrumented run still reports honest
    totals for the records that do have them.
    """
    usages = [r.get("usage") for r in records if isinstance(r.get("usage"), dict)]
    if not usages:
        return None

    def _total(key: str) -> int | None:
        vals = [
            u[key]
            for u in usages
            if isinstance(u.get(key), int) and not isinstance(u.get(key), bool)
        ]
        return sum(vals) if vals else None

    costs = [
        u["cost"]
        for u in usages
        if isinstance(u.get("cost"), (int, float)) and not isinstance(u.get("cost"), bool)
    ]
    sources = {
        u.get("cost_source")
        for u in usages
        if isinstance(u.get("cost"), (int, float))
    }
    if not costs:
        cost_source = None
    elif sources == {"provider"}:
        cost_source = "provider"
    else:
        cost_source = "mixed"
    return {
        "records_with_usage": len(usages),
        "prompt_tokens": _total("prompt_tokens"),
        "completion_tokens": _total("completion_tokens"),
        "reasoning_tokens": _total("reasoning_tokens"),
        "total_cost": sum(costs) if costs else None,
        "cost_source": cost_source,
    }


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute accuracy aggregates over prediction records."""
    n = len(records)
    n_correct = sum(1 for r in records if r.get("correct") is True)
    usage = _usage_summary(records)
    total_cost = usage.get("total_cost") if usage else None
    cost_per_correct = (
        total_cost / n_correct
        if isinstance(total_cost, (int, float)) and n_correct
        else None
    )
    return {
        "num_items": n,
        "num_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "by_class": _group_stats(records, "class"),
        "by_subclass": _group_stats(records, "subclass"),
        "by_level": _group_stats(records, "level"),
        "usage": usage,
        "cost_per_correct": cost_per_correct,
    }


def read_predictions(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read all prediction records from a run directory."""
    path = Path(run_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No predictions.jsonl in {run_dir}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_snapshot(run_dir: str | Path) -> dict[str, Any]:
    """Read the config snapshot dict (YAML preferred, JSON-tolerant)."""
    path = Path(run_dir) / "config.snapshot.yaml"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except ImportError:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def score_run(run_dir: str | Path) -> dict[str, Any]:
    """Score a run directory and write results.json (idempotent).

    Returns the results dict.
    """
    run_path = Path(run_dir)
    records = read_predictions(run_path)
    metrics = compute_metrics(records)
    snapshot = read_snapshot(run_path)
    backend = snapshot.get("backend", {}) if isinstance(snapshot, dict) else {}
    dataset = snapshot.get("dataset", {}) if isinstance(snapshot, dict) else {}
    results: dict[str, Any] = {
        "model": backend.get("model", records[0].get("model") if records else None),
        "backend": backend.get("family", records[0].get("backend") if records else None),
        "dataset": dataset if isinstance(dataset, dict) else {},
        "dataset_revision": (dataset.get("revision") if isinstance(dataset, dict) else None),
        "prompt_template_hash": snapshot.get("prompt_template_hash") if isinstance(snapshot, dict) else None,
        **metrics,
    }
    out = run_path / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
