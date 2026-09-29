"""Focused tests for the deterministic site data-generation lane (site/build.py).

All fixtures are temporary synthetic run directories; the real ``runs/``
tree is never touched. ``site/build.py`` is stdlib-only, so it is loaded
via importlib (``import site`` would resolve to the stdlib module).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD_PY = ROOT / "site" / "build.py"


def _load_build():
    spec = importlib.util.spec_from_file_location("kobalt_site_build", BUILD_PY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_build()
BuildError = build.BuildError

_MISSING = object()

SNAPSHOT_TEMPLATE = """\
backend:
  family: api
  model: {model}
  endpoint: {endpoint}
{engine_block}  max_retries: 3
  concurrency: 1
  api_key_env: OPENAI_API_KEY
  api_provider: openai
generation:
  do_sample: false
  max_new_tokens: {max_tokens}
prompt:
  system: test system prompt
  user_template: 'line one
    line two {{{{question}}}}'
extraction:
  regex: test-regex
prompt_template_hash: {prompt_hash}
dataset:
  name: {dataset_name}
  revision: {dataset_revision}
"""


def _engine_block(effort=_MISSING, style="nested", thinking=_MISSING):
    if effort is _MISSING and thinking is _MISSING:
        return "  engine_opts:\n    extra_body: null\n"
    body = ""
    if effort is not _MISSING:
        if effort is None:
            body += "      reasoning:\n        effort: null\n"
        elif style == "underscore":
            body += f"      reasoning_effort: {effort}\n"
        else:
            body += f"      reasoning:\n        effort: {effort}\n"
    if thinking is not _MISSING:
        if thinking is None:
            rendered = "null"
        elif isinstance(thinking, bool):
            rendered = "true" if thinking else "false"
        else:
            rendered = thinking
        body += f"      chat_template_kwargs:\n        enable_thinking: {rendered}\n"
    return "  engine_opts:\n    extra_body:\n" + body


def make_record(i, *, pred: Any = "A", truth: Any = "A", cls: Any = "Syntax",
                sub: Any = "Agreement", level: Any = 1, latency: Any = 100,
                raw: Any = "test raw output"):
    correct = pred is not None and pred == truth
    return {
        "id": f"syn-{i:04d}",
        "model": "testorg/test-model",
        "backend": "api",
        "raw_output": raw,
        "predicted_answer": pred,
        "ground_truth": truth,
        "correct": correct,
        "class": cls,
        "subclass": sub,
        "level": level,
        "latency_ms": latency,
    }


def results_for(records):
    n = len(records)
    nc = sum(1 for r in records if r.get("correct") is True)
    by_class: dict[str, dict] = {}
    by_level: dict[str, dict] = {}
    for r in records:
        for table, key in ((by_class, "class"), (by_level, "level")):
            name = str(r[key])
            g = table.setdefault(name, {"n": 0, "correct": 0})
            g["n"] += 1
            if r.get("correct") is True:
                g["correct"] += 1
    for table in (by_class, by_level):
        for g in table.values():
            g["accuracy"] = g["correct"] / g["n"]
    return {
        "model": "testorg/test-model",
        "backend": "api",
        "dataset": {"name": "test/Dataset", "revision": "rev-1"},
        "dataset_revision": "rev-1",
        "prompt_template_hash": "hash-1",
        "num_items": n,
        "num_correct": nc,
        "accuracy": (nc / n) if n else 0.0,
        "by_class": by_class,
        "by_level": by_level,
    }


def write_synthetic_run(parent, slug, records, *, model="testorg/test-model",
                        effort=_MISSING, effort_style="nested", thinking=_MISSING,
                        no_engine_opts=False,
                        results_override=None, snapshot_extra_lines=None):
    rundir = Path(parent) / slug
    rundir.mkdir(parents=True, exist_ok=True)
    if no_engine_opts:
        engine_block = ""
    else:
        engine_block = _engine_block(effort, style=effort_style, thinking=thinking)
    snapshot = SNAPSHOT_TEMPLATE.format(
        model=model,
        endpoint="https://example.test/v1",
        engine_block=engine_block,
        max_tokens=2048,
        prompt_hash="hash-1",
        dataset_name="test/Dataset",
        dataset_revision="rev-1",
    )
    if snapshot_extra_lines:
        snapshot += snapshot_extra_lines
    (rundir / "config.snapshot.yaml").write_text(snapshot, encoding="utf-8")
    with (rundir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    payload = results_override if results_override is not None else results_for(records)
    (rundir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rundir


def three_records():
    return [
        make_record(1, pred="A", truth="A", latency=100),
        make_record(2, pred="B", truth="C", latency=200),
        make_record(3, pred="A", truth="A", latency=300),
    ]


# ---------------------------------------------------------------------------
# build.py hygiene
# ---------------------------------------------------------------------------

def test_stdlib_only():
    tree = ast.parse(BUILD_PY.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert imported <= stdlib, f"non-stdlib imports: {sorted(imported - stdlib)}"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_discover_completed_and_skip_incomplete(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_synthetic_run(runs, "run-20260919-010203-complete", three_records())
    incomplete = runs / "run-20260919-010204-noresults"
    incomplete.mkdir()
    (incomplete / "config.snapshot.yaml").write_text("backend:\n  model: x\n", encoding="utf-8")
    (incomplete / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    nopred = runs / "run-20260919-010205-nopred"
    nopred.mkdir()
    (nopred / "config.snapshot.yaml").write_text("backend:\n  model: x\n", encoding="utf-8")
    (nopred / "results.json").write_text("{}", encoding="utf-8")
    (runs / "stray-file.txt").write_text("not a dir", encoding="utf-8")

    completed, skipped = build.discover_runs(runs)
    assert [p.name for p in completed] == ["run-20260919-010203-complete"]
    assert sorted(s["slug"] for s in skipped) == [
        "run-20260919-010204-noresults",
        "run-20260919-010205-nopred",
    ]
    assert all("missing" in s["reason"] for s in skipped)


def test_empty_runs_dir_builds_empty_payload(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    payload = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    assert payload["runs"] == []
    assert payload["item_count"] is None
    assert payload["dataset_groups"] is None


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_wilson_interval_properties_and_pins():
    lo, hi = build.wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(3.8416 / 13.8416, abs=1e-6)
    lo, hi = build.wilson_interval(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(10 / 13.8416, abs=1e-6)
    assert build.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = build.wilson_interval(525, 700)
    assert lo == pytest.approx(0.7166, abs=5e-4)
    assert hi == pytest.approx(0.7810, abs=5e-4)
    assert lo <= 525 / 700 <= hi


def test_invalid_empty_latency_derivations(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    records = [
        make_record(1, pred="A", truth="A", latency=10, raw="has output"),
        make_record(2, pred=None, truth="B", latency=20, raw=""),
        make_record(3, pred="AB", truth="A", latency=30, raw=None),
        make_record(4, pred="a", truth="a", latency=40),  # lowercase: invalid
        make_record(5, pred="", truth="C", latency=50, raw="   "),
        make_record(6, pred="K", truth="K", latency=60, raw="other"),
    ]
    del records[3]["raw_output"]  # missing key also counts as empty
    rundir = write_synthetic_run(runs, "run-20260919-010203-mixed", records)
    summary = build.summarize_run(rundir)
    assert summary["invalid"] == {"count": 5, "rate": pytest.approx(5 / 6)}
    assert summary["empty_raw_output"] == 4
    assert summary["latency_ms"]["n"] == 6
    assert summary["latency_ms"]["median"] == pytest.approx(35.0)
    assert summary["latency_ms"]["p95"] == pytest.approx(60.0)


def test_latency_median_odd_and_p95_nearest_rank(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    records = [make_record(i, latency=lat) for i, lat in enumerate([50, 10, 30, 20, 40])]
    rundir = write_synthetic_run(runs, "run-20260919-010203-lat", records)
    summary = build.summarize_run(rundir)
    assert summary["latency_ms"]["median"] == pytest.approx(30.0)
    # nearest-rank p95 of 5 values -> ceil(0.95*5)=5th smallest = max
    assert summary["latency_ms"]["p95"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# reasoning / ablation classification
# ---------------------------------------------------------------------------

def test_ablation_classification(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    d_none = write_synthetic_run(runs, "run-20260919-010203-none", recs, effort="none")
    d_min = write_synthetic_run(runs, "run-20260919-010204-min", recs, effort="minimal")
    d_null = write_synthetic_run(runs, "run-20260919-010205-null", recs)
    d_noeng = write_synthetic_run(runs, "run-20260919-010206-noeng", recs, no_engine_opts=True)

    s_none = build.summarize_run(d_none)
    assert s_none["category"] == "ablation"
    assert s_none["reasoning_mode"] == "none"
    assert s_none["reasoning_effort"] == "none"

    s_min = build.summarize_run(d_min)
    assert s_min["category"] == "primary"
    assert s_min["reasoning_mode"] == "minimal"
    assert "minimal" in s_min["label"]

    s_null = build.summarize_run(d_null)
    assert s_null["category"] == "primary"
    assert s_null["reasoning_mode"] == "provider-default"
    assert s_null["reasoning_effort"] is None

    s_noeng = build.summarize_run(d_noeng)
    assert s_noeng["category"] == "primary"
    assert s_noeng["reasoning_mode"] == "provider-default"


def test_reasoning_effort_underscore_form(tmp_path):
    assert build.find_reasoning_effort({"extra_body": {"reasoning_effort": "low"}}) == "low"
    assert build.find_reasoning_effort({"extra_body": {"reasoning_effort": None}}) is None
    assert build.find_reasoning_effort({"extra_body": {"reasoning": {"effort": "minimal"}}}) == "minimal"
    assert build.find_reasoning_effort(None) is None
    assert build.find_reasoning_effort({}) is None
    # agreeing duplicates across both forms are fine
    assert build.find_reasoning_effort(
        {"extra_body": {"reasoning": {"effort": "low"}, "reasoning_effort": "low"}}
    ) == "low"

    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    d_low = write_synthetic_run(
        runs, "run-20260919-010203-low", recs, effort="low", effort_style="underscore"
    )
    s_low = build.summarize_run(d_low)
    assert s_low["category"] == "primary"
    assert s_low["reasoning_mode"] == "low"
    assert s_low["reasoning_effort"] == "low"
    assert "low" in s_low["label"]

    d_none = write_synthetic_run(
        runs, "run-20260919-010204-none", recs, effort="none", effort_style="underscore"
    )
    s_none = build.summarize_run(d_none)
    assert s_none["category"] == "ablation"
    assert s_none["reasoning_mode"] == "none"


def test_reasoning_effort_conflict_fails(tmp_path):
    with pytest.raises(BuildError, match="conflicting reasoning effort"):
        build.find_reasoning_effort(
            {"extra_body": {"reasoning": {"effort": "minimal"}, "reasoning_effort": "low"}},
            source="test-snapshot",
        )

    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    rundir = write_synthetic_run(
        runs, "run-20260919-010203-conflict", recs, effort="minimal"
    )
    snapshot_path = rundir / "config.snapshot.yaml"
    text = snapshot_path.read_text(encoding="utf-8")
    text = text.replace(
        "      reasoning:\n        effort: minimal\n",
        "      reasoning:\n        effort: minimal\n      reasoning_effort: low\n",
    )
    snapshot_path.write_text(text, encoding="utf-8")
    with pytest.raises(BuildError, match="conflicting reasoning effort"):
        build.summarize_run(rundir)


def test_thinking_flag_disabled(tmp_path):
    assert build.find_thinking_enabled(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    ) is False
    assert build.find_thinking_enabled(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    ) is True
    assert build.find_thinking_enabled({"extra_body": None}) is None
    assert build.find_thinking_enabled({}) is None
    # non-boolean values are ignored (documented; preserved in extra_body)
    assert build.find_thinking_enabled(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": "false"}}}
    ) is None

    assert build.resolve_reasoning(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    ) == {"mode": "none", "effort": None, "source": "enable_thinking"}
    assert build.resolve_reasoning(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    ) == {"mode": "provider-default", "effort": None, "source": None}

    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    rundir = write_synthetic_run(
        runs, "run-20260919-010203-nothink", recs, thinking=False
    )
    summary = build.summarize_run(rundir)
    assert summary["reasoning_mode"] == "none"
    assert summary["reasoning_effort"] is None
    assert summary["reasoning_source"] == "enable_thinking"
    assert summary["category"] == "ablation"
    assert summary["label"] == "test-model (reasoning disabled)"
    assert summary["engine_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_thinking_true_and_absent_stay_default(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    d_true = write_synthetic_run(
        runs, "run-20260919-010203-true", recs, thinking=True
    )
    d_absent = write_synthetic_run(runs, "run-20260919-010204-absent", recs)
    for rundir in (d_true, d_absent):
        summary = build.summarize_run(rundir)
        assert summary["reasoning_mode"] == "provider-default"
        assert summary["reasoning_effort"] is None
        assert summary["reasoning_source"] is None
        assert summary["category"] == "primary"
        assert summary["label"] == "test-model"


def test_thinking_effort_conflicts(tmp_path):
    with pytest.raises(BuildError, match="conflicting reasoning signals"):
        build.resolve_reasoning(
            {"extra_body": {
                "reasoning": {"effort": "minimal"},
                "chat_template_kwargs": {"enable_thinking": False},
            }},
            source="test-snapshot",
        )
    with pytest.raises(BuildError, match="conflicting reasoning signals"):
        build.resolve_reasoning(
            {"extra_body": {
                "reasoning": {"effort": "none"},
                "chat_template_kwargs": {"enable_thinking": True},
            }},
            source="test-snapshot",
        )
    with pytest.raises(BuildError, match="conflicting 'enable_thinking'"):
        build.find_thinking_enabled(
            {"a": {"enable_thinking": True}, "b": {"enable_thinking": False}},
            source="test-snapshot",
        )
    # agreeing disable signals are fine; the effort form names the source
    assert build.resolve_reasoning(
        {"extra_body": {
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {"enable_thinking": False},
        }}
    ) == {"mode": "none", "effort": "none", "source": "reasoning.effort"}
    # effort provenance is reported for the primary forms too
    assert build.resolve_reasoning(
        {"extra_body": {"reasoning": {"effort": "minimal"}}}
    )["source"] == "reasoning.effort"
    assert build.resolve_reasoning(
        {"extra_body": {"reasoning_effort": "low"}}
    )["source"] == "reasoning_effort"

    runs = tmp_path / "runs"
    runs.mkdir()
    rundir = write_synthetic_run(
        runs, "run-20260919-010203-conflict", three_records(),
        effort="minimal", thinking=False,
    )
    with pytest.raises(BuildError, match="conflicting reasoning signals"):
        build.summarize_run(rundir)


def test_primary_runs_sort_before_ablations(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    low = [make_record(1, pred="A", truth="A"), make_record(2, pred="B", truth="C")]
    high = [make_record(1, pred="A", truth="A"), make_record(2, pred="B", truth="B")]
    write_synthetic_run(runs, "run-20260919-010203-low", low)
    write_synthetic_run(runs, "run-20260919-010204-high", high)
    write_synthetic_run(runs, "run-20260919-010205-abl", high, effort="none")
    payload = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    assert [r["slug"] for r in payload["runs"]] == [
        "run-20260919-010204-high",
        "run-20260919-010203-low",
        "run-20260919-010205-abl",
    ]


def test_timestamp_parsing():
    assert build.parse_run_timestamp("run-20260919-023453-foo") == "2026-09-19T02:34:53Z"
    assert build.parse_run_timestamp("not-a-run-dir") is None


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------

def test_duplicate_ids_fail(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    recs.append(dict(recs[0]))
    rundir = write_synthetic_run(runs, "run-20260919-010203-dup", recs)
    with pytest.raises(BuildError, match="duplicate id"):
        build.summarize_run(rundir)


def test_predictions_count_mismatch_fails(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    bad_results = results_for(recs)
    bad_results["num_items"] = 99
    rundir = write_synthetic_run(
        runs, "run-20260919-010203-mismatch", recs, results_override=bad_results
    )
    with pytest.raises(BuildError, match="mismatch"):
        build.summarize_run(rundir)


def test_inconsistent_item_counts_fail(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_synthetic_run(runs, "run-20260919-010203-three", three_records())
    write_synthetic_run(runs, "run-20260919-010204-four", three_records() + [make_record(9)])
    with pytest.raises(BuildError, match="inconsistent item counts"):
        build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")


def test_malformed_snapshot_fails(tmp_path):
    runs = tmp_path / "runs"
    rundir = runs / "run-20260919-010203-bad"
    rundir.mkdir(parents=True)
    (rundir / "config.snapshot.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    (rundir / "predictions.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")
    (rundir / "results.json").write_text('{"num_items": 1}', encoding="utf-8")
    with pytest.raises(BuildError, match="snapshot"):
        build.summarize_run(rundir)


# ---------------------------------------------------------------------------
# YAML subset: sequences
# ---------------------------------------------------------------------------

PROVIDER_LIST_SNAPSHOT = """\
backend:
  family: api
  model: google/gemma-4-31b-it
  endpoint: https://openrouter.ai/api/v1
  engine_opts:
    extra_body:
      provider:
        only:
        - deepinfra/turbo
  max_retries: 3
  concurrency: 1
  api_key_env: OPENAI_API_KEY
  api_provider: openai
generation:
  do_sample: false
  max_new_tokens: 2048
prompt:
  system: test system prompt
  user_template: 'line one
    line two'
extraction:
  regex: test-regex
prompt_template_hash: hash-1
dataset:
  name: test/Dataset
  revision: rev-1
"""


def test_nested_provider_only_list_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    rundir = runs / "run-20260920-072521-gemma-4-31b-it"
    rundir.mkdir(parents=True)
    recs = three_records()
    (rundir / "config.snapshot.yaml").write_text(PROVIDER_LIST_SNAPSHOT, encoding="utf-8")
    with (rundir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (rundir / "results.json").write_text(
        json.dumps(results_for(recs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = build.summarize_run(rundir)
    assert summary["model"] == "google/gemma-4-31b-it"
    assert summary["category"] == "primary"
    assert summary["reasoning_mode"] == "provider-default"
    assert summary["engine_extra_body"] == {"provider": {"only": ["deepinfra/turbo"]}}
    assert summary["timestamp"] == "2026-09-20T07:25:21Z"
    payload = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    assert [r["slug"] for r in payload["runs"]] == ["run-20260920-072521-gemma-4-31b-it"]
    assert payload["runs"][0]["engine_extra_body"] == {"provider": {"only": ["deepinfra/turbo"]}}


def test_yaml_sequences_preserve_scalars():
    doc = (
        "name: plain\n"
        "nothing: null\n"
        "flag: true\n"
        "count: 42\n"
        "ratio: 1.5\n"
        "quoted: 'a: b'\n"
        "items:\n"
        "- one\n"
        "- 2\n"
        "- null\n"
        "- false\n"
        "- 'quoted: scalar'\n"
        "nested:\n"
        "  deep:\n"
        "  - a\n"
        "  - b\n"
        "after: done\n"
    )
    assert build.parse_simple_yaml(doc) == {
        "name": "plain",
        "nothing": None,
        "flag": True,
        "count": 42,
        "ratio": 1.5,
        "quoted": "a: b",
        "items": ["one", 2, None, False, "quoted: scalar"],
        "nested": {"deep": ["a", "b"]},
        "after": "done",
    }


def test_yaml_sequence_in_sequence_and_map_in_sequence():
    doc = "matrix:\n- - a\n  - b\n- - c\npeople:\n- name: kim\n  level: 3\n- name: lee\n"
    assert build.parse_simple_yaml(doc) == {
        "matrix": [["a", "b"], ["c"]],
        "people": [{"name": "kim", "level": 3}, {"name": "lee"}],
    }


def test_yaml_single_quoted_folding():
    doc = "key: 'first second\n  continued here\n\n  new paragraph\n\n  '\n"
    assert build.parse_simple_yaml(doc) == {
        "key": "first second continued here\nnew paragraph\n"
    }
    assert build.parse_simple_yaml("a: 'x'\n") == {"a": "x"}
    assert build.parse_simple_yaml("a: 'it''s'\n") == {"a": "it's"}


def test_yaml_quoted_scalar_starting_on_next_line():
    doc = "prompt:\n  user_template:\n    'line one\n    line two'\n  other: 1\n"
    assert build.parse_simple_yaml(doc) == {
        "prompt": {"user_template": "line one line two", "other": 1}
    }


def test_yaml_malformed_list_mapping_mixtures():
    bad_docs = [
        # list item with no owning key inside a mapping block
        "a: 1\n- x\n",
        # indented block under a scalar value
        "a: 1\n  b: 2\n",
        # indented block under a scalar list item
        "key:\n- a\n  b: 2\n",
        # inconsistent list indentation
        "key:\n  - a\n   - b\n",
        # mapping entry at list indent after a sequence (stray key)
        "key:\n  - a\n  stray: 1\n",
        # duplicate keys
        "a: 1\na: 2\n",
        # duplicate keys inside a sequence mapping item
        "items:\n- name: x\n  name: y\n",
        # non-mapping line
        "just some text\n",
        # tab indentation
        "a:\n\t- x\n",
    ]
    for doc in bad_docs:
        with pytest.raises(BuildError, match="malformed snapshot"):
            build.parse_simple_yaml(doc)


# ---------------------------------------------------------------------------
# determinism + evidence outputs
# ---------------------------------------------------------------------------

def test_deterministic_output(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_synthetic_run(runs, "run-20260919-010203-b", three_records())
    write_synthetic_run(runs, "run-20260919-010204-a", three_records(), effort="minimal")
    first = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    second = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    assert first == second

    root = tmp_path / "root1"
    out1, _ = build.write_outputs(first, root, runs_dir=runs)
    text1 = out1.read_text(encoding="utf-8")
    root2 = tmp_path / "root2"
    out2, _ = build.write_outputs(second, root2, runs_dir=runs)
    assert out2.read_text(encoding="utf-8") == text1


def test_evidence_copies_only_safe_files(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    marker = "UNIQUE-MARKER-RAW-OUTPUT-12345"
    recs = [make_record(1, raw=marker), make_record(2, raw=marker)]
    rundir = write_synthetic_run(runs, "run-20260919-010203-evi", recs)
    (rundir / "run.log").write_text("secret log stuff", encoding="utf-8")
    payload = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    root = tmp_path / "root"
    _, data_dir = build.write_outputs(payload, root, runs_dir=runs)
    copied = sorted(p.relative_to(data_dir).as_posix() for p in data_dir.rglob("*") if p.is_file())
    assert copied == [
        "run-20260919-010203-evi/config.snapshot.yaml",
        "run-20260919-010203-evi/results.json",
    ]
    for rel in copied:
        src = (rundir / Path(rel).name).read_bytes()
        assert (data_dir / rel).read_bytes() == src
    blob = "".join((data_dir / rel).read_text(encoding="utf-8") for rel in copied)
    assert marker not in blob
    assert "secret log stuff" not in blob


# ---------------------------------------------------------------------------
# cost: measured usage + reported overlay
# ---------------------------------------------------------------------------

def _results_with_cost(records, *, total_cost=2.5):
    payload = results_for(records)
    nc = payload["num_correct"]
    payload["usage"] = {
        "records_with_usage": len(records),
        "prompt_tokens": 100 * len(records),
        "completion_tokens": 200 * len(records),
        "reasoning_tokens": None,
        "total_cost": total_cost,
        "cost_source": "provider",
    }
    payload["cost_per_correct"] = (total_cost / nc) if nc else None
    return payload


def test_cost_measured_from_results_usage(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()  # 2 correct of 3
    write_synthetic_run(
        runs, "run-20260919-010203-measured", recs,
        results_override=_results_with_cost(recs, total_cost=2.5),
    )
    payload = build.build_payload(runs, generated_at="2026-01-01T00:00:00Z")
    run = payload["runs"][0]
    assert run["cost"] == {
        "usd": pytest.approx(2.5),
        "per_correct": pytest.approx(1.25),
        "source": "measured",
    }
    assert run["usage"]["total_cost"] == 2.5


def test_cost_reported_overlay_used_when_measured_absent(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_synthetic_run(runs, "run-20260919-010203-reported", three_records())
    payload = build.build_payload(
        runs,
        generated_at="2026-01-01T00:00:00Z",
        cost_overlay={"run-20260919-010203-reported": 3.89},
    )
    run = payload["runs"][0]
    assert run["usage"] is None
    assert run["cost"] == {
        "usd": pytest.approx(3.89),
        "per_correct": pytest.approx(3.89 / 2),
        "source": "reported",
    }


def test_measured_cost_wins_over_overlay(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    recs = three_records()
    write_synthetic_run(
        runs, "run-20260919-010203-both", recs,
        results_override=_results_with_cost(recs, total_cost=1.0),
    )
    payload = build.build_payload(
        runs,
        generated_at="2026-01-01T00:00:00Z",
        cost_overlay={"run-20260919-010203-both": 99.0},
    )
    assert payload["runs"][0]["cost"]["source"] == "measured"
    assert payload["runs"][0]["cost"]["usd"] == pytest.approx(1.0)


def test_cost_null_when_measured_and_reported_absent(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_synthetic_run(runs, "run-20260919-010203-none", three_records())
    payload = build.build_payload(
        runs,
        generated_at="2026-01-01T00:00:00Z",
        cost_overlay={"some-other-run": 5.0},
    )
    run = payload["runs"][0]
    assert run["cost"] is None
    assert run["usage"] is None


def test_load_cost_overlay_skips_malformed_and_missing(tmp_path):
    path = tmp_path / "run_costs.json"
    path.write_text(
        json.dumps({
            "source": "operator-reported",
            "costs": {"ok": 1.5, "string": "nope", "negative": -3, "null": None, "bool": True},
        }),
        encoding="utf-8",
    )
    assert build.load_cost_overlay(path) == {"ok": 1.5}

    path.write_text("not json at all", encoding="utf-8")
    assert build.load_cost_overlay(path) == {}
    assert build.load_cost_overlay(tmp_path / "does-not-exist.json") == {}
