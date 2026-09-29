#!/usr/bin/env python3
"""Deterministic data-generation lane for the static KoBALT-700 benchmark site.

Reads completed run directories under ``runs/`` and emits a deterministic
``site/results.json`` plus publish-safe evidence copies under
``site/data/runs/<slug>/``.

Standard library only (no third-party imports). Runnable as::

    python3 site/build.py            # from the repo root (or any cwd)

Completion rule
---------------
A run directory counts as *completed* only when it contains all of::

    config.snapshot.yaml, predictions.jsonl, results.json

Directories missing any of these are *skipped* (listed under ``"skipped"``
in the payload) so incomplete/aborted runs never poison the site.

Malformed *completed* runs (bad JSON/YAML, duplicate prediction ids,
prediction count disagreeing with ``results.json``, inconsistent item
counts across runs, ...) fail the build loudly via :class:`BuildError`.

Per-run derivations
--------------------
Model id, display label, slug/path, timestamp (parsed from the ``run-``
dirname when possible), endpoint/provider, reasoning mode from
``engine_opts`` (``provider-default`` when absent/null, the explicit
effort string when present, via nested ``reasoning.effort`` or scalar
``reasoning_effort``), max tokens, prompt hash, dataset
name/revision, accuracy/correct/total, 95% Wilson interval, per-domain
and per-level tables, invalid-answer count/rate (``predicted_answer``
that is not exactly one of ``A``-``J``), empty raw-output count, and
median / nearest-rank p95 latency.

An explicit ``reasoning.effort: none`` (or ``reasoning_effort: none``),
or ``enable_thinking: false``, is classified as ``category ==
"ablation"`` with ``reasoning_mode == "none"`` so it stays separable
from primary results (the presentation tier groups these as
reasoning-disabled). Every other condition is preserved plainly
(``engine_extra_body``); no fairness claims are made here.

Only ``config.snapshot.yaml`` and ``results.json`` are copied into
``site/data/``. Raw predictions and logs are never published, and no
API keys or environment *values* are ever added (snapshots already
carry only the env-var *name*).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPLETION_FILES = ("config.snapshot.yaml", "predictions.jsonl", "results.json")
PUBLISH_SAFE_FILES = ("config.snapshot.yaml", "results.json")
VALID_CHOICES = frozenset("ABCDEFGHIJ")
WILSON_Z = 1.96  # 95% two-sided normal quantile

_RUN_DIR_RE = re.compile(
    r"^run-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-(.+)$"
)


class BuildError(Exception):
    """Fatal, clearly-attributed build failure for a malformed completed run."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Repo root derived from this file's location (robust to cwd)."""
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Minimal stdlib YAML-subset parser (for config.snapshot.yaml)
# ---------------------------------------------------------------------------

def _parse_scalar(text: str) -> Any:
    s = text.strip()
    if s in ("", "null", "Null", "NULL", "~"):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", s):
        try:
            return float(s)
        except ValueError:
            pass
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        return s[1:-1].replace("''", "'")
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s[1:-1]
    return s


def _consume_single_quoted(first: str, lines: list[str], idx: int) -> tuple[str, int]:
    """Consume a single-quoted scalar that starts in ``first`` (text after
    ``key:``) and may span following lines. Returns (raw_scalar, next_idx)."""
    buf = first
    i = idx
    while True:
        # Scan for closing quote, honouring '' escapes.
        j = 1  # skip opening quote
        closed = False
        while j < len(buf):
            if buf[j] == "'":
                if j + 1 < len(buf) and buf[j + 1] == "'":
                    j += 2
                    continue
                closed = j == len(buf) - 1 or buf[j + 1:].strip() == ""
                if closed:
                    break
                j += 1
            else:
                j += 1
        if closed:
            return buf, i
        if i >= len(lines):
            raise BuildError(f"malformed snapshot: unterminated quoted scalar: {first[:60]!r}")
        buf += "\n" + lines[i]
        i += 1


def _significant(lines: list[str], i: int) -> int | None:
    """Index of the next non-blank, non-comment line at/after ``i``."""
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            return i
        i += 1
    return None


def _indent_of(raw: str) -> int:
    """Space-indent width; tabs are rejected (they misparse silently)."""
    stripped_all = raw.lstrip()
    leading = raw[: len(raw) - len(stripped_all)]
    if "\t" in leading:
        raise BuildError(f"malformed snapshot: tab indentation not supported: {raw.strip()[:60]!r}")
    return len(raw) - len(raw.lstrip(" "))


def _is_dash_item(stripped: str) -> bool:
    return stripped == "-" or stripped.startswith("- ") or stripped.startswith("-\t")


def _split_key_value(stripped: str) -> tuple[str, str] | None:
    """Split ``key: value`` on the first colon followed by space/tab or EOL.

    Returns None when the line is not mapping-like, so values containing
    ':' (URLs, timestamps, ...) survive intact.
    """
    for pos, ch in enumerate(stripped):
        if ch == ":" and (pos + 1 == len(stripped) or stripped[pos + 1] in " \t"):
            return stripped[:pos].strip(), stripped[pos + 1 :].strip()
    return None


def _parse_quoted_value(rest: str, lines: list[str], i: int) -> tuple[Any, int]:
    buf, next_i = _consume_single_quoted(rest, lines, i)
    return _fold_single_quoted(buf), next_i


def _fold_single_quoted(buf: str) -> str:
    """Fold a single-quoted (possibly multi-line) scalar per YAML rules.

    ``buf`` starts with the opening quote and ends with the closing quote
    (plus trailing whitespace). Line breaks fold to a space; each blank
    line adds one newline; leading/trailing blank lines vanish. ``''``
    unescapes to ``'``.
    """
    text = buf.rstrip()
    assert text.endswith("'")  # guaranteed by _consume_single_quoted
    parts = text[1:-1].split("\n")
    if parts and parts[-1].strip() == "":
        parts.pop()  # closing-quote line carries no content
    chunks: list[str] = []
    cur: list[str] = []
    for line in parts:
        s = line.strip()
        if s == "":
            if cur:
                chunks.append(" ".join(cur))
                cur = []
                chunks.append("\n")
            elif chunks:
                chunks.append("\n")
        else:
            cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return "".join(chunks).replace("''", "'")


def _reject_nested_block(lines: list[str], i: int, parent_indent: int, context: str) -> None:
    """Fail when a deeper-indented block follows a scalar value."""
    nxt = _significant(lines, i)
    if nxt is not None and _indent_of(lines[nxt]) > parent_indent:
        raise BuildError(
            f"malformed snapshot: unexpected indented block after {context}: "
            f"{lines[nxt].strip()[:60]!r}"
        )


def _parse_nested_or_empty(lines: list[str], i: int, parent_indent: int) -> tuple[Any, int]:
    """Parse the nested block owned by an empty ``key:`` / ``-`` / ``- key:``.

    A ``-`` item at the same indent as the owning key is accepted (block
    sequences may sit at the parent key's level); anything else that is
    not deeper-indented means the key/item is an empty mapping.
    """
    nxt = _significant(lines, i)
    if nxt is None:
        return {}, i
    raw = lines[nxt]
    child_indent = _indent_of(raw)
    stripped = raw.strip()
    if _is_dash_item(stripped):
        if child_indent >= parent_indent:
            return _parse_sequence(lines, nxt, child_indent)
        return {}, i
    if child_indent > parent_indent:
        if stripped.startswith("'"):
            # Quoted scalar owned by the key but starting on the next line.
            value, i = _parse_quoted_value(stripped, lines, nxt + 1)
            _reject_nested_block(lines, i, child_indent, "quoted value")
            return value, i
        if stripped.startswith('"'):
            if not (len(stripped) >= 2 and stripped.endswith('"')):
                raise BuildError(
                    "malformed snapshot: unterminated quoted scalar: "
                    f"{stripped[:60]!r}"
                )
            value = _parse_scalar(stripped)
            _reject_nested_block(lines, i, child_indent, "quoted value")
            return value, i
        if _split_key_value(stripped) is not None:
            return _parse_mapping(lines, nxt, child_indent)
    return {}, i


def _parse_mapping(lines: list[str], i: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while True:
        nxt = _significant(lines, i)
        if nxt is None:
            return mapping, len(lines)
        raw = lines[nxt]
        line_indent = _indent_of(raw)
        if line_indent < indent:
            return mapping, nxt
        if line_indent > indent:
            raise BuildError(
                f"malformed snapshot: unexpected indentation: {raw.strip()[:60]!r}"
            )
        stripped = raw.strip()
        if _is_dash_item(stripped):
            raise BuildError(
                f"malformed snapshot: unexpected list item inside mapping: {stripped[:60]!r}"
            )
        kv = _split_key_value(stripped)
        if kv is None:
            raise BuildError(f"malformed snapshot: cannot parse line: {stripped[:80]!r}")
        key, rest = kv
        if not key:
            raise BuildError(f"malformed snapshot: empty key in line: {stripped[:80]!r}")
        if key in mapping:
            raise BuildError(f"malformed snapshot: duplicate key {key!r}")
        i = nxt + 1
        if rest.startswith("'"):
            value, i = _parse_quoted_value(rest, lines, i)
            _reject_nested_block(lines, i, indent, f"value of key {key!r}")
        elif rest != "":
            value = _parse_scalar(rest)
            _reject_nested_block(lines, i, indent, f"value of key {key!r}")
        else:
            value, i = _parse_nested_or_empty(lines, i, indent)
        mapping[key] = value
    # unreachable


def _parse_sequence(
    lines: list[str], i: int, indent: int, first_rest: str | None = None
) -> tuple[list[Any], int]:
    items: list[Any] = []
    if first_rest is not None:
        value, i = _parse_seq_item_value(first_rest, lines, i, indent)
        items.append(value)
    while True:
        nxt = _significant(lines, i)
        if nxt is None:
            return items, len(lines)
        raw = lines[nxt]
        line_indent = _indent_of(raw)
        if line_indent < indent:
            return items, nxt
        if line_indent > indent:
            raise BuildError(
                f"malformed snapshot: unexpected indentation in list: {raw.strip()[:60]!r}"
            )
        stripped = raw.strip()
        if not _is_dash_item(stripped):
            return items, nxt
        value, i = _parse_seq_item_value(stripped[1:].strip(), lines, nxt + 1, indent)
        items.append(value)
    # unreachable


def _parse_seq_item_value(
    rest: str, lines: list[str], i: int, indent: int
) -> tuple[Any, int]:
    """Parse one sequence item's content (text after ``-``)."""
    if rest == "":
        nxt = _significant(lines, i)
        if nxt is None or _indent_of(lines[nxt]) <= indent:
            return None, i
        child = lines[nxt].strip()
        if _is_dash_item(child):
            return _parse_sequence(lines, nxt, _indent_of(lines[nxt]))
        if child.startswith("'"):
            value, i = _parse_quoted_value(child, lines, nxt + 1)
            _reject_nested_block(lines, i, _indent_of(lines[nxt]), "list item")
            return value, i
        if child.startswith('"'):
            if not (len(child) >= 2 and child.endswith('"')):
                raise BuildError(
                    f"malformed snapshot: unterminated quoted scalar: {child[:60]!r}"
                )
            value = _parse_scalar(child)
            _reject_nested_block(lines, nxt + 1, _indent_of(lines[nxt]), "list item")
            return value, i
        if _split_key_value(child) is not None:
            return _parse_mapping(lines, nxt, _indent_of(lines[nxt]))
        raise BuildError(
            f"malformed snapshot: cannot parse list item block: {child[:60]!r}"
        )
    if rest.startswith("'"):
        value, i = _parse_quoted_value(rest, lines, i)
        _reject_nested_block(lines, i, indent, "list item")
        return value, i
    if rest.startswith('"'):
        value = _parse_scalar(rest)
        _reject_nested_block(lines, i, indent, "list item")
        return value, i
    if _is_dash_item(rest):
        # Nested sequence: '- - a' opens a sub-list whose items sit two
        # columns past the outer dash (the conventional layout).
        return _parse_sequence(lines, i, indent + 2, first_rest=rest[1:].strip())
    kv = _split_key_value(rest)
    if kv is not None and kv[0]:
        return _parse_seq_map_item(kv[0], kv[1], lines, i, indent)
    _reject_nested_block(lines, i, indent, "list item")
    return _parse_scalar(rest), i


def _parse_seq_map_item(
    key: str, rest: str, lines: list[str], i: int, dash_indent: int
) -> tuple[dict[str, Any], int]:
    """Parse a ``- key: value`` sequence item plus its deeper-indented
    continuation entries. Returns (sub_mapping, next_i)."""
    if not key:
        raise BuildError(f"malformed snapshot: empty key in list item: {key!r}")
    sub: dict[str, Any] = {}
    if rest.startswith("'"):
        first_value, i = _parse_quoted_value(rest, lines, i)
        # A quoted scalar item cannot own a mapping continuation.
        _reject_nested_block(lines, i, dash_indent, "list item value")
    elif rest != "":
        first_value = _parse_scalar(rest)
    else:
        first_value, i = _parse_nested_or_empty(lines, i, dash_indent)
    sub[key] = first_value
    nxt = _significant(lines, i)
    if nxt is not None and _indent_of(lines[nxt]) > dash_indent:
        stripped = lines[nxt].strip()
        if _is_dash_item(stripped):
            raise BuildError(
                "malformed snapshot: unexpected list item in mapping item: "
                f"{stripped[:60]!r}"
            )
        if _split_key_value(stripped) is None:
            raise BuildError(
                "malformed snapshot: unexpected indented content in mapping item: "
                f"{stripped[:60]!r}"
            )
        cont, i = _parse_mapping(lines, nxt, _indent_of(lines[nxt]))
        for cont_key in cont:
            if cont_key in sub:
                raise BuildError(f"malformed snapshot: duplicate key {cont_key!r}")
        sub.update(cont)
    return sub, i


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the indentation-based YAML subset written by ``yaml.safe_dump``.

    Supports nested mappings, sequences of scalars (including sequences
    nested under a mapping key at the same indent level, sequences
    nested in sequences, and mappings nested in sequences), null/bool/
    int/float scalars, single-quoted scalars (inline, multi-line with
    YAML line folding, or starting on the line after their key) and
    plain scalars. Malformed indentation and mapping/list mixtures raise
    :class:`BuildError` instead of misparsing.
    """
    lines = text.splitlines()
    first = _significant(lines, 0)
    if first is None:
        return {}
    raw = lines[first]
    indent = _indent_of(raw)
    if _is_dash_item(raw.strip()):
        raise BuildError("malformed snapshot: top-level mapping expected")
    if _split_key_value(raw.strip()) is None:
        raise BuildError(f"malformed snapshot: cannot parse line: {raw.strip()[:80]!r}")
    value, next_i = _parse_mapping(lines, first, indent)
    trailing = _significant(lines, next_i)
    if trailing is not None:
        raise BuildError(
            "malformed snapshot: unexpected trailing content: "
            f"{lines[trailing].strip()[:60]!r}"
        )
    return value


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a snapshot file (YAML subset; JSON-tolerant, stdlib only)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise BuildError(f"{p.parent.name}: cannot read config.snapshot.yaml ({e})")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise BuildError(f"{p.parent.name}: malformed config.snapshot.yaml ({e})")
        if not isinstance(data, dict):
            raise BuildError(f"{p.parent.name}: malformed config.snapshot.yaml (top-level mapping expected)")
        return data
    try:
        data = parse_simple_yaml(text)
    except BuildError:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise BuildError(f"{p.parent.name}: malformed config.snapshot.yaml ({e})")
    return data


# ---------------------------------------------------------------------------
# Small derivations
# ---------------------------------------------------------------------------

def _collect_effort_entries(engine_opts: Any) -> list[tuple[str, Any]]:
    """Collect ``(form, value)`` reasoning-effort candidates in deterministic order.

    Forms are ``"reasoning.effort"`` (nested mapping) and
    ``"reasoning_effort"`` (scalar key), searched depth-first with
    sorted keys anywhere inside ``engine_opts``.
    """
    entries: list[tuple[str, Any]] = []

    def _search(node: Any) -> None:
        if isinstance(node, dict):
            reasoning = node.get("reasoning")
            if isinstance(reasoning, dict) and "effort" in reasoning:
                entries.append(("reasoning.effort", reasoning["effort"]))
            if "reasoning_effort" in node:
                entries.append(("reasoning_effort", node["reasoning_effort"]))
            for key in sorted(node, key=str):
                _search(node[key])
        elif isinstance(node, list):
            for entry in node:
                _search(entry)

    _search(engine_opts if isinstance(engine_opts, dict) else {})
    return entries


def find_reasoning_effort(engine_opts: Any, source: str = "engine_opts") -> str | None:
    """Return the explicit reasoning effort string, or None when absent/null.

    Recognizes both the nested ``reasoning: {effort: X}`` mapping and the
    scalar ``reasoning_effort: X`` key, anywhere inside ``engine_opts``
    (normally under ``extra_body``). Search is depth-first in
    deterministic (sorted-key) order; null values count as absent.

    When disagreeing explicit values are present in either form, raises
    :class:`BuildError` naming ``source`` instead of silently picking one,
    since the choice can flip a run between primary and ablation.
    """
    entries = _collect_effort_entries(engine_opts)
    distinct = sorted({str(v) for _, v in entries if v is not None})
    if len(distinct) > 1:
        raise BuildError(
            f"conflicting reasoning effort values {distinct} in {source} "
            "(nested 'reasoning.effort' vs scalar 'reasoning_effort')"
        )
    return distinct[0] if distinct else None


def find_thinking_enabled(engine_opts: Any, source: str = "engine_opts") -> bool | None:
    """Return the explicit ``enable_thinking`` boolean, or None when absent.

    Searches for boolean ``enable_thinking`` values anywhere inside
    ``engine_opts`` (e.g. ``extra_body.chat_template_kwargs``), depth-first
    in deterministic order. Non-boolean values are ignored: only real
    booleans are the documented signal (everything stays preserved
    verbatim in ``engine_extra_body``). Disagreeing booleans raise
    :class:`BuildError` naming ``source``.
    """
    found: list[bool] = []

    def _search(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("enable_thinking"), bool):
                found.append(node["enable_thinking"])
            for key in sorted(node, key=str):
                _search(node[key])
        elif isinstance(node, list):
            for entry in node:
                _search(entry)

    _search(engine_opts if isinstance(engine_opts, dict) else {})
    distinct = sorted(set(found))
    if len(distinct) > 1:
        raise BuildError(
            f"conflicting 'enable_thinking' values {distinct} in {source}"
        )
    return distinct[0] if distinct else None


def resolve_reasoning(engine_opts: Any, source: str = "engine_opts") -> dict[str, Any]:
    """Resolve reasoning metadata to ``{"mode", "effort", "source"}``.

    - ``mode`` is ``"provider-default"``, an explicit effort string, or
      ``"none"`` when reasoning is explicitly disabled (effort ``none``
      or ``enable_thinking: false``).
    - ``effort`` is the explicit effort string or None (a thinking-flag
      disable leaves it None: no effort value was configured).
    - ``source`` names the winning signal: ``"reasoning.effort"``,
      ``"reasoning_effort"`` (nested form preferred when both agree),
      ``"enable_thinking"``, or None for provider-default.

    Conflict rules (rejection preferred: the presentation tier flips on
    this classification):
    - disagreeing effort values -> BuildError (via find_reasoning_effort);
    - disagreeing ``enable_thinking`` booleans -> BuildError;
    - explicit non-``none`` effort together with ``enable_thinking:
      false`` -> BuildError;
    - effort ``none`` together with ``enable_thinking: true`` ->
      BuildError.
    ``enable_thinking: true`` is otherwise default-compatible: it never
    disables, and with no effort value the mode stays
    ``provider-default``.
    """
    effort = find_reasoning_effort(engine_opts, source=source)
    thinking = find_thinking_enabled(engine_opts, source=source)
    if thinking is False:
        if effort is not None and effort != "none":
            raise BuildError(
                f"conflicting reasoning signals in {source}: effort {effort!r} "
                "vs 'enable_thinking: false'"
            )
        if effort == "none":
            entries = _collect_effort_entries(engine_opts)
            forms = sorted({form for form, v in entries if v is not None})
            return {"mode": "none", "effort": "none", "source": forms[0]}
        return {"mode": "none", "effort": None, "source": "enable_thinking"}
    if thinking is True and effort == "none":
        raise BuildError(
            f"conflicting reasoning signals in {source}: effort 'none' "
            "vs 'enable_thinking: true'"
        )
    if effort is None:
        return {"mode": "provider-default", "effort": None, "source": None}
    entries = _collect_effort_entries(engine_opts)
    forms = sorted({form for form, v in entries if v is not None})
    return {"mode": effort, "effort": effort, "source": forms[0]}


def short_model_label(model_id: Any) -> str:
    if not model_id:
        return ""
    return str(model_id).split("/")[-1]


def parse_run_timestamp(slug: str) -> str | None:
    """Infer an ISO-8601 UTC timestamp from a ``run-YYYYMMDD-HHMMSS-*`` slug."""
    m = _RUN_DIR_RE.match(slug)
    if not m:
        return None
    try:
        dt = datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def wilson_interval(num_correct: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 1.0)
    p = num_correct / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def median_of(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def nearest_rank_percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (e.g. p95): ceil(pct/100 * n)-th smallest."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(pct / 100.0 * len(ordered))
    rank = min(max(rank, 1), len(ordered))
    return float(ordered[rank - 1])


def is_valid_answer(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 1 and value in VALID_CHOICES


# ---------------------------------------------------------------------------
# Run discovery / validation
# ---------------------------------------------------------------------------

def discover_runs(runs_dir: str | Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Split run dirs into (completed, skipped).

    Completed = contains all of :data:`COMPLETION_FILES`. Anything else is
    skipped with a plain reason; discovery never fails.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise BuildError(f"runs directory not found: {runs_dir}")
    completed: list[Path] = []
    skipped: list[dict[str, str]] = []
    for child in sorted(runs_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        missing = [f for f in COMPLETION_FILES if not (child / f).is_file()]
        if missing:
            skipped.append({"slug": child.name, "reason": f"missing: {', '.join(missing)}"})
        else:
            completed.append(child)
    return completed, skipped


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise BuildError(f"{p.parent.name}: cannot read predictions.jsonl ({e})")
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise BuildError(f"{p.parent.name}: malformed predictions.jsonl line {lineno} ({e})")
        if not isinstance(rec, dict):
            raise BuildError(f"{p.parent.name}: malformed predictions.jsonl line {lineno} (mapping expected)")
        records.append(rec)
    return records


def load_results(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BuildError(f"{p.parent.name}: malformed results.json ({e})")
    except OSError as e:
        raise BuildError(f"{p.parent.name}: cannot read results.json ({e})")
    if not isinstance(data, dict):
        raise BuildError(f"{p.parent.name}: malformed results.json (top-level mapping expected)")
    return data


def load_cost_overlay(path: str | Path) -> dict[str, float]:
    """Operator-reported run costs (USD), keyed by run slug.

    Used for runs that predate usage capture. Missing file, unreadable file,
    or malformed entries are skipped rather than fatal — a cost overlay must
    never break the build.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    costs = data.get("costs")
    if not isinstance(costs, dict):
        return {}
    out: dict[str, float] = {}
    for slug, usd in costs.items():
        if (
            isinstance(slug, str)
            and isinstance(usd, (int, float))
            and not isinstance(usd, bool)
            and math.isfinite(float(usd))
            and float(usd) >= 0
        ):
            out[slug] = float(usd)
    return out


def _usage_block(results: dict[str, Any]) -> dict[str, Any] | None:
    """Validated aggregate usage from results.json (None when absent)."""
    usage = results.get("usage")
    if not isinstance(usage, dict):
        return None
    if not isinstance(usage.get("records_with_usage"), int):
        return None
    return usage


def _cost_block(
    slug: str,
    results: dict[str, Any],
    num_correct: int,
    cost_overlay: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Per-run cost: measured provider cost wins over the reported overlay."""
    usage = _usage_block(results)
    total_cost = usage.get("total_cost") if usage else None
    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
        if not math.isfinite(float(total_cost)) or float(total_cost) < 0:
            return None
        per_correct = results.get("cost_per_correct")
        if not isinstance(per_correct, (int, float)) or isinstance(per_correct, bool):
            per_correct = (float(total_cost) / num_correct) if num_correct else None
        return {
            "usd": float(total_cost),
            "per_correct": float(per_correct) if isinstance(per_correct, (int, float)) else None,
            "source": "measured",
        }
    reported = (cost_overlay or {}).get(slug)
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        return {
            "usd": float(reported),
            "per_correct": (float(reported) / num_correct) if num_correct else None,
            "source": "reported",
        }
    return None


def _normalize_group_table(raw: Any, slug: str, field: str) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        raise BuildError(f"{slug}: malformed results.json ({field!r} table missing or empty)")
    rows: list[dict[str, Any]] = []
    for name, stats in raw.items():
        if not isinstance(stats, dict):
            raise BuildError(f"{slug}: malformed results.json ({field}[{name!r}] not a mapping)")
        n = stats.get("n")
        correct = stats.get("correct")
        if not isinstance(n, int) or not isinstance(correct, int) or n < 0 or correct < 0 or correct > n:
            raise BuildError(f"{slug}: malformed results.json ({field}[{name!r}] has bad n/correct)")
        rows.append({"name": str(name), "n": n, "correct": correct,
                     "accuracy": (correct / n) if n else 0.0})
    return rows


def _sort_key_for_group(name: str) -> tuple[int, Any]:
    try:
        return (0, int(name))
    except ValueError:
        try:
            return (0, float(name))
        except ValueError:
            return (1, name)


def summarize_run(run_dir: str | Path, cost_overlay: dict[str, float] | None = None) -> dict[str, Any]:
    """Validate one completed run dir and derive its site record.

    ``cost_overlay`` supplies operator-reported USD costs for runs that
    predate usage capture; measured provider cost always wins.

    Raises :class:`BuildError` with a clearly-attributed message for any
    malformed completed run.
    """
    run_path = Path(run_dir)
    slug = run_path.name

    snapshot = load_snapshot(run_path / "config.snapshot.yaml")
    if not snapshot:
        raise BuildError(f"{slug}: malformed config.snapshot.yaml (empty)")
    results = load_results(run_path / "results.json")
    records = read_predictions(run_path / "predictions.jsonl")
    if not records:
        raise BuildError(f"{slug}: malformed predictions.jsonl (no records)")

    # --- prediction id validation ---------------------------------------
    seen: set[str] = set()
    for rec in records:
        rid = rec.get("id")
        if rid is None or (isinstance(rid, str) and rid == ""):
            raise BuildError(f"{slug}: malformed predictions.jsonl (record missing id)")
        key = str(rid)
        if key in seen:
            raise BuildError(f"{slug}: malformed predictions.jsonl (duplicate id {key!r})")
        seen.add(key)

    # --- headline counts --------------------------------------------------
    total = results.get("num_items")
    correct = results.get("num_correct")
    accuracy = results.get("accuracy")
    if not isinstance(total, int) or total <= 0:
        raise BuildError(f"{slug}: malformed results.json (bad num_items {total!r})")
    if not isinstance(correct, int) or correct < 0 or correct > total:
        raise BuildError(f"{slug}: malformed results.json (bad num_correct {correct!r})")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy):
        raise BuildError(f"{slug}: malformed results.json (bad accuracy {accuracy!r})")
    if len(records) != total:
        raise BuildError(
            f"{slug}: predictions/results mismatch ({len(records)} prediction records "
            f"vs num_items={total})"
        )
    if abs(float(accuracy) - correct / total) > 1e-6:
        raise BuildError(
            f"{slug}: malformed results.json (accuracy {accuracy!r} disagrees "
            f"with num_correct/num_items={correct}/{total})"
        )
    accuracy = correct / total

    # --- snapshot-derived conditions --------------------------------------
    backend = snapshot.get("backend")
    if backend is None:
        backend = {}
    if not isinstance(backend, dict):
        raise BuildError(f"{slug}: malformed config.snapshot.yaml (backend not a mapping)")
    generation = snapshot.get("generation")
    if generation is None:
        generation = {}
    if not isinstance(generation, dict):
        raise BuildError(f"{slug}: malformed config.snapshot.yaml (generation not a mapping)")
    dataset = snapshot.get("dataset")
    if dataset is None:
        dataset = {}
    if not isinstance(dataset, dict):
        raise BuildError(f"{slug}: malformed config.snapshot.yaml (dataset not a mapping)")

    model = backend.get("model") or results.get("model")
    if not model or not str(model).strip():
        raise BuildError(f"{slug}: malformed run (no model id in snapshot or results)")
    model = str(model)
    engine_opts = backend.get("engine_opts")
    resolved = resolve_reasoning(engine_opts, source=f"{slug}/config.snapshot.yaml")
    effort = resolved["effort"]
    reasoning_mode = resolved["mode"]
    category = "ablation" if reasoning_mode == "none" else "primary"
    short = short_model_label(model)
    if effort is not None:
        label = f"{short} (reasoning effort: {effort})"
    elif reasoning_mode == "none":
        label = f"{short} (reasoning disabled)"
    else:
        label = short

    extra_body = None
    if isinstance(engine_opts, dict):
        extra_body = engine_opts.get("extra_body")

    dataset_name = dataset.get("name")
    dataset_revision = dataset.get("revision")
    if dataset_revision is None:
        dataset_revision = results.get("dataset_revision")

    # --- groups ------------------------------------------------------------
    by_domain = _normalize_group_table(results.get("by_class"), slug, "by_class")
    by_domain.sort(key=lambda r: r["name"])
    by_level = _normalize_group_table(results.get("by_level"), slug, "by_level")
    by_level.sort(key=lambda r: _sort_key_for_group(r["name"]))

    # --- answer quality -----------------------------------------------------
    invalid = sum(1 for rec in records if not is_valid_answer(rec.get("predicted_answer")))
    def _is_empty_raw(rec: dict[str, Any]) -> bool:
        raw = rec.get("raw_output")
        return not (isinstance(raw, str) and raw.strip() != "")

    empty_raw = sum(1 for rec in records if _is_empty_raw(rec))

    # --- latency --------------------------------------------------------------
    latencies = [
        float(rec["latency_ms"]) for rec in records
        if isinstance(rec.get("latency_ms"), (int, float))
        and not isinstance(rec.get("latency_ms"), bool)
        and math.isfinite(float(rec["latency_ms"]))
    ]
    lo, hi = wilson_interval(correct, total)
    usage = _usage_block(results)
    cost = _cost_block(slug, results, correct, cost_overlay)

    return {
        "slug": slug,
        "path": f"runs/{slug}",
        "model": model,
        "label": label,
        "timestamp": parse_run_timestamp(slug),
        "backend": backend.get("family", results.get("backend")),
        "provider": backend.get("api_provider"),
        "endpoint": backend.get("endpoint"),
        "reasoning_mode": reasoning_mode,
        "reasoning_effort": effort,
        "reasoning_source": resolved["source"],
        "category": category,
        "engine_extra_body": extra_body,
        "max_new_tokens": generation.get("max_new_tokens"),
        "prompt_template_hash": snapshot.get("prompt_template_hash"),
        "dataset": {"name": dataset_name, "revision": dataset_revision},
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "wilson_95": {"lo": round(lo, 6), "hi": round(hi, 6)},
        "by_domain": by_domain,
        "by_level": by_level,
        "invalid": {"count": invalid, "rate": round(invalid / total, 6)},
        "empty_raw_output": empty_raw,
        "usage": usage,
        "cost": cost,
        "latency_ms": {
            "n": len(latencies),
            "median": median_of(latencies),
            "p95": nearest_rank_percentile(latencies, 95),
        },
        "evidence": {
            "snapshot": f"site/data/runs/{slug}/config.snapshot.yaml",
            "results": f"site/data/runs/{slug}/results.json",
        },
    }


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def build_payload(
    runs_dir: str | Path,
    generated_at: str | None = None,
    cost_overlay: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Discover, validate and summarize every completed run.

    Primary runs come first sorted by descending accuracy, then ablations
    (each group: descending accuracy, slug ascending for full determinism).
    """
    completed, skipped = discover_runs(runs_dir)
    summaries = [summarize_run(d, cost_overlay=cost_overlay) for d in completed]

    totals = {s["total"] for s in summaries}
    if len(totals) > 1:
        detail = ", ".join(f"{s['slug']}={s['total']}" for s in summaries)
        raise BuildError(f"inconsistent item counts across runs: {detail}")

    summaries.sort(key=lambda s: (0 if s["category"] == "primary" else 1, -s["accuracy"], s["slug"]))

    item_count = summaries[0]["total"] if summaries else None

    dataset_groups: dict[str, Any] | None = None
    dataset_meta: dict[str, Any] = {"name": None, "revision": None, "consistent": True}
    if summaries:
        names = {json.dumps(s["dataset"], sort_keys=True) for s in summaries}
        dataset_meta = {
            "name": summaries[0]["dataset"]["name"],
            "revision": summaries[0]["dataset"]["revision"],
            "consistent": len(names) == 1,
        }
        domain_sizes = [
            json.dumps([[r["name"], r["n"]] for r in s["by_domain"]], sort_keys=True)
            for s in summaries
        ]
        level_sizes = [
            json.dumps([[r["name"], r["n"]] for r in s["by_level"]], sort_keys=True)
            for s in summaries
        ]
        if len(set(domain_sizes)) == 1 and len(set(level_sizes)) == 1:
            dataset_groups = {
                "by_domain": [{"name": r["name"], "n": r["n"]} for r in summaries[0]["by_domain"]],
                "by_level": [{"name": r["name"], "n": r["n"]} for r in summaries[0]["by_level"]],
            }

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "generated_at": generated_at,
        "generator": "site/build.py",
        "item_count": item_count,
        "dataset": dataset_meta,
        "dataset_groups": dataset_groups,
        "runs": summaries,
        "skipped": sorted(skipped, key=lambda s: s["slug"]),
    }


def write_outputs(
    payload: dict[str, Any],
    root: str | Path,
    *,
    out_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write ``site/results.json`` and refresh publish-safe evidence copies.

    Only ``config.snapshot.yaml`` and ``results.json`` are copied per run;
    raw predictions and logs are never published.
    """
    root_path = Path(root)
    out = Path(out_path) if out_path else root_path / "site" / "results.json"
    data_runs = Path(data_dir) if data_dir else root_path / "site" / "data" / "runs"
    runs_path = Path(runs_dir) if runs_dir else root_path / "runs"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if data_runs.exists():
        shutil.rmtree(data_runs)
    for run in payload.get("runs", []):
        slug = run["slug"]
        dest = data_runs / slug
        dest.mkdir(parents=True, exist_ok=True)
        for fname in PUBLISH_SAFE_FILES:
            src = runs_path / slug / fname
            shutil.copyfile(src, dest / fname)
    return out, data_runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic site data for the KoBALT-700 benchmark.")
    parser.add_argument("--runs-dir", default=None, help="runs directory (default: <repo>/runs)")
    parser.add_argument("--out", default=None, help="results.json path (default: <repo>/site/results.json)")
    parser.add_argument("--data-dir", default=None, help="evidence dir (default: <repo>/site/data/runs)")
    parser.add_argument(
        "--costs",
        default=None,
        help="operator-reported run costs JSON (default: <repo>/site/run_costs.json)",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    runs_dir = Path(args.runs_dir) if args.runs_dir else root / "runs"
    costs_path = Path(args.costs) if args.costs else root / "site" / "run_costs.json"
    try:
        payload = build_payload(runs_dir, cost_overlay=load_cost_overlay(costs_path))
    except BuildError as e:
        print(f"site/build.py: error: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else root / "site" / "results.json"
    data_dir = Path(args.data_dir) if args.data_dir else root / "site" / "data" / "runs"
    write_outputs(payload, root, out_path=out_path, data_dir=data_dir, runs_dir=runs_dir)

    n_primary = sum(1 for r in payload["runs"] if r["category"] == "primary")
    n_ablation = len(payload["runs"]) - n_primary
    n_reported = sum(
        1 for r in payload["runs"]
        if isinstance(r.get("cost"), dict) and r["cost"].get("source") == "reported"
    )
    print(
        f"site/build.py: {len(payload['runs'])} runs "
        f"({n_primary} primary, {n_ablation} ablation, {n_reported} reported-cost), "
        f"{len(payload['skipped'])} skipped -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
