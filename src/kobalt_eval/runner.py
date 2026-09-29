"""Run orchestration + resumability."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kobalt_eval.backends.base import AuthenticationFailed
from kobalt_eval.config import RunConfig, write_config_snapshot
from kobalt_eval.dataset import get_dataset_revision, load_dataset_items
from kobalt_eval.extraction import extract_answer
from kobalt_eval.prompts import build_messages_from_config

PREDICTIONS_FILENAME = "predictions.jsonl"
SNAPSHOT_FILENAME = "config.snapshot.yaml"
RESULTS_FILENAME = "results.json"
LOG_FILENAME = "run.log"


def default_run_dir_name(model: str) -> str:
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in (model or "model").split("/")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{stamp}-{slug or 'model'}"


def load_existing_predictions(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load existing predictions.jsonl keyed by item id (for resume)."""
    path = Path(run_dir) / PREDICTIONS_FILENAME
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "id" in rec:
                existing[str(rec["id"])] = rec
    return existing


def _setup_run_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"kobalt_eval.run.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fh = logging.FileHandler(run_dir / LOG_FILENAME, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def _require_api_key_for_api_backend(config: RunConfig, backend) -> None:
    """Fail fast when a real API-family backend has no key configured.

    Applies only to concrete API backend instances (OpenAI-compatible /
    Anthropic); injected doubles (e.g. test stubs) are untouched. Genuine
    per-item inference failures are still recorded as error records, not
    raised — only the missing-key pre-flight is fail-fast.
    """
    import os

    from kobalt_eval.backends.api_anthropic import AnthropicBackend
    from kobalt_eval.backends.api_openai import OpenAICompatibleBackend

    if isinstance(backend, (OpenAICompatibleBackend, AnthropicBackend)):
        env_var = config.backend.api_key_env or "OPENAI_API_KEY"
        if not os.environ.get(env_var):
            raise RuntimeError(
                f"{env_var} environment variable is not set — configure it "
                "before running an api-family backend"
            )


def run_eval(
    config: RunConfig,
    run_dir: str | Path,
    limit: int | None = None,
    resume: bool = False,
    backend=None,
    dataset_items: list[dict[str, Any]] | None = None,
) -> Path:
    """Execute a full evaluation run.

    Args:
        config: Resolved run config.
        run_dir: Output directory (created if needed).
        limit: Optional subset size (first N dataset items).
        resume: If True, skip items whose id is already in predictions.jsonl.
        backend: Optional pre-built backend (defaults to factory from config).
        dataset_items: Optional pre-loaded items (for tests / offline use).

    Returns:
        Path to the run directory.
    """
    from kobalt_eval.backends import create_backend

    if backend is None:
        backend = create_backend(config)
    # Fail fast on a missing API key before any artifacts are written, so a
    # missing key can never surface as exit-0 garbage records (the sequential
    # path records per-item inference errors instead of raising).
    _require_api_key_for_api_backend(config, backend)

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    logger = _setup_run_logger(run_path)

    if dataset_items is None:
        items = load_dataset_items(limit=limit)
        dataset_revision = get_dataset_revision()
    else:
        items = list(dataset_items[:limit] if limit is not None else dataset_items)
        dataset_revision = None

    # Snapshot + log header at start (fresh or resume).
    write_config_snapshot(config, run_path, dataset_revision=dataset_revision)
    logger.info("model=%s backend=%s items=%d resume=%s", config.backend.model, config.backend.family, len(items), resume)

    existing = load_existing_predictions(run_path) if resume else {}
    if resume and existing:
        logger.info("resume: found %d existing predictions", len(existing))

    mode = "a" if resume and (run_path / PREDICTIONS_FILENAME).exists() else "w"
    family = (config.backend.family or "").lower()
    concurrency = config.backend.concurrency if family == "api" else 1

    def infer(item: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages_from_config(item["question"], config)
        t0 = time.perf_counter()
        result = backend.generate([messages], config)[0]
        latency_ms = int((time.perf_counter() - t0) * 1000)
        predicted = extract_answer(result.text, config.extraction.regex)
        return {
            "id": item["id"],
            "model": config.backend.model,
            "backend": family,
            "raw_output": result.text,
            "predicted_answer": predicted,
            "ground_truth": item["ground_truth"],
            "correct": predicted is not None and predicted == item["ground_truth"],
            "class": item["class"],
            "subclass": item["subclass"],
            "level": item["level"],
            "latency_ms": latency_ms,
            "usage": result.usage,
        }

    todo = [it for it in items if str(it["id"]) not in existing]
    logger.info("todo=%d skipped=%d", len(todo), len(items) - len(todo))

    n_written = 0
    predictions_path = run_path / PREDICTIONS_FILENAME
    try:
        # Incremental appends (REQ-007): every completed record — success or
        # per-item error — hits disk immediately with a flush, so a crash or
        # Ctrl-C mid-run never loses completed work and --resume can skip it.
        # Resume appends without duplicating ids: `todo` already excludes ids
        # present in the file, and mode is "a" whenever that file exists.
        with predictions_path.open(mode, encoding="utf-8") as f:
            def emit(rec: dict[str, Any]) -> None:
                nonlocal n_written
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1

            if concurrency > 1:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(infer, item) for item in todo]
                    for fut in as_completed(futures):
                        emit(fut.result())
            else:
                for i, item in enumerate(todo):
                    try:
                        rec = infer(item)
                    except AuthenticationFailed:
                        # Fatal: abort without an error record for this item.
                        raise
                    except Exception as e:
                        logger.exception("inference failed for item %s", item.get("id"))
                        rec = {
                            "id": item["id"],
                            "model": config.backend.model,
                            "backend": family,
                            "raw_output": "",
                            "predicted_answer": None,
                            "ground_truth": item["ground_truth"],
                            "correct": False,
                            "class": item["class"],
                            "subclass": item["subclass"],
                            "level": item["level"],
                            "latency_ms": 0,
                            "usage": None,
                            "error": str(e),
                        }
                    emit(rec)
                    if (i + 1) % 10 == 0:
                        logger.info("progress %d/%d", i + 1, len(todo))
    except AuthenticationFailed as e:
        # Abort immediately: completed records are already on disk, skip
        # scoring, and re-raise so the CLI exits non-zero.
        logger.error("aborted: authentication failed — %s", e)
        for h in logger.handlers:
            h.close()
        logger.handlers.clear()
        raise
    logger.info("wrote %d predictions to %s", n_written, PREDICTIONS_FILENAME)

    # Score the run (no model calls) so the dir is complete.
    from kobalt_eval.scoring import score_run

    try:
        results = score_run(run_path)
        logger.info("accuracy=%.4f n=%d", results.get("accuracy", 0.0), results.get("num_items", 0))
    except Exception:
        logger.exception("scoring failed")

    for h in logger.handlers:
        h.close()
    logger.handlers.clear()
    return run_path
