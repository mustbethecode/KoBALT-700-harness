# Plan: usage + cost tracking

Spec: `tasks/spec-cost-tracking.md`

## Approach

Three increments, each independently verifiable:

1. **Harness capture (vertical slice).** Make usage part of the backend
   contract, write it per item, aggregate it in scoring. This is the
   load-bearing change; everything else consumes it.
2. **Site data.** Merge measured usage and the reported-cost overlay into
   the build payload. Pure data change, testable offline.
3. **Site presentation.** Cost column, accuracy-vs-cost chart, archive fact,
   Method note. No new data, only rendering.

Ordering is dependency-driven: 2 needs 1's schema; 3 needs 2's payload.
Within increment 1, order is contract → API backends → local backends →
runner → scoring, each test-first.

## Risks / mitigations

- **Interface change ripples across test doubles.** All in-repo doubles are
  updated in the same increment; the full suite is the safety net (107 tests).
- **SDK response shapes differ from fakes.** Extraction is written
  defensively (attribute or dict access, `model_dump()` when present) so a
  2026 SDK change degrades to `usage: null` rather than a crash.
- **Adding keys breaks strict schema tests.** `PREDICTION_FIELDS` and site
  payload assertions are updated deliberately, not loosened.
- **Cost provenance confusion.** `cost_source` is mandatory whenever a cost
  is present; the site renders `reported` vs `measured` distinctly.

## Verification checkpoints

- After increment 1: `uv run pytest -m "not gpu" -q` green; a CLI-shaped stub
  run produces `usage` in `predictions.jsonl` and totals in `results.json`.
- After increment 2: site build tests green; payload shows measured/reported/
  null cases.
- After increment 3: rebuild the real site; cost appears for the three
  overlay runs and existing runs are otherwise unchanged in meaning.

## Files touched

| Increment | Files |
|---|---|
| 1 | `backends/base.py`, `backends/api_openai.py`, `backends/api_anthropic.py`, `backends/transformers_backend.py`, `backends/vllm_backend.py`, `runner.py`, `scoring.py`, `tests/helpers.py`, `tests/test_usage.py` (new), `tests/test_runner.py`, `tests/test_scoring.py`, `tests/test_auth_fatal.py`, `tests/test_local_engines.py` |
| 2 | `site/build.py`, `site/run_costs.json` (new), `tests/test_site_build.py` |
| 3 | `site/charts.js`, `site/results.html`, `site/method.html`, `site/runs.html` (if needed), `site/style.css` (if needed) |
