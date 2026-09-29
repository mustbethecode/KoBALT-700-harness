# Tasks: usage + cost tracking

- [x] 1. Harness contract: `GenerationResult` + usage normalization helper
  - Acceptance: `generate()` returns `GenerationResult`; helper yields the
    canonical usage object or `None`; all four backends extract usage from
    their real response shapes.
  - Verify: `uv run pytest -m "not gpu" -q` — done (127 passed; new
    `tests/test_usage.py` covers extraction, provenance, and the contract)
  - Files: `backends/*.py`, `tests/test_usage.py`, `tests/helpers.py`,
    `tests/test_auth_fatal.py`, `tests/test_local_engines.py`

- [x] 2. Runner writes per-item usage; scoring aggregates it
  - Acceptance: prediction records gain `usage` (null when absent); error
    records null; `results.json` has aggregate `usage` + `cost_per_correct`;
    historical records without usage still score to `usage: null`.
  - Verify: `uv run pytest -m "not gpu" -q`; real historical run rescored on
    a copy produced identical accuracy with `usage: null` (backward compat).
  - Files: `runner.py`, `scoring.py`, `tests/helpers.py`, `tests/test_runner.py`,
    `tests/test_scoring.py`

- [x] 3. Site payload: measured usage + reported overlay
  - Acceptance: `site/run_costs.json` overlay applies only when measured cost
    is absent; `cost.source` is `measured` / `reported` / null; unknown slugs
    ignored; payload carries per-run `usage` totals.
  - Verify: `uv run pytest tests/test_site_build.py -q` — done (6 new tests;
    a real bug caught: `per_correct` divided by item count, not correct count)
  - Files: `site/build.py`, `site/run_costs.json`, `tests/test_site_build.py`

- [x] 4. Site presentation: cost column, accuracy-vs-cost chart, method note
  - Acceptance: tier tables show USD cost with source distinction; Results
    page has an accuracy-vs-cost chart (runs with cost only); archive shows
    cost per correct; Method explains measured vs reported.
  - Verify: browser pass (Playwright) on a temp-built preview — cost section,
    exact-values table, run index, archive all render; zero console errors on
    results/index/runs pages; label-overflow and collision bugs found by
    screenshot and fixed.
  - Files: `site/charts.js`, `site/results.html`, `site/method.html`

- [ ] 5. Full verification and demo run
  - Acceptance: full offline suite green; `site/build.py` regenerates payload
    including the three reported costs; no historical run's score changes.
  - Verify: `uv run pytest -q` (127 passed, 5 skipped) ✓; preview build
    verified ✓; **publishing the real `site/results.json` + evidence is
    pending the user's go-ahead** (it also publishes motif-3).
  - Files: generated `site/results.json` + `site/data/runs/*` — not yet rebuilt
