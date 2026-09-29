# Spec: per-item usage + cost tracking (harness → site)

Status: approved in chat (per-item granularity chosen over run-level totals).
Date: 2026-09-26.

## Objective

Record token usage and provider-reported cost **per item** during inference,
aggregate them per run, and surface run cost on the results site — so cost
becomes evidence that travels with each run instead of being scraped from a
provider dashboard after the fact.

Why now: the top of the board (claude-opus-5.5 90.4%, gpt-6-astra 90.1%,
gpt-6-sol 86.1%) is statistically tied on accuracy, so cost is the remaining
discriminator among frontier configs. Runs also resume repeatedly and abort
mid-run; cost must survive on the record like latency does.

## Scope

In scope:

- Backend contract carries usage: `generate()` returns `GenerationResult`
  (`text` + `usage`) instead of bare strings.
- Usage extraction for all four backends:
  - OpenAI-compatible: prompt/completion tokens, reasoning tokens, and
    `cost` when the provider reports it (OpenRouter returns `usage.cost`
    automatically; credits are USD).
  - Anthropic SDK: input/output token counts; no cost field.
  - transformers / vLLM: token counts counted from the tokenizer/generator;
    no cost (local compute).
- Prediction records gain `usage` (per item; `null` when unavailable).
- `results.json` gains an aggregate `usage` block and `cost_per_correct`.
- Site payload gains per-run `cost` (`usd`, `per_correct`, `source`) and
  `usage` token totals.
- `site/run_costs.json`: operator-reported cost overlay for historical runs
  that predate usage capture, merged at build time and labeled `reported`.
- Site presentation: cost column in the tier tables, an accuracy-vs-cost
  chart on the Results page, a cost fact in the run archive, and a Method
  note explaining measured vs reported.

Out of scope (explicit non-goals):

- Spend gating / `--max-cost` abort.
- `estimated` cost from tokens × price (field reserved but never emitted in v1).
- Re-running historical models to obtain measured cost.
- Any change to the official protocol (prompt, decoding, extraction, scoring).

## Domain contract

Canonical usage object (all keys optional; omit unknown; `null` when no
usage at all):

```json
{
  "prompt_tokens": 812,
  "completion_tokens": 1543,
  "reasoning_tokens": 640,
  "cost": 0.0031,
  "cost_source": "provider"
}
```

- `cost_source` is present iff `cost` is: `"provider"` (measured — provider
  returned a cost), `"reported"` (site overlay only, never written by the
  harness), `"estimated"` (reserved, not emitted in v1).
- Units: USD. OpenRouter credits are USD-denominated.

Prediction record (adds one key to the existing schema):

```json
{"id": "...", "usage": { ... } | null, "...": "existing fields unchanged"}
```

`results.json` aggregate:

```json
"usage": {
  "records_with_usage": 700,
  "prompt_tokens": 568400,
  "completion_tokens": 1080100,
  "reasoning_tokens": 448000,
  "total_cost": 14.52,
  "cost_source": "provider"
} | null,
"cost_per_correct": 0.0229 | null
```

Site payload per run:

```json
"usage": { "...same shape as results.json..." } | null,
"cost": {
  "usd": 14.50,
  "per_correct": 0.0229,
  "source": "measured" | "reported"
} | null
```

Precedence: measured `usage.total_cost` wins; else overlay; else null.

## Commands

```bash
uv run pytest -m "not gpu" -q            # CI selection (must stay green)
uv run pytest -q                          # full offline suite
uv run python site/build.py               # regenerate site/results.json + evidence
```

## Project structure

```
src/kobalt_eval/backends/base.py        GenerationResult + contract
src/kobalt_eval/backends/api_openai.py  usage/cost extraction
src/kobalt_eval/backends/api_anthropic.py
src/kobalt_eval/backends/transformers_backend.py
src/kobalt_eval/backends/vllm_backend.py
src/kobalt_eval/runner.py               writes usage per record
src/kobalt_eval/scoring.py              aggregates usage + cost_per_correct
tests/                                  mirrors module layout
site/build.py                           overlay merge + payload fields
site/run_costs.json                     operator-reported historical costs
site/charts.js / results.html / method.html / runs.html
```

## Code style

Follow the existing modules: `from __future__ import annotations`, typed
signatures, no new dependencies, lazy heavy imports. Usage extraction is a
small module-level helper (`_usage_from_response(resp)`) with defensive
attribute/dict access, unit-tested against fakes — not against the SDKs.

## Testing strategy

pytest, offline by default (existing convention). TDD per task:

- Unit: usage extraction for fakes shaped like each SDK response; missing
  usage → `None`; malformed fields ignored.
- Integration: runner records carry `usage`; error records carry `null`;
  scoring sums tokens/cost and computes `cost_per_correct`; runs without
  usage produce `usage: null` (backward compatible).
- Site: payload picks measured over reported; overlay applies only when a
  run lacks measured cost; unknown slugs ignored (never fatal).
- Live-engine token counting is covered by the existing GPU-gated tests
  (skipped in CI), asserting usage is present when engines really run.

## Boundaries

- Always: run `uv run pytest -m "not gpu" -q` before considering a task done;
  keep prediction/result schemas additive (new keys only).
- Ask first: changing the backend contract again, adding dependencies,
  changing CI config, publishing (committing) regenerated site data.
- Never: log or persist API keys, rewrite historical predictions files,
  invent a measured cost where the provider gave none.

## Success criteria

1. A fake OpenAI response carrying `usage` (tokens + `cost`) yields a
   `GenerationResult.usage` with `cost_source: "provider"`; a response with
   no usage yields `None`.
2. Anthropic fakes yield tokens with no cost key.
3. A stub run produces `usage` on every prediction record (null for stubs
   without usage), `results.json` aggregates totals, and
   `cost_per_correct == total_cost / num_correct`.
4. Historical run dirs (no usage in predictions) still score, with
   `usage: null` and no crash.
5. `site/build.py` emits `cost.source == "reported"` for a slug in
   `site/run_costs.json` and `"measured"` when `results.json` carries a
   provider cost; a run with neither gets `cost: null`.
6. `uv run pytest -m "not gpu" -q` is green.

## Open questions

- None blocking. The three dashboard figures (opus 14.50 / astra 3.89 /
  sol 1.81 USD) seed the overlay; remaining runs stay `not recorded` until
  the operator supplies more.
