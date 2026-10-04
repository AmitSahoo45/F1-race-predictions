# Apex Forecast

A cinematic Formula 1 forecasting application: a local Python data and ML pipeline publishes validated, immutable JSON to a static Next.js site. There is no runtime API or database server.

**Current evidence:** the application uses real provider calendars and archived session observations. Synthetic predictions have been removed from its public data bundle. Full historical ingestion and chronological evaluation are in progress; accuracy is not claimed before those reports exist. A prospective race weekend and public deployment remain separate release gates; see [verification](docs/verification.md).

## Run locally

Prerequisites: Python 3.13, Node.js 22, uv, and pnpm 9.15.9. Run commands from the project root unless noted.

```powershell
uv sync --locked --all-groups
uv run f1forecast validate
pnpm --dir web install --frozen-lockfile
pnpm --dir web dev
```

Open `http://localhost:3000`. The local public artifact bundle is sufficient to run the application offline after dependency installation. If using the environment installed by Codex on Windows, uv is also available at `.venv/Scripts/uv.exe` and Python at `.venv/Scripts/python.exe`.

```powershell
uv run pytest -q
uv run pytest pipeline/tests/test_models.py::test_pl_samples_are_reproducible_normalized_and_nested -q
uv run ruff check pipeline
uv run pyright
pnpm --dir web lint
pnpm --dir web test:unit
pnpm --dir web build
pnpm --dir web typecheck
pnpm --dir web exec playwright install chromium
pnpm --dir web test:e2e --workers=1
```

Windows restricted environments can require an elevated tool execution for Python temporary directories. Use a dedicated workspace test directory: `uv run pytest --basetemp=.cache/pytest-local -q`. Never point `--basetemp` at a directory containing work you want to keep: pytest manages that directory.

## Data → training → release

The pipeline exposes commands for cached, resumable ingestion. Direct FastF1 downloads honor its provider limits; the historical archive and its coverage are documented separately in [data feasibility](docs/data-feasibility.md). Raw archives stay local.

```powershell
# Inspect one real session first: Bahrain 2025 FP1.
uv run f1forecast ingest 2025 4 FP1

# Full eligible history; progress is preserved across failures/restarts.
uv run f1forecast backfill --start-year 2022 --end-year 2026
uv run f1forecast prepare
uv run f1forecast export-history

# Use the same prepared dataset and recipe for evaluation and final fitting.
uv run f1forecast evaluate qualifying
uv run f1forecast evaluate race
uv run f1forecast train qualifying --training-cutoff 2026-09-26T00:00:00Z
uv run f1forecast train race --training-cutoff 2026-09-26T00:00:00Z

# Publish real evaluation evidence only after the historical test has run.
uv run f1forecast publish-evaluation --reports models/runs/qualifying.evaluation.json models/runs/race.evaluation.json

# Baseline is eligible if evaluated; CatBoost additionally needs the promotion gate.
uv run f1forecast approve --kind baseline --metadata models/runs/race.metadata.json --report models/runs/race.evaluation.json --history data/history.parquet
```

Choose a training cutoff after the completed races you intend to include and before the forecast being served. Run the corresponding approval command for qualifying too. `--kind catboost` is rejected if the learned model has not beaten the strongest development-selected baseline on paired held-out events without worsening position error. Reports and model metadata must match their data and training recipe. Published probability claims remain model estimates.

Approved releases contain small **derived** history summaries, model weights, evaluation evidence and integrity hashes under `models/approved/`; keep these together when committing a release. Raw Parquet archives, DuckDB and FastF1 caches stay under ignored `data/`. Each driver probability comes from the same 20,000 seeded full-order samples. Top ten is a points-position proxy, not awarded-points probability.

## Weekend operation

```powershell
uv run f1forecast sync-calendar 2026
uv run f1forecast tick --refresh-calendar
uv run f1forecast validate
```

Calendar synchronization refreshes actual provider events. Forecasts are issued once, in the final polling window before the 30-minute cutoff. Delayed jobs never backfill an issued forecast. A missing approved model, required input, or unvalidated optional-data case prevents publication. Missing forecasts become explicit unavailable states in the UI even if a static build has become old.

Entrants can be inferred from the latest completed pre-target session, with this provenance visible in the forecast. To incorporate a confirmed substitution before a specific target, use a JSON array of driver objects (`id`, `code`, `name`, `team`, `color`) and record its source:

```powershell
uv run f1forecast set-entrants 2026-18 --target race --file data/race-entry-list.json --source "URL or reference for the pre-race entry list"
```

Target rosters are independent; a replacement before the race must not rewrite the qualifying forecast. A late withdrawal or a change not yet available to the pipeline remains a forecasting limitation. Actual results are exported separately once the provider confirms session finalization.

```powershell
uv run f1forecast export-analysis 2025-04
```

Only use `demo` for the demonstration site or a separate output directory; it restores synthetic examples. The archive rejects changes to an existing forecast ID. Recovery from an interrupted publication is documented in [operations](docs/operations.md).

## Static hosting and automation

The workflows are ready for a public GitHub repository, but no repository has been created, pushed or deployed by this implementation.

1. Push reviewed code and derived release artifacts to your repository.
2. Select **GitHub Actions** as the repository's Pages build source.
3. Run **Deploy static site**. Its build receives the correct project `basePath`.
4. Scheduled inference runs at minutes 7 and 37. It exits without provider calls when no model is approved and without artifact writes outside active weekend windows.

The forecast workflow preserves its cache, serializes publication, verifies existing forecast hashes and explicitly deploys the new publication commit. GitHub scheduling is best effort; runs may be delayed or disabled after repository inactivity. Check the workflow summary and the forecast's recorded cutoff. No paid service or secret is needed for historical ingestion.

## Structure and contracts

`pipeline/` owns ingestion, features, models, evaluation and publication. `web/` owns the static application. `site-data/` is the public artifact bundle. `schemas/` is generated from Pydantic; TypeScript types are generated from those schemas. Do not hand-edit generated schemas or types.

```powershell
uv run f1forecast schema
pnpm --dir web generate:types
uv run f1forecast validate
```

See [PLAN.md](PLAN.md), [model method](docs/model-method.md), [data feasibility](docs/data-feasibility.md), [community dataset review](docs/dataset-research.md), and [operations](docs/operations.md). Code is MIT-licensed; provider data and bundled font assets retain their own terms. This is an unofficial project, not associated with Formula 1.
