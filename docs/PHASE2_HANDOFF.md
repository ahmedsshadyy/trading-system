# Phase 2 Starting Point

## Phase 1 Complete ✅
- Historical data: 16 Parquet files in data/raw/ (MetaApi/OANDA Global Markets MT5)
- PostgreSQL: candles table loaded, signals and model_versions tables ready
- Alembic migrations applied
- Spread stored as integer points (1 point = $0.001 for XAU/USD)
- USD/JPY data incomplete — to be re-fetched at MVP2

## Data Source
- MetaApi cloud (account ID: cd8deca8-372b-43cd-a07c-7ba9fdd50e90)
- Currently UNDEPLOYED — redeploy for MVP2 live feed
- Symbols: XAUUSD.sml, USOIL.sml, EURUSD.sml, USDJPY.sml

## Environment
- Mac M-series, VSCode, Python 3.13
- Poetry virtualenv: trading-system-BdQ8SZcS-py3.13
- PostgreSQL 15 running locally
- Project path: ~/Developer/trading-system

## Next: Phase 2 — Indicator Library
Build order:
1. Standard indicators (ATR, RSI, MACD, EMA, ADX, BB) — validate vs TradingView
2. SMC detectors (FVG, OB, CHoCH, BOS, sweep) — validate by visual inspection
3. Order flow proxies (delta, VSA, wick ratio) — internal logic only

## Regime Freeze Status

Step 6A finalizes the canonical regime core.

- Canonical regime ontology is frozen to:
  - `RANGING`
  - `TRANSITIONAL`
  - `TRENDING`
- Canonical regime remains non-directional and distinct from trend/bias.
- Stabilization is part of the canonical regime semantics, not temporary
  smoothing glue.
- The current caution contract is frozen for now:
  - `regime_boundary_flag == 1`
  - `regime_confidence < 0.60`
  - `bars_in_regime <= 2`
  - or `regime_context_caution == 1`
- Future richer taxonomies belong in derived layers, not the base regime.
- Advanced regime-detection research is deferred to
  `docs/REGIME_DETECTION_NOTES.md`.

Only bug fixes or downstream semantic conflicts should change the canonical
regime core after this freeze.

## Trend-State Hardening Order

Trend now becomes the next priority before any further regime-core changes.

- First harden `trend_state` semantics:
  - tighten what `NEUTRAL` means
  - replace the old confidence ladder with normalized, decomposed confidence
  - add full trend transition and dwell fields
  - formalize bias inheritance / expiry / contradiction behavior
- Then re-run trend/regime interaction analysis using the dedicated trend
  validator and the existing regime validator.
- Only after that should any further regime tightening be considered.

Regime remains frozen during this trend hardening pass.

## Key Files
- scripts/fetch_data.py — MetaApi data fetch
- scripts/load_candles.py — Parquet to PostgreSQL loader
- alembic/versions/ — DB schema migrations
- docs/SPREAD_ASSUMPTIONS.md — spread/slippage reference
- docs/DATA_NOTES.md — USD/JPY issue documented
