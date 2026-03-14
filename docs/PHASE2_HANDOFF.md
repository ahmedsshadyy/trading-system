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

## Key Files
- scripts/fetch_data.py — MetaApi data fetch
- scripts/load_candles.py — Parquet to PostgreSQL loader
- alembic/versions/ — DB schema migrations
- docs/SPREAD_ASSUMPTIONS.md — spread/slippage reference
- docs/DATA_NOTES.md — USD/JPY issue documented
