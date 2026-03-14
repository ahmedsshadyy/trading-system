# Data Fetch Notes

## Fetch Date: 2026-03-14
## Source: MetaApi (OANDA Global Markets Demo, MT5)
## Account: MT5-1600105886, Server: OANDA_Global-Demo-1

## Instruments fetched:
- XAU_USD (XAUUSD.sml) — all timeframes complete, 2014 → 2026 ✅
- USOIL (USOIL.sml) — all timeframes complete, 2014 → 2026 ✅
- EUR_USD (EURUSD.sml) — all timeframes complete, 2014 → 2026 ✅
- USD_JPY (USDJPY.sml) — INCOMPLETE ⚠️

## Known Issue: USD/JPY truncated
USD/JPY pagination did not complete correctly. Candle counts are far below expected:
- D:   1,000 candles (expected ~3,997)
- H4:  1,999 candles (expected ~17,983)
- H1:  7,993 candles (expected ~65,000+)
- M15: 29,971 candles (expected ~260,000+)

## Fix:
Re-fetch USD/JPY when MetaApi account is redeployed for MVP2.
USD/JPY is not a primary instrument for MVP1 (XAU/USD and USOIL are).
Not a blocker for MVP1.

## Pricing note:
XAUUSD.sml prices are quoted correctly — verified $5,020 close on 2026-03-13
matches market price. Pricing is in USD per troy ounce as expected.

## Standard date cutoff for analysis:
All timeframes will be filtered to START_DATE = 2015-01-01 in the pipeline
to ensure consistent aligned history across all instruments and timeframes.
