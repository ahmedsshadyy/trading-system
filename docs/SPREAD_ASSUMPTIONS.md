# Spread & Slippage Assumptions

## Source
Real historical spread is stored in the `candles.spread` column (from MetaApi/MT5).
The fixed assumptions below are used as fallback and for strategy design reference.

## Fixed Assumptions (used in triple-barrier labeling)

| Instrument | Spread  | Slippage | Total Cost | Unit     |
|------------|---------|----------|------------|----------|
| XAU/USD    | $0.35   | $0.15    | $0.50      | per side |
| USOIL      | $0.05   | $0.02    | $0.07      | per side |
| EUR/USD    | 1.5 pip | 0.5 pip  | 2.0 pips   | per side |
| USD/JPY    | 1.5 pip | 0.5 pip  | 2.0 pips   | per side |

## Application in Labeling
For a LONG signal:
- Adjusted TP1 = TP1 - spread_cost (exit at bid)
- Adjusted SL  = SL  + spread_cost (stop hits sooner)

For a SHORT signal:
- Adjusted TP1 = TP1 + spread_cost
- Adjusted SL  = SL  - spread_cost

## MVP1 vs MVP2+
- MVP1: use fixed assumptions above
- MVP2+: use real spread from candles.spread column per candle


## Real Spread Column (candles.spread)
- Unit: points (integer). 1 point = $0.001 for XAU/USD.
- To convert to dollars: spread_points / 1000
- Example: spread=270 → $0.27 actual spread
- Source: MT5 via MetaApi
