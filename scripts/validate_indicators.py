"""

validate_indicators.py

Indicator validation script.

Loads real XAU_USD H4 data from PostgreSQL, runs the full indicator
pipeline, and prints summary statistics for sanity checking.

Usage: poetry run python scripts/validate_indicators.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from src.indicators import build_all_indicators
from src.indicators._helpers.schema import normalize_candle_schema

DEFAULT_LIMIT = 1500


def load_candles(instrument: str, timeframe: str, engine) -> pd.DataFrame:
    """Load candles from PostgreSQL."""
    query = f"""
        SELECT timestamp, open, high, low, close, volume, spread
        FROM candles
        WHERE instrument = '{instrument}' AND timeframe = '{timeframe}'
        ORDER BY timestamp ASC
    """
    df = normalize_candle_schema(pd.read_sql(query, engine), require_volume=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Summary validation for the full indicator stack."
    )
    parser.add_argument("--full", action="store_true", help="Load the full dataset.")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Trailing rows to load when --full is not set. Default: {DEFAULT_LIMIT}.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger = logging.getLogger("validate_indicators")
    engine = create_engine(os.getenv("DATABASE_URL"))

    for instrument in ["XAU_USD", "USOIL"]:
        for tf in ["H4", "H1"]:
            print(f"\n{'='*60}")
            print(f"  {instrument} {tf}")
            print(f"{'='*60}")

            df = load_candles(instrument, tf, engine)
            if not args.full:
                df = df.tail(args.limit).copy()
            print(f"  Loaded {len(df):,} candles")
            print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
            logger.info("running stack for %s %s on %s rows", instrument, tf, len(df))

            # Run full pipeline (skip VP on H1 — too slow)
            include_vp = tf == "H4"
            result = build_all_indicators(
                df, instrument=instrument, include_vp=include_vp
            )

            print(f"  Total columns: {len(result.columns)}")
            print(
                f"  NaN rows (first 200 expected): {result.iloc[:200].isna().any(axis=1).sum()}"
            )

            # Key indicator ranges
            print("\n  --- Key Indicator Ranges ---")
            for col in [
                "ema_20",
                "atr_14",
                "rsi_14",
                "adx_14",
                "macd_hist",
                "bb_width",
            ]:
                if col in result.columns:
                    s = result[col].dropna()
                    print(
                        f"  {col:25s}  min={s.min():10.4f}  max={s.max():10.4f}  mean={s.mean():10.4f}"
                    )

            # Binary flag activation rates
            print("\n  --- Binary Flag Activation Rates ---")
            binary_cols = [
                "bos_bull",
                "bos_bear",
                "choch_bull",
                "choch_bear",
                "fvg_bull",
                "fvg_bear",
                "ob_bull",
                "ob_bear",
                "sweep_high",
                "sweep_low",
                "displacement_candle",
                "rsi_div_bearish",
                "rsi_div_bullish",
                "equal_highs",
                "equal_lows",
            ]
            for col in binary_cols:
                if col in result.columns:
                    total = result[col].sum()
                    rate = total / len(result) * 100
                    print(f"  {col:25s}  count={int(total):5d}  rate={rate:.2f}%")

            # Regime distribution
            if "regime_label" in result.columns:
                print("\n  --- Regime Distribution ---")
                dist = result["regime_label"].value_counts()
                for label, count in dist.items():
                    print(f"  {label:20s}  {count:6d}  ({count/len(result)*100:.1f}%)")

            # Trend state distribution
            if "trend_state" in result.columns:
                print("\n  --- Trend State ---")
                dist = result["trend_state"].value_counts().sort_index()
                labels = {-1: "Bearish", 0: "Undefined", 1: "Bullish"}
                for val, count in dist.items():
                    print(
                        f"  {labels.get(val, str(val)):20s}  {count:6d}  ({count/len(result)*100:.1f}%)"
                    )

            # Spot-check: print last 3 rows of key columns for manual TradingView comparison
            print("\n  --- Last 3 Rows (for TradingView comparison) ---")
            spot_cols = [
                "timestamp",
                "close",
                "ema_20",
                "ema_50",
                "rsi_14",
                "atr_14",
                "adx_14",
                "macd_hist",
            ]
            spot_cols = [c for c in spot_cols if c in result.columns]
            print(result[spot_cols].tail(3).to_string(index=False))

    print(f"\n{'='*60}")
    print("  Validation complete. Compare last rows against TradingView.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
