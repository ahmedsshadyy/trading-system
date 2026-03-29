"""
scripts/validate_fibonacci.py

Interactive HTML validation chart for the Fibonacci level engine.
Filters to a recent window for the chart view.

Layout (4 rows, shared x-axis):
  Row 1  Price + all active Fibonacci levels (bull=blue, bear=red, golden 0.618=gold)
  Row 2  ATR-normalised distance to nearest fib level (bull + bear)
  Row 3  Active structure count over time (bull + bear)
  Row 4  Quality score of selected active structures

Output: notebooks/foundation/fibonacci_validation_{instrument}_{timeframe}.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.foundation.fibonacci import add_fibonacci_levels
from src.indicators.research.fibonacci_research import build_fibonacci_research_table
from src.validation.indicators.fibonacci import validate_fibonacci

DATE_FROM = "2026-01-10"
OUT_DIR = Path("notebooks/foundation")
RUNS = (("XAU_USD", "H4"),)


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            else:
                print(f"{prefix}{key}: {child}")
        return
    if isinstance(value, list):
        for item in value:
            print(f"{prefix}- {item}")
        return
    print(f"{prefix}{value}")


def main(instrument: str = "XAU_USD", timeframe: str = "H4") -> None:
    data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
    if not data_file.exists():
        print(f"[ERROR] data file not found: {data_file}")
        sys.exit(1)

    raw_df = pd.read_parquet(data_file)
    df = build_research_indicators(
        raw_df,
        instrument=instrument,
        include_vp=False,
        include_avwap=False,
    )
    df = add_fibonacci_levels(df)
    research_table = build_fibonacci_research_table(df)

    date_from = pd.Timestamp(DATE_FROM, tz="UTC")
    plot_df = df[df["timestamp"] >= date_from].copy()
    if plot_df.empty:
        raise ValueError(
            f"No {instrument} {timeframe} rows found for validation window starting {DATE_FROM}."
        )

    out_html = OUT_DIR / f"fibonacci_validation_{instrument}_{timeframe}.html"
    strongest_bull_csv = (
        OUT_DIR / f"fibonacci_strongest_bull_{instrument}_{timeframe}.csv"
    )
    strongest_bear_csv = (
        OUT_DIR / f"fibonacci_strongest_bear_{instrument}_{timeframe}.csv"
    )
    all_events_csv = OUT_DIR / f"fibonacci_all_events_{instrument}_{timeframe}.csv"

    result = validate_fibonacci(
        plot_df,
        full_df=df,
        outpath=out_html,
        title=(
            f"Fibonacci Levels Validation — {instrument} {timeframe} "
            f"({DATE_FROM} to {plot_df['timestamp'].max().date()}, chart window only)"
        ),
        research_table=research_table,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["tables"]["strongest_bull"].to_csv(strongest_bull_csv, index=False)
    result["tables"]["strongest_bear"].to_csv(strongest_bear_csv, index=False)
    result["tables"]["all_events"].to_csv(all_events_csv, index=False)

    print(f"\nFibonacci Validation — {instrument} {timeframe}")
    print(f"  Chart window: {date_from.date()} → {plot_df['timestamp'].max().date()}")
    print(f"  Full dataset rows: {len(df):,}")
    print(f"  Chart window rows: {len(plot_df):,}")
    print(f"  Total fib events: {len(research_table):,}")
    bull_events = research_table[research_table["fib_r_direction"] == 1]
    bear_events = research_table[research_table["fib_r_direction"] == -1]
    print(f"  Bull structures: {len(bull_events):,}")
    print(f"  Bear structures: {len(bear_events):,}")
    print()
    print("Research Summary:")
    _print_summary(result["research_summary"])
    print()
    print(f"  HTML: {out_html}")
    print(f"  Strongest bull CSV: {strongest_bull_csv}")
    print(f"  Strongest bear CSV: {strongest_bear_csv}")
    print(f"  All events CSV: {all_events_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Fibonacci levels")
    parser.add_argument("--instrument", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    args = parser.parse_args()
    main(instrument=args.instrument, timeframe=args.timeframe)
