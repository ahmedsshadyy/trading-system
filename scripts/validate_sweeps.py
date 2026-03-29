from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings
from src.indicators.smc.sweeps import add_liquidity_sweep
from src.validation.indicators.sweeps import validate_sweeps

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
HTML_FILE = OUT_DIR / "sweeps_validation_XAU_USD_H4.html"
STRONGEST_CSV = OUT_DIR / "sweeps_strongest_XAU_USD_H4.csv"
CONFIRMED_CSV = OUT_DIR / "sweeps_confirmed_XAU_USD_H4.csv"
NEAR_MISS_CSV = OUT_DIR / "sweeps_near_miss_XAU_USD_H4.csv"

SWING_WINDOW = 4
SWING_RETRACE = 0.7
SWING_CONFIRM_BARS = 2
PLOT_START = pd.Timestamp("2026-01-01", tz="UTC")


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
    print(f"{prefix}{value}")


def main() -> None:
    df = normalize_candle_schema(pd.read_parquet(DATA_FILE), require_volume=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = add_atr(df)
    df = add_swings(
        df,
        window=SWING_WINDOW,
        min_retrace_atr=SWING_RETRACE,
        min_confirm_bars=SWING_CONFIRM_BARS,
    )
    df = add_liquidity_sweep(df, require_next_bar_confirmation=True)

    plot_df = df[df["timestamp"] >= PLOT_START].copy()
    if plot_df.empty:
        raise ValueError(
            f"No XAU_USD H4 rows found for validation window starting {PLOT_START}."
        )

    title = (
        "Sweeps Validation — XAU_USD H4 "
        f"(swing w={SWING_WINDOW}, ret={SWING_RETRACE}, confirm={SWING_CONFIRM_BARS}) "
        f"({PLOT_START.date()} to {plot_df['timestamp'].max().date()})"
    )
    result = validate_sweeps(
        plot_df,
        full_df=df,
        outpath=HTML_FILE,
        title=title,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["tables"]["strongest"].to_csv(STRONGEST_CSV, index=False)
    result["tables"]["confirmed"].to_csv(CONFIRMED_CSV, index=False)
    result["tables"]["near_miss"].to_csv(NEAR_MISS_CSV, index=False)

    print("\n=== SWEEPS SUMMARY ===")
    _print_summary(result["summary"])
    print(f"\nWrote chart to: {result['html_path']}")
    print(f"Wrote strongest table to: {STRONGEST_CSV}")
    print(f"Wrote confirmed table to: {CONFIRMED_CSV}")
    print(f"Wrote near-miss table to: {NEAR_MISS_CSV}")


if __name__ == "__main__":
    main()
