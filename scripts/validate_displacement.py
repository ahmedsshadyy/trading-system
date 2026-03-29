from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.research.displacement_research import (
    build_displacement_research_table,
)
from src.validation.indicators.displacement import (
    build_displacement_analysis_base_frame,
    build_displacement_analysis_frame,
    validate_displacement,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/smc")
HTML_FILE = OUT_DIR / "displacement_validation_XAU_USD_H4.html"
STRONGEST_CSV = OUT_DIR / "displacement_strongest_XAU_USD_H4.csv"
NEAR_MISS_CSV = OUT_DIR / "displacement_near_miss_XAU_USD_H4.csv"
INVALID_INPUTS_CSV = OUT_DIR / "displacement_invalid_inputs_XAU_USD_H4.csv"
RESEARCH_QUALITY_CSV = (
    OUT_DIR / "displacement_research_strongest_quality_XAU_USD_H4.csv"
)
RESEARCH_FOLLOW_CSV = (
    OUT_DIR / "displacement_research_strongest_follow_through_XAU_USD_H4.csv"
)
RESEARCH_FAILURE_CSV = (
    OUT_DIR / "displacement_research_strongest_failures_XAU_USD_H4.csv"
)
PLOT_START = pd.Timestamp("2026-01-11", tz="UTC")

ATR_LENGTH = 14
BODY_ATR_MULT = 1.35
CLOSE_EXTREME_FRAC = 0.20
MIN_BODY_FRAC = 0.60
MAX_OPPOSITE_WICK_FRAC = 0.20


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
            if isinstance(item, dict):
                print(f"{prefix}-")
                _print_summary(item, indent=indent + 2)
            else:
                print(f"{prefix}- {item}")
        return
    print(f"{prefix}{value}")


def main() -> None:
    raw_df = pd.read_parquet(DATA_FILE)
    base_df = build_displacement_analysis_base_frame(
        raw_df,
        instrument="XAU_USD",
    )
    df = build_displacement_analysis_frame(
        base_df,
        atr_length=ATR_LENGTH,
        body_atr_mult=BODY_ATR_MULT,
        close_extreme_frac=CLOSE_EXTREME_FRAC,
        min_body_frac=MIN_BODY_FRAC,
        max_opposite_wick_frac=MAX_OPPOSITE_WICK_FRAC,
    )
    research_table = build_displacement_research_table(df, atr_length=ATR_LENGTH)

    plot_df = df[df["timestamp"] >= PLOT_START].copy()
    if plot_df.empty:
        raise ValueError(
            f"No XAU_USD H4 rows found for validation window starting {PLOT_START}."
        )

    title = (
        "Displacement Validation — XAU_USD H4 "
        f"(ATR={ATR_LENGTH}, body_atr={BODY_ATR_MULT}, body_frac={MIN_BODY_FRAC}, "
        f"close_extreme={CLOSE_EXTREME_FRAC}, opp_wick={MAX_OPPOSITE_WICK_FRAC}) "
        f"({PLOT_START.date()} to {plot_df['timestamp'].max().date()})"
    )
    result = validate_displacement(
        plot_df,
        full_df=df,
        outpath=HTML_FILE,
        title=title,
        atr_length=ATR_LENGTH,
        body_atr_mult=BODY_ATR_MULT,
        close_extreme_frac=CLOSE_EXTREME_FRAC,
        min_body_frac=MIN_BODY_FRAC,
        max_opposite_wick_frac=MAX_OPPOSITE_WICK_FRAC,
        research_table=research_table,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result["tables"]["detector"]["strongest"].to_csv(STRONGEST_CSV, index=False)
    result["tables"]["detector"]["near_miss"].to_csv(NEAR_MISS_CSV, index=False)
    result["tables"]["detector"]["invalid_inputs"].to_csv(
        INVALID_INPUTS_CSV, index=False
    )
    result["tables"]["research"]["strongest_quality"].to_csv(
        RESEARCH_QUALITY_CSV, index=False
    )
    result["tables"]["research"]["strongest_follow_through"].to_csv(
        RESEARCH_FOLLOW_CSV, index=False
    )
    result["tables"]["research"]["strongest_failures"].to_csv(
        RESEARCH_FAILURE_CSV, index=False
    )

    print("\n=== DISPLACEMENT SUMMARY ===")
    _print_summary(result["summary"])
    print(f"\nWrote chart to: {result['html_path']}")
    print(f"Wrote strongest table to: {STRONGEST_CSV}")
    print(f"Wrote near-miss table to: {NEAR_MISS_CSV}")
    print(f"Wrote invalid-input table to: {INVALID_INPUTS_CSV}")
    print(f"Wrote strongest-quality research table to: {RESEARCH_QUALITY_CSV}")
    print(f"Wrote strongest-follow-through research table to: {RESEARCH_FOLLOW_CSV}")
    print(f"Wrote strongest-failure research table to: {RESEARCH_FAILURE_CSV}")


if __name__ == "__main__":
    main()
