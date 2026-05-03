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
from src.indicators.structure.trend_state import add_trend_state
from src.indicators.structure.wedges import add_wedges
from src.indicators.structure.bos import add_bos
from src.indicators.smc.displacement import add_displacement_candle
from src.indicators.smc.fvg import add_fvg
from src.indicators.smc.ob import add_ob
from src.indicators.smc.equal_hl import add_equal_hl
from src.indicators.features.bos_context import add_bos_context
from src.validation.indicators.bos_context import validate_bos_context

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")

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
    df = add_trend_state(df)
    df = add_wedges(df)
    df = add_bos(
        df,
        min_source_age_bars=1,
        min_break_distance_atr=0.05,
        min_body_atr=0.10,
    )
    df = add_displacement_candle(df)
    df = add_fvg(df)
    df = add_ob(df)
    df = add_equal_hl(df)
    df = add_bos_context(df, include_forward_diagnostics=True)

    plot_end = df["timestamp"].max()
    plot_df = df[df["timestamp"] >= PLOT_START].copy().reset_index(drop=True)
    if plot_df.empty:
        raise ValueError(
            f"No XAU_USD H4 rows found for validation window starting {PLOT_START}."
        )

    title = (
        "BOS Context Validation — XAU_USD H4 "
        f"(swing w={SWING_WINDOW}, ret={SWING_RETRACE}, confirm={SWING_CONFIRM_BARS}) "
        f"({PLOT_START.date()} to {plot_end.date()})"
    )

    result = validate_bos_context(
        plot_df,
        outpath=OUT_DIR / "bos_context_validation_XAU_USD_H4.html",
        title=title,
    )

    print("\n=== BOS CONTEXT SUMMARY ===")
    _print_summary(result["summary"])
    print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
