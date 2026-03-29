# scripts/validate_wedges.py

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
from src.indicators.structure.wedges import add_wedges
from src.validation.indicators.wedges import validate_wedges

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")

SWING_WINDOW = 4
SWING_RETRACE = 0.7
SWING_CONFIRM_BARS = 2


def main() -> None:
    df = normalize_candle_schema(pd.read_parquet(DATA_FILE), require_volume=False)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_atr(df)
    df = add_swings(
        df,
        window=SWING_WINDOW,
        min_retrace_atr=SWING_RETRACE,
        min_confirm_bars=SWING_CONFIRM_BARS,
    )
    df = add_wedges(df)

    plot_df = (
        df[df["timestamp"] >= pd.Timestamp("2026-01-01", tz="UTC")]
        .copy()
        .reset_index(drop=True)
    )

    title = (
        f"Wedges Validation — XAU_USD H4 "
        f"(swing w={SWING_WINDOW}, ret={SWING_RETRACE}, confirm={SWING_CONFIRM_BARS}) "
        f"(2026-01-01 onward)"
    )

    result = validate_wedges(
        plot_df,
        outpath=OUT_DIR / "wedges_validation.html",
        title=title,
        n_windows=5,
    )

    print("\n=== WEDGES SUMMARY ===")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")

    print("\n=== SAMPLE WEDGE BREAKOUT UP WINDOWS ===")
    for i, win in enumerate(result["breakout_up_windows"], start=1):
        print(f"\n--- breakout up window {i} ---")
        print(win.to_string(index=True))

    print("\n=== SAMPLE WEDGE BREAKOUT DOWN WINDOWS ===")
    for i, win in enumerate(result["breakout_down_windows"], start=1):
        print(f"\n--- breakout down window {i} ---")
        print(win.to_string(index=True))

    print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
