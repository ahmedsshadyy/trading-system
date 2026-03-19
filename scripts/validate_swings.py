from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings
from src.validation.indicators.swings import validate_swings, plot_swings_validation

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Full-history numerics
    df = add_atr(df)
    df = add_swings(df, window=6)

    # Recent-slice visuals (last ~90 days)
    plot_end_ts = df["timestamp"].max()
    plot_start_ts = plot_end_ts - pd.Timedelta(days=90)
    plot_df = df[df["timestamp"] >= plot_start_ts].copy().reset_index(drop=True)

    result = validate_swings(
        df,
        outpath=OUT_DIR / "swings_validation.html",
        title="Swings Validation — XAU_USD H4",
        n_windows=3,
    )

    # Overwrite chart with readable recent slice
    result["html_path"] = plot_swings_validation(
        plot_df,
        outpath=OUT_DIR / "swings_validation.html",
        title="Swings Validation — XAU_USD H4 (Last 90 Days)",
    )

    print("\n=== SWINGS SUMMARY ===")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")

    print("\n=== SAMPLE HIGH SWING WINDOWS ===")
    for i, win in enumerate(result["high_windows"], start=1):
        print(f"\n--- high window {i} ---")
        print(win.to_string(index=True))

    print("\n=== SAMPLE LOW SWING WINDOWS ===")
    for i, win in enumerate(result["low_windows"], start=1):
        print(f"\n--- low window {i} ---")
        print(win.to_string(index=True))

    print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
