from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings
from src.validation.indicators.swings import validate_swings

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")

SWING_WINDOW = 4
SWING_RETRACE = 0.7
SWING_CONFIRM_BARS = 2


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = add_atr(df)
    df = add_swings(
        df,
        window=SWING_WINDOW,
        min_retrace_atr=SWING_RETRACE,
        min_confirm_bars=SWING_CONFIRM_BARS,
    )

    title_base = (
        f"Swings Validation — XAU_USD H4 "
        f"(w={SWING_WINDOW}, ret={SWING_RETRACE}, confirm={SWING_CONFIRM_BARS})"
    )

    result = validate_swings(
        df,
        outpath=OUT_DIR / "swings_validation.html",
        title=f"{title_base} (2026-01-01 to 2026-03-14)",
        start_ts="2026-01-01",
        end_ts="2026-03-15",
        n_windows=3,
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
