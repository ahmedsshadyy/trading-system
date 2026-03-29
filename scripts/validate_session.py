from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.foundation.session import add_session_features
from src.validation.indicators.session import validate_session

OUT_DIR = Path("notebooks/foundation")
PLOT_ROWS = 300
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
)


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


def _parity_ok(live_df: pd.DataFrame, research_df: pd.DataFrame) -> bool:
    live_cols = [col for col in research_df.columns if not col.startswith("r_")]
    return live_df[live_cols].equals(research_df[live_cols])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)

        live_df = add_session_features(raw, include_research_only=False)
        research_df = add_session_features(raw, include_research_only=True)
        parity_ok = _parity_ok(live_df, research_df)

        plot_df = research_df.tail(PLOT_ROWS).copy()
        html_path = OUT_DIR / f"session_validation_{instrument}_{timeframe}.html"
        result = validate_session(
            plot_df,
            summary_df=research_df,
            live_df=live_df,
            outpath=html_path,
            title=f"Session Validation — {instrument} {timeframe}",
            parity_ok=parity_ok,
        )

        print(f"\n=== SESSION SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])
        print(f"Wrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
