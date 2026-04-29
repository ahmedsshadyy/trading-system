"""Render a single-period range_boundaries plot from the warm validation cache.

Reads the most recent cached `range_selected_debug` frame parquet (no recompute)
and renders the standard candlestick + active range high/low + confirm marker +
strength panel for an arbitrary `[date_from, date_to]` window.

Default window is 2023-06-01 .. 2023-09-30 (inclusive) for XAU_USD H4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.validation.indicators.range_boundaries import (
    plot_range_boundaries_validation,
)

CACHE_ROOT = ROOT / "data" / "validation_cache" / "validate_range_boundaries"
OUT_DIR = ROOT / "notebooks" / "foundation"


def _latest_frame_parquet(instrument: str, timeframe: str) -> Path:
    cache_dir = CACHE_ROOT / instrument / timeframe / "range_selected_debug"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"No cached range_selected_debug for {instrument}/{timeframe}. "
            f"Run: poetry run python scripts/validate_range_boundaries.py "
            f"--target selected-debug"
        )
    frames = sorted(
        cache_dir.glob("*.frame.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not frames:
        raise FileNotFoundError(f"No frame parquet under {cache_dir}.")
    return frames[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--date-from", default="2023-06-01")
    parser.add_argument("--date-to", default="2023-09-30")
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML path (defaults to notebooks/foundation/...).",
    )
    args = parser.parse_args()

    frame_path = _latest_frame_parquet(args.instrument, args.timeframe)
    frame = pd.read_parquet(frame_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    date_from = pd.Timestamp(args.date_from, tz="UTC")
    date_to = pd.Timestamp(args.date_to, tz="UTC") + pd.Timedelta(days=1)
    window = frame[
        (frame["timestamp"] >= date_from) & (frame["timestamp"] < date_to)
    ].copy()
    if window.empty:
        raise ValueError(
            f"No bars in {args.date_from}..{args.date_to} for "
            f"{args.instrument}/{args.timeframe}."
        )

    if args.out is not None:
        out_path = Path(args.out)
    else:
        from_tag = pd.Timestamp(args.date_from).strftime("%Y_%m")
        to_tag = pd.Timestamp(args.date_to).strftime("%Y_%m")
        out_path = (
            OUT_DIR / f"range_boundaries_period_{from_tag}_to_{to_tag}_"
            f"{args.instrument}_{args.timeframe}.html"
        )

    confirmed = int((window["range_detect_flag"] == 1).sum())
    active_rows = int((window["range_active"] == 1).sum())
    title = (
        f"Range Boundaries — {args.instrument} {args.timeframe} | "
        f"{args.date_from} to {args.date_to} | "
        f"bars={len(window)} confirms={confirmed} active_rows={active_rows}"
    )
    written = plot_range_boundaries_validation(window, outpath=out_path, title=title)
    print(f"frame_cache: {frame_path}")
    print(f"window_bars: {len(window)}")
    print(f"confirms_in_window: {confirmed}")
    print(f"active_rows_in_window: {active_rows}")
    print(f"html_path: {written}")


if __name__ == "__main__":
    main()
