from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.features.cross_asset import (
    build_global_market_context,
    load_raw_context_frames,
)
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.research.smt_research import build_smt_research_table
from src.validation.indicators.smt import summarize_smt, validate_smt

OUT_DIR = Path("notebooks/cross_asset")
PLOT_ROWS = 300
INPUT_ROWS = 2500
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
    ("USOIL", "H1"),
    ("USOIL", "H4"),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SMT cross-asset output.")
    parser.add_argument(
        "--instrument",
        choices=("all", "XAU_USD", "USOIL"),
        default="all",
        help="Primary instrument to validate.",
    )
    parser.add_argument(
        "--timeframe",
        choices=("all", "H1", "H4"),
        default="all",
        help="Timeframe to validate.",
    )
    parser.add_argument(
        "--plot-rows",
        type=int,
        default=PLOT_ROWS,
        help="Rows to keep in the HTML chart window.",
    )
    parser.add_argument(
        "--input-rows",
        type=int,
        default=INPUT_ROWS,
        help=(
            "Recent raw bars to load before running the pipeline. "
            "Use 0 to keep the full file."
        ),
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=5,
        help="Number of sample bullish/bearish SMT windows to print.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Run numeric validation only and skip HTML chart generation.",
    )
    return parser.parse_args()


def _selected_runs(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for instrument, timeframe in RUNS:
        if args.instrument != "all" and instrument != args.instrument:
            continue
        if args.timeframe != "all" and timeframe != args.timeframe:
            continue
        selected.append((instrument, timeframe))
    return selected


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in _selected_runs(args):
        raw = pd.read_parquet(Path(f"data/raw/{instrument}_{timeframe}.parquet"))
        if args.input_rows > 0:
            raw = raw.tail(args.input_rows).reset_index(drop=True)
        full_df = build_research_indicators(
            raw,
            instrument=instrument,
            include_vp=False,
            include_avwap=False,
            timeframe=timeframe,
            include_cross_asset=True,
            raw_data_root="data/raw",
        )
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
        market_context_frames = load_raw_context_frames(
            raw_data_root="data/raw",
            timeframe=timeframe,
        )
        if args.input_rows > 0:
            market_context_frames = {
                symbol: frame.tail(args.input_rows).reset_index(drop=True)
                for symbol, frame in market_context_frames.items()
            }
        market_context_frames[instrument] = raw.copy()
        market_context = build_global_market_context(
            market_context_frames,
            timeframe=timeframe,
        )
        research_table = build_smt_research_table(full_df)
        plot_df = full_df.tail(args.plot_rows).copy()

        title = f"SMT Validation — {instrument} {timeframe}"
        outpath = OUT_DIR / f"smt_validation_{instrument}_{timeframe}.html"

        if args.numeric_only:
            result = {
                "summary": summarize_smt(
                    full_df,
                    market_context=market_context,
                    research_table=research_table,
                ),
                "bull_windows": [],
                "bear_windows": [],
                "html_path": None,
            }
        else:
            result = validate_smt(
                plot_df,
                full_df=full_df,
                market_context=market_context,
                research_table=research_table,
                outpath=outpath,
                title=title,
                n_windows=args.n_windows,
            )

        print(f"\n=== SMT SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])

        if result["bull_windows"]:
            print("\n=== SAMPLE BULL SMT WINDOWS ===")
            for idx, window in enumerate(result["bull_windows"], start=1):
                print(f"\n--- bull window {idx} ---")
                print(window.to_string(index=True))

        if result["bear_windows"]:
            print("\n=== SAMPLE BEAR SMT WINDOWS ===")
            for idx, window in enumerate(result["bear_windows"], start=1):
                print(f"\n--- bear window {idx} ---")
                print(window.to_string(index=True))

        if result["html_path"] is not None:
            print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
