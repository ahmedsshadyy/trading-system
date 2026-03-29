from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.volume_profile import add_volume_profile
from src.validation.common.reporting import (
    get_logger,
    load_windowed_parquet,
    mark_report_written,
    report_fingerprint,
    should_write_report,
)
from src.validation.indicators.volume_profile import validate_volume_profile

OUT_DIR = Path("notebooks/foundation")
PLOT_ROWS = 300
LOOKBACK = 80
VALIDATION_ROWS = PLOT_ROWS + LOOKBACK + 40
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
    if isinstance(value, list):
        for child in value:
            print(f"{prefix}- {child}")
        return
    print(f"{prefix}{value}")


def _as_tick_only(raw: pd.DataFrame) -> pd.DataFrame:
    if "tickVolume" in raw.columns and "volume" not in raw.columns:
        return raw.copy()
    out = raw.copy()
    if "tickVolume" not in out.columns:
        volume_idx = int(out.columns.get_loc("volume"))
        tick_volume = normalize_candle_schema(out, require_volume=True)["volume"]
        out = out.drop(columns=["volume"], errors="ignore")
        out.insert(volume_idx, "tickVolume", tick_volume.astype(float))
        return out
    return out.drop(columns=["volume"], errors="ignore")


def _as_volume_only(raw: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_candle_schema(raw, require_volume=True)
    out = raw.copy()
    if "tickVolume" in out.columns:
        tick_idx = int(out.columns.get_loc("tickVolume"))
        out = out.drop(columns=["tickVolume"], errors="ignore")
        out.insert(tick_idx, "volume", normalized["volume"].astype(float))
        return out
    out["volume"] = normalized["volume"].astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate volume profile features with lightweight defaults."
    )
    parser.add_argument("--html", action="store_true", help="Write HTML charts.")
    parser.add_argument("--full", action="store_true", help="Load the full dataset.")
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=VALIDATION_ROWS,
        help=f"Trailing rows to load when --full is not set. Default: {VALIDATION_ROWS}.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate HTML even when unchanged."
    )
    args = parser.parse_args()
    logger = get_logger("validate_volume_profile")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = load_windowed_parquet(data_file, full=args.full, tail_rows=args.tail_rows)

        tick_df = add_volume_profile(_as_tick_only(raw), lookback=LOOKBACK)
        volume_df = add_volume_profile(_as_volume_only(raw), lookback=LOOKBACK)
        stepped_df = add_volume_profile(
            _as_tick_only(raw),
            lookback=LOOKBACK,
            mode="stepped",
        )
        source_parity_ok = bool(tick_df.equals(volume_df))

        plot_df = tick_df.tail(PLOT_ROWS).copy()
        html_path = OUT_DIR / f"volume_profile_validation_{instrument}_{timeframe}.html"
        fingerprint = report_fingerprint(
            plot_df,
            extra={
                "instrument": instrument,
                "timeframe": timeframe,
                "mode": "volume_profile",
            },
        )
        write_html = args.html and should_write_report(
            html_path,
            fingerprint=fingerprint,
            force=args.force,
        )
        result = validate_volume_profile(
            plot_df,
            summary_df=tick_df,
            stepped_df=stepped_df,
            lookback=LOOKBACK,
            outpath=html_path if write_html else None,
            title=f"Volume Profile Validation — {instrument} {timeframe}",
            source_parity_ok=source_parity_ok,
        )

        print(f"\n=== VOLUME PROFILE SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])
        if args.html:
            if write_html:
                mark_report_written(html_path, fingerprint=fingerprint)
                logger.info("chart saved -> %s", html_path)
            else:
                logger.info("chart unchanged, cached -> %s", html_path)
        else:
            logger.info("html disabled; summary only for %s %s", instrument, timeframe)


if __name__ == "__main__":
    main()
