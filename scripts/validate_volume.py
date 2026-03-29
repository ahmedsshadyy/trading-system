from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.volume import add_volume_features
from src.validation.indicators.volume import validate_volume

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)

        tick_df = add_volume_features(_as_tick_only(raw), include_research_only=False)
        volume_df = add_volume_features(
            _as_volume_only(raw), include_research_only=False
        )
        source_parity_ok = bool(tick_df.equals(volume_df))

        live_df = tick_df
        research_df = add_volume_features(
            _as_tick_only(raw), include_research_only=True
        )
        plot_df = research_df.tail(PLOT_ROWS).copy()
        html_path = OUT_DIR / f"volume_validation_{instrument}_{timeframe}.html"
        result = validate_volume(
            plot_df,
            summary_df=research_df,
            live_df=live_df,
            research_df=research_df,
            outpath=html_path,
            title=f"Volume Validation — {instrument} {timeframe}",
            source_parity_ok=source_parity_ok,
        )

        print(f"\n=== VOLUME SUMMARY: {instrument} {timeframe} ===")
        _print_summary(result["summary"])
        print(f"Wrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
