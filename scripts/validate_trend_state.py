from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.regime import add_regime
from src.indicators.foundation.session import add_session_features
from src.indicators.foundation.volatility import add_atr, add_bb_width
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.validation.indicators.trend_state import validate_trend_state

OUT_DIR = Path("notebooks/structure")
PLOT_ROWS = 300
RUNS = (
    ("XAU_USD", "H1"),
    ("XAU_USD", "H4"),
)
SWING_WINDOW = 6


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            elif hasattr(child, "to_string"):
                print(f"{prefix}{key}:")
                print(
                    child.to_string(index=False)
                    if isinstance(child, pd.DataFrame)
                    else child.to_string()
                )
            else:
                print(f"{prefix}{key}: {child}")
        return
    print(f"{prefix}{value}")


def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candle_schema(df, require_volume=False)
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = add_atr(out)
    out = add_emas(out)
    out = add_adx(out)
    out = add_bb_width(out)
    out = add_swings(out, window=SWING_WINDOW)
    out = add_trend_state(out)

    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:
        out = add_session_features(out, include_research_only=False)

    out = add_regime(out, include_research_only=False)
    return out


def _print_validation_sections(summary: dict[str, object]) -> None:
    ordered_keys = [
        "current_trend_snapshot",
        "strict_state_counts",
        "strict_state_pct",
        "bias_state_counts",
        "bias_state_pct",
        "transition_count",
        "transition_matrix",
        "bias_transition_matrix",
        "dwell_diagnostics",
        "confidence_by_state",
        "confidence_ordering_check",
        "confidence_separation_check",
        "neutral_confidence_cap_check",
        "strength_by_state",
        "commitment_by_state",
        "bias_interaction",
        "regime_interaction",
        "neutral_overuse_audit",
        "neutral_confidence_audit",
        "neutral_in_trend_audit",
        "directional_in_range_audit",
        "old_neutral_strong_env_audit",
        "stale_neutral_promotion_candidate_audit",
        "mature_directional_in_range_decay_audit",
        "neutral_with_directional_bias_audit",
        "commit_gap_audit",
        "neutral_age_audit",
        "semantic_buckets",
    ]
    print(f"row_count: {summary['row_count']}")
    for key in ordered_keys:
        if key in summary:
            print(f"{key}:")
            _print_summary(summary[key], indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for instrument, timeframe in RUNS:
        data_file = Path(f"data/raw/{instrument}_{timeframe}.parquet")
        raw = pd.read_parquet(data_file)
        full_df = _build_context(raw)
        plot_df = full_df.tail(PLOT_ROWS).copy()

        html_path = OUT_DIR / f"trend_state_validation_{instrument}_{timeframe}.html"
        result = validate_trend_state(
            plot_df,
            summary_df=full_df,
            outpath=html_path,
            title=f"Trend State Validation — {instrument} {timeframe}",
            n_windows=5,
        )

        print(f"\n=== TREND STATE SUMMARY: {instrument} {timeframe} ===")
        _print_validation_sections(result["summary"])

        print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
