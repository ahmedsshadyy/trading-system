"""scripts/validate_trend_state.py"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.indicators.foundation.volatility import add_atr
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.validation.indicators.trend_state import (
    validate_trend_state,
    plot_trend_state_validation,
)

DATA_FILE = Path("data/raw/XAU_USD_H4.parquet")
OUT_DIR = Path("notebooks/structure")


def build_recent_transition_plot_df(
    df: pd.DataFrame,
    *,
    transition_col: str = "trend_state_changed",
    n_transitions: int = 8,
    pad: int = 12,
) -> pd.DataFrame:
    """
    Build a readable plotting slice centered around the most recent strict transitions.

    This keeps visual validation focused on regime changes instead of plotting a long,
    congested recent time range.
    """
    if transition_col not in df.columns:
        return df.tail(200).copy().reset_index(drop=True)

    transition_idxs = df.index[df[transition_col].fillna(0).astype(int) == 1].tolist()
    if not transition_idxs:
        return df.tail(200).copy().reset_index(drop=True)

    selected = transition_idxs[-n_transitions:]
    keep: set[int] = set()

    for idx in selected:
        start = max(0, idx - pad)
        end = min(len(df), idx + pad + 1)
        keep.update(range(start, end))

    ordered = sorted(keep)
    return df.iloc[ordered].copy().reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    print("loaded parquet")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # full-data computation for numeric validation
    df = add_atr(df)
    print("atr done")
    df = add_swings(df, window=6)
    print("swings done")
    df = add_trend_state(df)
    print("trend_state done")

    # transition-centered readable plotting slice
    plot_df = build_recent_transition_plot_df(
        df,
        transition_col="trend_state_changed",
        n_transitions=8,
        pad=12,
    )

    # numeric validation on full dataset
    result = validate_trend_state(
        df,
        outpath=OUT_DIR / "trend_state_validation.html",
        title="Trend State Validation — XAU_USD H4",
        n_windows=5,
    )
    print("numeric validation done")

    # overwrite chart with a readable transition-focused slice
    result["html_path"] = plot_trend_state_validation(
        plot_df,
        outpath=OUT_DIR / "trend_state_validation.html",
        title="Trend State Validation — XAU_USD H4 (Recent Transition Windows)",
    )
    print("plot done")

    summary = result["summary"]

    print("\n=== TREND STATE SUMMARY ===")
    print("strict_state_counts:", summary["strict_state_counts"])
    print("strict_state_pct:", summary["strict_state_pct"])
    print("bias_state_counts:", summary["bias_state_counts"])
    print("bias_state_pct:", summary["bias_state_pct"])
    print("transition_count:", summary["transition_count"])
    print("avg_bias_score_by_strict_state:", summary["avg_bias_score_by_strict_state"])
    print("avg_age_by_strict_state:", summary["avg_age_by_strict_state"])
    print(
        "avg_strength_raw_by_strict_state:", summary["avg_strength_raw_by_strict_state"]
    )
    print(
        "avg_strength_ema_by_strict_state:", summary["avg_strength_ema_by_strict_state"]
    )
    print("structure_loss_bull_rows:", summary["structure_loss_bull_rows"])
    print("structure_loss_bear_rows:", summary["structure_loss_bear_rows"])
    print("emerging_bull_rows:", summary["emerging_bull_rows"])
    print("emerging_bear_rows:", summary["emerging_bear_rows"])
    print("regime_phase_counts:", summary["regime_phase_counts"])
    print("strict_neutral_rows:", summary["strict_neutral_rows"])
    print("bias_carry_rows:", summary["bias_carry_rows"])

    print("\n=== STRICT READY CONSISTENCY ===")
    print("strict_bull_not_ready_rows:", summary["strict_bull_not_ready_rows"])
    print("strict_bear_not_ready_rows:", summary["strict_bear_not_ready_rows"])

    print("\n=== STRICT CONFIDENCE BURDEN ===")
    print("strict_bull_fresh_rows:", summary["strict_bull_fresh_rows"])
    print("strict_bull_intact_rows:", summary["strict_bull_intact_rows"])
    print("strict_bear_fresh_rows:", summary["strict_bear_fresh_rows"])
    print("strict_bear_intact_rows:", summary["strict_bear_intact_rows"])

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    print(summary["confidence_distribution"].to_string(index=False))

    print("\n=== STRICT TRANSITIONS TABLE ===")
    print(summary["transitions_table"].to_string(index=False))

    print("\n=== BIAS TRANSITIONS TABLE ===")
    if hasattr(summary["bias_transitions_table"], "to_string"):
        print(summary["bias_transitions_table"].to_string(index=False))
    else:
        print(summary["bias_transitions_table"])

    print("\n=== DURATION STATS ===")
    if hasattr(summary["duration_stats"], "to_string"):
        print(summary["duration_stats"].to_string())
    else:
        print(summary["duration_stats"])

    print("\n=== SAMPLE TREND TRANSITION WINDOWS ===")
    for i, win in enumerate(result["transition_windows"], start=1):
        print(f"\n--- transition window {i} ---")
        print(win.to_string(index=True))

    print(f"\nWrote chart to: {result['html_path']}")


if __name__ == "__main__":
    main()
