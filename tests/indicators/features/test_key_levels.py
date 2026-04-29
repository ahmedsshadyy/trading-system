from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.features.key_levels import add_key_levels
from src.indicators.foundation.sr_levels import SR_SIDE_SUPPORT

ATR = 5.0


def _base(n: int, close: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.full(n, close),
            "high": np.full(n, close + 1.0),
            "low": np.full(n, close - 1.0),
            "close": np.full(n, close),
            "volume": np.ones(n),
            "atr_14": np.full(n, ATR),
        }
    )


def _ensure_swing_cols(df: pd.DataFrame) -> None:
    if "swing_high_confirm_flag" not in df.columns:
        df["swing_high_confirm_flag"] = 0
        df["swing_high_confirm_price"] = np.nan
        df["swing_high_confirm_origin_idx"] = np.nan
        df["swing_low_confirm_flag"] = 0
        df["swing_low_confirm_price"] = np.nan
        df["swing_low_confirm_origin_idx"] = np.nan


def _swing_confirm(
    df: pd.DataFrame,
    *,
    bar: int,
    price: float,
    side: int,
    origin_bar: int | None = None,
) -> None:
    _ensure_swing_cols(df)
    if origin_bar is None:
        origin_bar = max(0, bar - 3)
    if side == SR_SIDE_SUPPORT:
        df.at[bar, "swing_low_confirm_flag"] = 1
        df.at[bar, "swing_low_confirm_price"] = price
        df.at[bar, "swing_low_confirm_origin_idx"] = origin_bar
    else:
        df.at[bar, "swing_high_confirm_flag"] = 1
        df.at[bar, "swing_high_confirm_price"] = price
        df.at[bar, "swing_high_confirm_origin_idx"] = origin_bar


def test_key_levels_prefers_eqhl_support_with_structure_confirmation() -> None:
    df = _base(70)
    df["prev_day_low"] = np.nan
    df.loc[10:, "prev_day_low"] = 99.0
    _swing_confirm(df, bar=14, price=97.6, side=SR_SIDE_SUPPORT, origin_bar=8)

    df["eql_detect_flag"] = 0
    df["eql_level_on_detect"] = np.nan
    df["eql_origin_idx"] = np.nan
    df["eql_score_on_detect"] = np.nan
    df["eql_member_count_on_detect"] = np.nan
    df["eql_active"] = 0
    df.at[15, "eql_detect_flag"] = 1
    df.at[15, "eql_level_on_detect"] = 97.7
    df.at[15, "eql_origin_idx"] = 10
    df.at[15, "eql_score_on_detect"] = 0.95
    df.at[15, "eql_member_count_on_detect"] = 3
    df.loc[15:, "eql_active"] = 1

    df["trend_state"] = 1
    df["trend_bias_state"] = 1
    df["effective_trend_state"] = 1
    df["sweep_low_confirm_flag"] = 0
    df["choch_bull"] = 0
    df["bos_bull"] = 0
    df.at[28, "sweep_low_confirm_flag"] = 1
    df.at[29, "choch_bull"] = 1
    df.at[30, "bos_bull"] = 1

    out = add_key_levels(df)
    probe = out.iloc[32]
    assert probe["key_support_best_source_family"] == "eqhl"
    assert probe["key_support_selection_source"] in {"primary", "nearest"}
    assert probe["key_support_score"] > probe["nearest_support_strength"]


def test_key_levels_degrades_to_sr_only_when_structure_missing() -> None:
    df = _base(50)
    _swing_confirm(df, bar=6, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    out = add_key_levels(df)
    probe = out.iloc[20]
    assert np.isfinite(probe["key_support_zone_id"])
    assert np.isfinite(probe["key_support_score"])
    assert probe["key_support_best_source_family"] == "swing"
