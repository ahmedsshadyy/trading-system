from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.sr_levels import SR_SIDE_RESISTANCE, SR_SIDE_SUPPORT
from src.indicators.foundation.sr_range_proxy import add_sr_range_proxy

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


def test_sr_range_proxy_activates_for_stable_band() -> None:
    df = _base(60)
    _swing_confirm(df, bar=6, price=95.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(df, bar=7, price=105.0, side=SR_SIDE_RESISTANCE, origin_bar=3)

    for bar in range(14, 40, 4):
        df.loc[bar, ["low", "close", "high"]] = [95.1, 99.8, 104.8]

    out = add_sr_range_proxy(df)
    probe = out.iloc[32]
    assert int(probe["sr_range_proxy_active"]) == 1
    assert np.isfinite(probe["sr_range_proxy_low"])
    assert np.isfinite(probe["sr_range_proxy_high"])
    assert probe["sr_range_proxy_quality_score"] >= 0.58


def test_sr_range_proxy_rejects_very_wide_band() -> None:
    df = _base(60)
    _swing_confirm(df, bar=6, price=88.0, side=SR_SIDE_SUPPORT, origin_bar=2)
    _swing_confirm(df, bar=7, price=128.0, side=SR_SIDE_RESISTANCE, origin_bar=3)

    out = add_sr_range_proxy(df)
    probe = out.iloc[25]
    assert probe["sr_range_proxy_width_atr"] > 4.5
    assert int(probe["sr_range_proxy_active"]) == 0
