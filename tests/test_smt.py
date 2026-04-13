from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smt import add_smt_divergence


def _base_processed(rows: int = 8) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    close = np.linspace(100.0, 101.0, rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "atr_14": np.full(rows, 1.0, dtype=float),
            "swing_high_confirm_flag": np.zeros(rows, dtype=np.int8),
            "swing_low_confirm_flag": np.zeros(rows, dtype=np.int8),
            "swing_high_confirm_origin_idx": np.full(rows, np.nan, dtype=float),
            "swing_low_confirm_origin_idx": np.full(rows, np.nan, dtype=float),
            "swing_high_confirm_price": np.full(rows, np.nan, dtype=float),
            "swing_low_confirm_price": np.full(rows, np.nan, dtype=float),
        }
    )


def _set_low_event(
    df: pd.DataFrame,
    *,
    detect_idx: int,
    origin_idx: int,
    price: float,
) -> None:
    df.loc[detect_idx, "swing_low_confirm_flag"] = 1
    df.loc[detect_idx, "swing_low_confirm_origin_idx"] = float(origin_idx)
    df.loc[detect_idx, "swing_low_confirm_price"] = float(price)


def _set_high_event(
    df: pd.DataFrame,
    *,
    detect_idx: int,
    origin_idx: int,
    price: float,
) -> None:
    df.loc[detect_idx, "swing_high_confirm_flag"] = 1
    df.loc[detect_idx, "swing_high_confirm_origin_idx"] = float(origin_idx)
    df.loc[detect_idx, "swing_high_confirm_price"] = float(price)


def test_inverse_pair_bullish_divergence_is_detected() -> None:
    primary = _base_processed()
    partner = _base_processed()
    _set_low_event(primary, detect_idx=2, origin_idx=1, price=100.0)
    _set_low_event(primary, detect_idx=4, origin_idx=3, price=90.0)
    _set_high_event(partner, detect_idx=2, origin_idx=1, price=110.0)
    _set_high_event(partner, detect_idx=4, origin_idx=3, price=105.0)

    result = add_smt_divergence(
        primary,
        partner,
        inverse=True,
        partner_name="dxy",
        confirm_window=0,
    )

    assert result.loc[4, "xasset_smt_dxy_bull_flag"] == 1
    assert result.loc[4, "xasset_smt_dxy_bear_flag"] == 0
    assert result.loc[4, "xasset_smt_dxy_dir"] == 1
    assert 0.0 <= result.loc[4, "xasset_smt_dxy_score"] <= 1.0
    assert int(result["xasset_smt_dxy_bull_flag"].sum()) == 1


def test_positive_pair_bearish_divergence_is_detected() -> None:
    primary = _base_processed()
    partner = _base_processed()
    _set_high_event(primary, detect_idx=2, origin_idx=1, price=110.0)
    _set_high_event(primary, detect_idx=4, origin_idx=3, price=120.0)
    _set_high_event(partner, detect_idx=2, origin_idx=1, price=111.0)
    _set_high_event(partner, detect_idx=4, origin_idx=3, price=108.0)

    result = add_smt_divergence(
        primary,
        partner,
        inverse=False,
        partner_name="usd_jpy",
        confirm_window=0,
    )

    assert result.loc[4, "xasset_smt_usd_jpy_bear_flag"] == 1
    assert result.loc[4, "xasset_smt_usd_jpy_bull_flag"] == 0
    assert result.loc[4, "xasset_smt_usd_jpy_dir"] == -1
    assert 0.0 <= result.loc[4, "xasset_smt_usd_jpy_score"] <= 1.0


def test_no_signal_emits_before_confirmation_window_matches() -> None:
    primary = _base_processed()
    partner = _base_processed()
    _set_low_event(primary, detect_idx=2, origin_idx=1, price=100.0)
    _set_low_event(primary, detect_idx=4, origin_idx=3, price=90.0)
    _set_low_event(partner, detect_idx=2, origin_idx=1, price=101.0)
    _set_low_event(partner, detect_idx=6, origin_idx=5, price=102.0)

    result = add_smt_divergence(
        primary,
        partner,
        inverse=False,
        partner_name="usd_jpy",
        confirm_window=1,
    )

    assert int(result["xasset_smt_usd_jpy_bull_flag"].sum()) == 0
    assert int(result["xasset_smt_usd_jpy_bear_flag"].sum()) == 0
    assert result["xasset_smt_usd_jpy_score"].dropna().empty


def test_add_smt_divergence_is_pure_and_dtype_stable() -> None:
    primary = _base_processed()
    partner = _base_processed()
    primary_before = primary.copy(deep=True)
    partner_before = partner.copy(deep=True)

    result = add_smt_divergence(
        primary,
        partner,
        inverse=True,
        partner_name="dxy",
    )

    pd.testing.assert_frame_equal(primary, primary_before)
    pd.testing.assert_frame_equal(partner, partner_before)
    assert pd.api.types.is_integer_dtype(result["xasset_smt_dxy_bull_flag"])
    assert pd.api.types.is_integer_dtype(result["xasset_smt_dxy_bear_flag"])
    assert pd.api.types.is_integer_dtype(result["xasset_smt_dxy_dir"])
    assert pd.api.types.is_float_dtype(result["xasset_smt_dxy_score"])
