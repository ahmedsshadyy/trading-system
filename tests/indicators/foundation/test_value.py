from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.value import (
    add_anchored_vwap,
    add_avwap_from_last_swing,
    compute_anchored_vwap,
)
from src.validation.indicators.value import validate_avwap


def _make_value_df(n: int = 40) -> pd.DataFrame:
    steps = np.arange(n, dtype=float)
    close = 2000.0 + np.cumsum(0.4 + 0.1 * np.sin(steps / 4.0))
    open_ = np.r_[close[0] - 0.2, close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = 1000.0 + steps * 10.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr_14": np.full(n, 2.0, dtype=float),
        }
    )


def _as_tick_only(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    volume_idx = columns.index("volume")
    out = df.drop(columns=["volume"]).copy()
    out.insert(volume_idx, "tickVolume", df["volume"].astype(float))
    return out


def test_compute_anchored_vwap_preserves_tick_volume_parity() -> None:
    df = _make_value_df()
    tick_df = _as_tick_only(df)

    pd.testing.assert_frame_equal(
        compute_anchored_vwap(df, anchor_idx=10),
        compute_anchored_vwap(tick_df, anchor_idx=10),
        check_dtype=False,
    )


def test_compute_anchored_vwap_has_stable_constant_price_behavior() -> None:
    df = _make_value_df(20)
    df.loc[:, ["open", "high", "low", "close"]] = 100.0

    result = compute_anchored_vwap(df, anchor_idx=5)
    active = result.loc[5:]

    assert (active["avwap"] == 100.0).all()
    assert (active["avwap_std"] == 0.0).all()
    assert (active["avwap_dev_sigma"] == 0.0).all()


def test_add_anchored_vwap_masks_until_live_activation_and_sets_metadata() -> None:
    df = _make_value_df()
    result = add_anchored_vwap(
        df,
        anchor_idx=5,
        anchor_label="confirmed_swing_high",
        anchor_class="hybrid",
        anchor_origin_idx=5,
        anchor_confirm_idx=8,
        anchor_live_from_idx=8,
    )

    assert result.loc[:7, "avwap"].isna().all()
    assert result.loc[8, "avwap_anchor_label"] == "confirmed_swing_high"
    assert result.loc[8, "avwap_anchor_class"] == "hybrid"
    assert result.loc[8, "avwap_anchor_idx"] == 5.0
    assert result.loc[8, "avwap_anchor_confirm_idx"] == 8.0
    assert result.loc[8, "avwap_anchor_live_from_idx"] == 8.0
    assert result.loc[8, "bars_since_anchor"] == 3.0


def test_add_anchored_vwap_uses_position_based_activation_masking() -> None:
    df = _make_value_df()
    df.index = pd.Index(np.arange(100, 100 + len(df)), name="row_id")

    result = add_anchored_vwap(
        df,
        anchor_idx=5,
        anchor_label="confirmed_swing_high",
        anchor_class="hybrid",
        anchor_origin_idx=5,
        anchor_confirm_idx=8,
        anchor_live_from_idx=8,
    )

    assert result.iloc[:8]["avwap"].isna().all()
    assert result.iloc[8]["avwap_anchor_label"] == "confirmed_swing_high"
    assert result.iloc[8]["avwap_anchor_live_from_idx"] == 8.0


def test_add_anchored_vwap_research_columns_are_gated_and_non_research_parity_holds() -> (
    None
):
    df = _make_value_df()
    live = add_anchored_vwap(
        df,
        anchor_idx=4,
        anchor_label="day_open",
        anchor_class="live_safe",
    )
    research = add_anchored_vwap(
        df,
        anchor_idx=4,
        anchor_label="day_open",
        anchor_class="live_safe",
        include_research_only=True,
    )

    assert not any(col.startswith("r_") for col in live.columns)
    assert "r_avwap_forward_1_return" in research.columns
    non_research_cols = [col for col in research.columns if not col.startswith("r_")]
    pd.testing.assert_frame_equal(
        live[non_research_cols],
        research[non_research_cols],
        check_dtype=False,
    )


def test_add_avwap_from_last_swing_uses_confirm_bar_activation_when_available() -> None:
    df = _make_value_df()
    df["swing_high_confirm_flag"] = 0
    df["swing_low_confirm_flag"] = 0
    df["swing_high_confirm_origin_idx"] = np.nan
    df["swing_low_confirm_origin_idx"] = np.nan
    df.loc[10, "swing_high_confirm_flag"] = 1
    df.loc[10, "swing_high_confirm_origin_idx"] = 6

    result = add_avwap_from_last_swing(df)

    assert result.loc[:9, "avwap"].isna().all()
    assert result.loc[10, "avwap_anchor_class"] == "hybrid"
    assert result.loc[10, "avwap_anchor_label"] == "last_confirmed_swing_high"
    assert result.loc[10, "avwap_anchor_idx"] == 6.0
    assert result.loc[10, "avwap_anchor_confirm_idx"] == 10.0


def test_validate_avwap_reports_core_contract_checks() -> None:
    df = _make_value_df(80)
    live = add_anchored_vwap(
        df,
        anchor_idx=20,
        anchor_label="day_open",
        anchor_class="live_safe",
    )
    research = add_anchored_vwap(
        df,
        anchor_idx=20,
        anchor_label="day_open",
        anchor_class="live_safe",
        include_research_only=True,
    )

    result = validate_avwap(
        research.tail(40),
        summary_df=research,
        live_df=live,
        source_parity_ok=True,
    )
    summary = result["summary"]

    assert summary["checks"]["required_columns_present"] is True
    assert summary["checks"]["band_order_ok"] is True
    assert summary["checks"]["std_contract_ok"] is True
    assert summary["checks"]["bars_since_anchor_ok"] is True
    assert summary["checks"]["no_values_before_live_activation"] is True
    assert summary["checks"]["first_active_row_matches_live_from"] is True
    assert summary["checks"]["live_research_parity_ok"] is True
    assert summary["checks"]["no_research_columns_in_live_ok"] is True


def test_validate_avwap_reports_anchor_family_audits() -> None:
    df = _make_value_df(80)
    live_safe = add_anchored_vwap(
        df,
        anchor_idx=20,
        anchor_label="day_open",
        anchor_class="live_safe",
    )
    hybrid = add_anchored_vwap(
        df,
        anchor_idx=15,
        anchor_label="confirmed_swing_high",
        anchor_class="hybrid",
        anchor_origin_idx=15,
        anchor_confirm_idx=18,
        anchor_live_from_idx=18,
    )

    result = validate_avwap(
        live_safe.tail(40),
        summary_df=live_safe,
        family_frames={
            "live_safe": live_safe,
            "hybrid": hybrid,
        },
        source_parity_ok=True,
    )
    audits = result["summary"]["anchor_family_audits"]
    assert "live_safe" in audits
    assert "hybrid" in audits
    assert audits["live_safe"]["checks"]["band_order_ok"] is True
    assert audits["hybrid"]["checks"]["first_active_row_matches_live_from"] is True


def test_validate_avwap_aggregates_multi_sample_family_audits() -> None:
    df = _make_value_df(100)
    hybrid_a = add_anchored_vwap(
        df,
        anchor_idx=20,
        anchor_label="confirmed_swing_low",
        anchor_class="hybrid",
        anchor_origin_idx=20,
        anchor_confirm_idx=24,
        anchor_live_from_idx=24,
    )
    hybrid_b = add_anchored_vwap(
        df,
        anchor_idx=40,
        anchor_label="confirmed_swing_high",
        anchor_class="hybrid",
        anchor_origin_idx=40,
        anchor_confirm_idx=45,
        anchor_live_from_idx=45,
    )

    result = validate_avwap(
        hybrid_b.tail(40),
        summary_df=hybrid_b,
        family_frames={"hybrid": [hybrid_a, hybrid_b]},
        source_parity_ok=True,
    )
    audit = result["summary"]["anchor_family_audits"]["hybrid"]
    assert audit["available"] is True
    assert audit["sample_count"] == 2
    assert (
        audit["active_row_count_stats"]["max"] >= audit["active_row_count_stats"]["min"]
    )
    assert audit["checks"]["first_active_row_matches_live_from"] is True


def test_add_anchored_vwap_supports_sweep_detect_to_confirm_hybrid_semantics() -> None:
    df = _make_value_df(60)
    detect_idx = 24
    confirm_idx = 27

    result = add_anchored_vwap(
        df,
        anchor_idx=detect_idx,
        anchor_label="sweep_bull_detect",
        anchor_class="hybrid",
        anchor_origin_idx=detect_idx,
        anchor_confirm_idx=confirm_idx,
        anchor_live_from_idx=confirm_idx,
    )

    assert result.iloc[:confirm_idx]["avwap"].isna().all()
    assert result.iloc[confirm_idx]["avwap_anchor_label"] == "sweep_bull_detect"
    assert result.iloc[confirm_idx]["avwap_anchor_class"] == "hybrid"
    assert result.iloc[confirm_idx]["avwap_anchor_idx"] == float(detect_idx)
    assert result.iloc[confirm_idx]["avwap_anchor_confirm_idx"] == float(confirm_idx)
