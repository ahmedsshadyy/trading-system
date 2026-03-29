from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.volume import add_volume_features
from src.validation.indicators.volume import validate_volume


def _make_volume_df(n: int = 160) -> pd.DataFrame:
    steps = np.arange(n, dtype=float)
    close = 2000.0 + np.cumsum(0.4 * np.sin(steps / 5.0) + 0.2 * np.cos(steps / 9.0))
    open_ = np.r_[close[0] - 0.3, close[:-1] + 0.05 * np.sin(steps[:-1] / 4.0)]
    high = np.maximum(open_, close) + 1.0 + (steps % 5) * 0.03
    low = np.minimum(open_, close) - 1.0 - (steps % 7) * 0.02
    volume = 1000.0 + steps * 11.0 + (steps % 9) * 5.0
    df = pd.DataFrame(
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
    return df


def _as_tick_only(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    volume_idx = columns.index("volume")
    out = df.drop(columns=["volume"]).copy()
    out.insert(volume_idx, "tickVolume", df["volume"].astype(float))
    return out


def test_add_volume_features_is_pure_and_live_research_split_is_stable() -> None:
    raw = _make_volume_df()
    original = raw.copy(deep=True)

    live = add_volume_features(raw, include_research_only=False)
    research = add_volume_features(raw, include_research_only=True)
    tick_live = add_volume_features(_as_tick_only(raw), include_research_only=False)

    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(live, tick_live, check_dtype=False)

    assert "signed_tick_pressure_blend" in live.columns
    assert "signed_tick_pressure_z" in live.columns
    assert not any(col.startswith("r_") for col in live.columns)
    assert "r_vol_forward_1_return" in research.columns
    assert "r_pressure_forward_1_return" in research.columns
    non_research_cols = [col for col in research.columns if not col.startswith("r_")]
    pd.testing.assert_frame_equal(
        live[non_research_cols],
        research[non_research_cols],
        check_dtype=False,
    )


def test_volume_warmup_contract_matches_frozen_thresholds() -> None:
    result = add_volume_features(_make_volume_df(100), include_research_only=False)

    assert result.loc[18, "vol_ratio"] != result.loc[18, "vol_ratio"]
    assert np.isfinite(result.loc[19, "vol_ratio"])
    assert result.loc[3, "vol_slope_5"] != result.loc[3, "vol_slope_5"]
    assert np.isfinite(result.loc[4, "vol_slope_5"])
    assert result.loc[98, "vol_pct_rank_100"] != result.loc[98, "vol_pct_rank_100"]
    assert np.isfinite(result.loc[99, "vol_pct_rank_100"])
    assert result.loc[0, "vol_above_1_5x"] == 0
    assert result.loc[0, "vol_extreme_pct95"] == 0


def test_volume_zero_volume_rows_do_not_produce_inf() -> None:
    raw = _make_volume_df(120)
    raw.loc[:40, "volume"] = 0.0
    raw.loc[60, "volume"] = 5000.0

    result = add_volume_features(raw)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    assert np.isfinite(numeric[~np.isnan(numeric)]).all()


def test_zero_range_rows_follow_frozen_behavior() -> None:
    raw = _make_volume_df(80)
    raw.loc[40, ["open", "high", "low", "close"]] = 2010.0

    result = add_volume_features(raw)
    row = result.loc[40]

    assert row["bar_range"] == 0.0
    assert row["bar_body_frac"] == 0.0
    assert np.isnan(row["close_pos_in_range"])
    assert row["upper_wick_ratio"] == 0.0
    assert row["lower_wick_ratio"] == 0.0
    assert row["close_strength"] == 0.0
    assert row["delta_proxy_raw"] == 0.0
    assert row["candle_delta_proxy"] == 0.0


def test_signed_delta_proxy_is_directionally_honest() -> None:
    bullish = _make_volume_df(25)
    bullish.loc[24, ["open", "low", "high", "close", "volume"]] = [
        100.0,
        99.0,
        105.0,
        104.8,
        6000.0,
    ]
    bullish_result = add_volume_features(bullish)
    assert bullish_result.loc[24, "candle_delta_proxy"] > 0

    bearish = _make_volume_df(25)
    bearish.loc[24, ["open", "low", "high", "close", "volume"]] = [
        104.0,
        99.0,
        105.0,
        99.2,
        6000.0,
    ]
    bearish_result = add_volume_features(bearish)
    assert bearish_result.loc[24, "candle_delta_proxy"] < 0

    doji = _make_volume_df(25)
    doji.loc[24, ["open", "low", "high", "close", "volume"]] = [
        102.0,
        99.0,
        105.0,
        102.0,
        6000.0,
    ]
    doji_result = add_volume_features(doji)
    assert np.isclose(doji_result.loc[24, "candle_delta_proxy"], 0.0)


def test_signed_tick_pressure_blend_uses_wick_and_body_context() -> None:
    bearish_rejection = _make_volume_df(30)
    bearish_rejection.loc[29, ["open", "low", "high", "close", "volume"]] = [
        100.0,
        99.5,
        110.0,
        101.0,
        8000.0,
    ]
    bearish_result = add_volume_features(bearish_rejection)
    assert bearish_result.loc[29, "wick_bias"] < 0
    assert bearish_result.loc[29, "signed_tick_pressure_wick"] < 0

    bullish_rejection = _make_volume_df(30)
    bullish_rejection.loc[29, ["open", "low", "high", "close", "volume"]] = [
        109.0,
        99.0,
        109.5,
        108.8,
        8000.0,
    ]
    bullish_result = add_volume_features(bullish_rejection)
    assert bullish_result.loc[29, "wick_bias"] > 0
    assert bullish_result.loc[29, "signed_tick_pressure_wick"] > 0


def test_signed_tick_pressure_blend_is_not_just_close_location_with_large_rejection_wicks() -> (
    None
):
    rejection = _make_volume_df(120)
    rejection.loc[119, ["open", "low", "high", "close", "volume"]] = [
        100.0,
        99.0,
        115.0,
        107.0,
        10000.0,
    ]

    result = add_volume_features(rejection)
    row = result.loc[119]

    assert row["signed_tick_pressure_wick"] < 0
    assert row["signed_tick_pressure_blend"] < row["candle_delta_proxy"]


def test_effort_result_and_vsa_proxies_trigger_on_intended_shapes() -> None:
    absorption = _make_volume_df(120)
    absorption.loc[119, ["open", "low", "high", "close", "volume", "atr_14"]] = [
        100.0,
        99.0,
        101.0,
        100.1,
        10000.0,
        3.0,
    ]
    absorption_result = add_volume_features(absorption)
    assert (
        absorption_result.loc[119, "effort_vs_result"]
        > absorption_result.loc[119, "result_vs_effort"]
    )
    assert absorption_result.loc[119, "vsa_absorption"] == 1

    directional = _make_volume_df(120)
    directional.loc[119, ["open", "low", "high", "close", "volume", "atr_14"]] = [
        100.0,
        99.0,
        106.0,
        105.8,
        10000.0,
        2.0,
    ]
    directional_result = add_volume_features(directional)
    assert directional_result.loc[119, "vsa_directional"] == 1
    assert directional_result.loc[119, "vsa_climactic_up"] == 1


def test_effort_result_uses_meaningful_floors_instead_of_exploding() -> None:
    tiny = _make_volume_df(120)
    tiny.loc[119, ["open", "high", "low", "close", "volume", "atr_14"]] = [
        100.0,
        100.01,
        100.0,
        100.005,
        10000.0,
        10.0,
    ]

    result = add_volume_features(tiny)
    row = result.loc[119]

    assert row["bar_range_atr"] < 0.05
    assert np.isclose(row["effective_range_atr_floor"], 0.05)
    assert row["effort_vs_result"] < 100.0


def test_wick_effort_proxies_capture_rejection_side() -> None:
    upper = _make_volume_df(120)
    upper.loc[119, ["open", "low", "high", "close", "volume"]] = [
        100.0,
        99.8,
        110.0,
        100.5,
        9000.0,
    ]
    upper_result = add_volume_features(upper)
    assert (
        upper_result.loc[119, "upper_rejection_effort"]
        > upper_result.loc[119, "lower_rejection_effort"]
    )

    lower = _make_volume_df(120)
    lower.loc[119, ["open", "low", "high", "close", "volume"]] = [
        109.0,
        99.0,
        109.2,
        108.5,
        9000.0,
    ]
    lower_result = add_volume_features(lower)
    assert (
        lower_result.loc[119, "lower_rejection_effort"]
        > lower_result.loc[119, "upper_rejection_effort"]
    )


def test_validate_volume_reports_full_frame_warmup_context_for_clipped_slice() -> None:
    live = add_volume_features(_make_volume_df(140), include_research_only=False)
    research = add_volume_features(_make_volume_df(140), include_research_only=True)
    plot_df = research.tail(40).copy()

    result = validate_volume(
        plot_df,
        summary_df=research,
        live_df=live,
        research_df=research,
        source_parity_ok=True,
    )
    summary = result["summary"]

    assert summary["display_window"]["displayed_slice_start_row"] == 100
    assert summary["display_window"]["reported_slice_is_post_warmup_only"] is True
    assert summary["warmup_first_valid_rows"]["vol_20_family_first_valid_row"] == 19
    assert summary["warmup_first_valid_rows"]["vol_pct_rank_100_first_valid_row"] == 99
    assert summary["warmup_first_valid_rows"]["vol_slope_5_first_valid_row"] == 4
    assert "pressure_relationship" in summary
    assert summary["pressure_relationship"]["valid_rows"] > 0
    assert summary["pressure_relationship"]["blend_near_zero_rate_pct"] is not None
    assert summary["pressure_relationship"]["mean_abs_body_component_share"] < 5.0
