from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.indicators.swing_post_confirm import (
    build_swing_post_confirm_diagnostics,
    build_swing_post_confirm_events,
    swing_confirmation_contract,
)


def _base_frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    n = len(rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "atr_14": np.full(n, 1.0, dtype=float),
            "regime_label": np.array(["TRENDING"] * n, dtype=object),
            "session_name": np.array(["London"] * n, dtype=object),
            "swing_high_confirm_flag": np.zeros(n, dtype=int),
            "swing_low_confirm_flag": np.zeros(n, dtype=int),
            "swing_high_confirm_origin_idx": np.full(n, np.nan, dtype=float),
            "swing_low_confirm_origin_idx": np.full(n, np.nan, dtype=float),
            "swing_high_confirm_price": np.full(n, np.nan, dtype=float),
            "swing_low_confirm_price": np.full(n, np.nan, dtype=float),
            "swing_high_strength": np.full(n, np.nan, dtype=float),
            "swing_low_strength": np.full(n, np.nan, dtype=float),
        }
    )


def test_swing_post_confirm_anchors_from_confirm_idx_not_origin_idx() -> None:
    df = _base_frame(
        [
            (99.0, 100.0, 98.8, 99.5),
            (100.0, 103.0, 99.5, 102.5),  # origin swing high
            (102.0, 102.4, 101.0, 101.5),
            (101.2, 101.4, 99.8, 100.5),  # confirm bar
            (100.4, 100.6, 98.8, 99.0),  # favorable bar after confirm
            (99.0, 99.2, 98.6, 98.8),
            (98.8, 99.0, 98.5, 98.7),
            (98.7, 98.9, 98.4, 98.6),
            (98.6, 98.8, 98.3, 98.5),
        ]
    )
    df.loc[3, "swing_high_confirm_flag"] = 1
    df.loc[3, "swing_high_confirm_origin_idx"] = 1.0
    df.loc[3, "swing_high_confirm_price"] = 103.0
    df.loc[1, "swing_high_strength"] = 1.2

    events = build_swing_post_confirm_events(df)
    assert len(events) == 1
    event = events.iloc[0]
    assert int(event["swing_idx"]) == 1
    assert int(event["confirm_idx"]) == 3
    assert float(event["confirmation_latency_bars"]) == 2.0
    assert event["fwd_close_ret_atr_1"] == 1.5
    assert event["fwd_path_label_1"] == "clean_reversal"
    assert event["first_favorable_1p0_bar"] == 1.0


def test_swing_post_confirm_direction_flips_for_high_and_low() -> None:
    high_df = _base_frame(
        [
            (99.0, 100.0, 98.8, 99.5),
            (100.0, 103.0, 99.5, 102.5),
            (102.0, 102.4, 101.0, 101.5),
            (101.2, 101.4, 99.8, 100.5),
            (100.4, 100.6, 98.8, 99.0),
            (99.0, 99.2, 98.6, 98.8),
            (98.8, 99.0, 98.5, 98.7),
            (98.7, 98.9, 98.4, 98.6),
            (98.6, 98.8, 98.3, 98.5),
        ]
    )
    high_df.loc[3, "swing_high_confirm_flag"] = 1
    high_df.loc[3, "swing_high_confirm_origin_idx"] = 1.0
    high_df.loc[3, "swing_high_confirm_price"] = 103.0

    low_df = _base_frame(
        [
            (101.0, 101.2, 100.8, 101.0),
            (100.5, 100.7, 98.0, 98.4),
            (98.6, 99.3, 98.5, 99.1),
            (99.0, 100.2, 98.9, 99.8),
            (99.9, 101.5, 99.8, 101.1),
            (101.0, 101.2, 100.7, 101.0),
            (100.9, 101.1, 100.6, 100.8),
            (100.7, 101.0, 100.5, 100.7),
            (100.6, 100.9, 100.4, 100.6),
        ]
    )
    low_df.loc[3, "swing_low_confirm_flag"] = 1
    low_df.loc[3, "swing_low_confirm_origin_idx"] = 1.0
    low_df.loc[3, "swing_low_confirm_price"] = 98.0

    high_events = build_swing_post_confirm_events(high_df)
    low_events = build_swing_post_confirm_events(low_df)
    assert high_events["fwd_close_ret_atr_1"].iloc[0] > 0.0
    assert low_events["fwd_close_ret_atr_1"].iloc[0] > 0.0


def test_swing_post_confirm_handles_insufficient_future_bars_with_nan() -> None:
    df = _base_frame(
        [
            (99.0, 100.0, 98.8, 99.5),
            (100.0, 103.0, 99.5, 102.5),
            (102.0, 102.4, 101.0, 101.5),
            (101.2, 101.4, 99.8, 100.5),
            (100.4, 100.6, 98.8, 99.0),
        ]
    )
    df.loc[3, "swing_high_confirm_flag"] = 1
    df.loc[3, "swing_high_confirm_origin_idx"] = 1.0
    df.loc[3, "swing_high_confirm_price"] = 103.0

    event = build_swing_post_confirm_events(df).iloc[0]
    assert np.isnan(event["fwd_close_ret_atr_2"])
    assert np.isnan(event["first_favorable_1p0_bar"])
    assert np.isnan(event["swing_reversed_by_5"])


def test_swing_post_confirm_diagnostics_and_contract_exist() -> None:
    df = _base_frame(
        [
            (99.0, 100.0, 98.8, 99.5),
            (100.0, 103.0, 99.5, 102.5),
            (102.0, 102.4, 101.0, 101.5),
            (101.2, 101.4, 99.8, 100.5),
            (100.4, 100.6, 98.8, 99.0),
            (99.0, 99.2, 98.6, 98.8),
            (98.8, 99.0, 98.5, 98.7),
            (98.7, 98.9, 98.4, 98.6),
            (98.6, 98.8, 98.3, 98.5),
        ]
    )
    df.loc[3, "swing_high_confirm_flag"] = 1
    df.loc[3, "swing_high_confirm_origin_idx"] = 1.0
    df.loc[3, "swing_high_confirm_price"] = 103.0
    df.loc[1, "swing_high_strength"] = 1.2

    contract = swing_confirmation_contract()
    assert contract["active_from_confirm_idx_only"].startswith("Yes.")

    diagnostics = build_swing_post_confirm_diagnostics(df)
    assert diagnostics["summary"]["total_confirmed_swings"] == 1
    assert not diagnostics["by_side"].empty
    assert not diagnostics["by_latency"].empty
    assert not diagnostics["by_regime"].empty
    assert not diagnostics["by_session"].empty
    assert not diagnostics["by_distance"].empty
    assert "Step 11S Swing Post-Confirmation Edge Audit" in diagnostics["memo_markdown"]
