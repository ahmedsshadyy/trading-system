from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.foundation.volume_profile import (
    add_volume_profile,
    compute_volume_profile,
)
from src.validation.indicators.volume_profile import validate_volume_profile


def _make_volume_df(n: int = 140) -> pd.DataFrame:
    steps = np.arange(n, dtype=float)
    close = 2000.0 + np.cumsum(0.3 * np.sin(steps / 4.0) + 0.15 * np.cos(steps / 8.0))
    open_ = np.r_[close[0] - 0.25, close[:-1]]
    high = np.maximum(open_, close) + 1.0 + (steps % 5) * 0.04
    low = np.minimum(open_, close) - 1.0 - (steps % 6) * 0.03
    volume = 1000.0 + steps * 7.0 + (steps % 10) * 11.0
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr_14": np.full(n, 2.5, dtype=float),
        }
    )


def _as_tick_only(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    volume_idx = columns.index("volume")
    out = df.drop(columns=["volume"]).copy()
    out.insert(volume_idx, "tickVolume", df["volume"].astype(float))
    return out


def test_compute_volume_profile_allocates_expected_total_volume() -> None:
    df = _make_volume_df(100)
    vp = compute_volume_profile(df, lookback=80, n_bins=24)

    assert np.isclose(vp["profile"].sum(), df.tail(80)["volume"].sum())
    assert vp["val"] <= vp["poc"] <= vp["vah"]


def test_add_volume_profile_excludes_current_bar_in_exact_mode() -> None:
    base = _make_volume_df(120)
    changed = base.copy()
    changed.loc[80, ["low", "high", "close", "volume"]] = [
        1500.0,
        2500.0,
        2499.0,
        500000.0,
    ]

    base_result = add_volume_profile(base, lookback=80, n_bins=20, mode="exact")
    changed_result = add_volume_profile(changed, lookback=80, n_bins=20, mode="exact")

    assert np.isclose(
        base_result.loc[80, "vp_poc"], changed_result.loc[80, "vp_poc"], equal_nan=True
    )
    assert not np.isclose(
        base_result.loc[81, "vp_poc"], changed_result.loc[81, "vp_poc"], equal_nan=True
    )


def test_volume_profile_is_deterministic_and_volume_parity_safe() -> None:
    df = _make_volume_df()

    exact_once = add_volume_profile(df, lookback=80, n_bins=24, mode="exact")
    exact_twice = add_volume_profile(df, lookback=80, n_bins=24, mode="exact")
    tick_exact = add_volume_profile(
        _as_tick_only(df), lookback=80, n_bins=24, mode="exact"
    )

    pd.testing.assert_frame_equal(exact_once, exact_twice, check_dtype=False)
    pd.testing.assert_frame_equal(exact_once, tick_exact, check_dtype=False)


def test_volume_profile_context_columns_are_internally_consistent() -> None:
    result = add_volume_profile(_make_volume_df(), lookback=80, n_bins=24, mode="exact")
    valid = result["vp_poc"].notna()

    assert (result.loc[valid, "vp_value_width"] >= 0).all()
    assert (
        result.loc[valid, "vp_inside_value_area"]
        + result.loc[valid, "vp_above_vah"]
        + result.loc[valid, "vp_below_val"]
        == 1
    ).all()


def test_volume_profile_validator_reports_full_audit_and_warmup_contract() -> None:
    df = _make_volume_df(140)
    exact = add_volume_profile(df, lookback=80, n_bins=24, mode="exact")
    stepped = add_volume_profile(df, lookback=80, n_bins=24, mode="stepped")

    result = validate_volume_profile(
        exact.tail(60),
        summary_df=exact,
        stepped_df=stepped,
        lookback=80,
        n_bins=24,
    )
    summary = result["summary"]

    assert summary["mode"] == "exact"
    assert summary["approximation_audit"]["canonical_mode"] == "exact"
    assert summary["approximation_audit"]["stepped_mode_role"] == "approximation_only"
    assert summary["approximation_audit"]["stepped_mode_parity_equivalent"] is False
    assert summary["checks"]["warmup_count_ok"] is True
    assert summary["checks"]["first_valid_row_matches_expected"] is True
    assert summary["checks"]["no_premature_vp_values_before_lookback"] is True
    assert summary["checks"]["full_current_bar_exclusion_ok"] is True
    assert summary["checks"]["full_allocated_volume_ok"] is True
    assert summary["checks"]["full_value_area_coverage_ok"] is True
    assert summary["checks"]["no_inf_in_vp_columns"] is True
    assert summary["full_window_audit"]["audited_row_count"] == len(df) - 80
    assert summary["approximation_audit"]["comparison_performed"] is True
    assert "vp_value_width_atr" in summary["distribution_summary"]["summary_stats"]
