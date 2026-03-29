from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.smc.equal_hl import add_equal_hl


def _make_eqhl_df(
    rows: list[tuple[float, float, float, float]],
    *,
    sh_conf: list[int] | None = None,
    sl_conf: list[int] | None = None,
    sh_price: list[float | None] | None = None,
    sl_price: list[float | None] | None = None,
    sh_origin: list[int | None] | None = None,
    sl_origin: list[int | None] | None = None,
    sh_strength: list[float | None] | None = None,
    sl_strength: list[float | None] | None = None,
    atr_values: list[float] | None = None,
) -> pd.DataFrame:
    n = len(rows)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["atr_14"] = np.array(
        atr_values if atr_values is not None else [1.0] * n,
        dtype=float,
    )
    df["swing_high_confirm_flag"] = np.array(sh_conf or [0] * n, dtype=np.int8)
    df["swing_low_confirm_flag"] = np.array(sl_conf or [0] * n, dtype=np.int8)
    df["swing_high_confirm_price"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_price or [None] * n)],
        dtype=float,
    )
    df["swing_low_confirm_price"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_price or [None] * n)],
        dtype=float,
    )
    df["swing_high_confirm_origin_idx"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_origin or [None] * n)],
        dtype=float,
    )
    df["swing_low_confirm_origin_idx"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_origin or [None] * n)],
        dtype=float,
    )
    df["last_swing_high"] = np.nan
    df["last_swing_low"] = np.nan
    df["last_swing_high_idx"] = np.nan
    df["last_swing_low_idx"] = np.nan
    df["swing_high_strength"] = np.array(
        [np.nan if v is None else float(v) for v in (sh_strength or [None] * n)],
        dtype=float,
    )
    df["swing_low_strength"] = np.array(
        [np.nan if v is None else float(v) for v in (sl_strength or [None] * n)],
        dtype=float,
    )
    df["swing_high_prominence_atr"] = df["swing_high_strength"]
    df["swing_low_prominence_atr"] = df["swing_low_strength"]
    return df


def test_eqh_detect_occurs_on_second_touch_and_active_state_starts_on_detect() -> None:
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
            (99.9, 100.1, 99.5, 99.7),
            (99.7, 100.15, 99.4, 99.6),
        ],
        sh_conf=[0, 1, 0, 1],
        sh_price=[None, 100.0, None, 100.05],
        sh_origin=[None, 0, None, 2],
        sh_strength=[0.8, 0.8, 0.7, 0.7],
    )

    result = add_equal_hl(df)

    assert result["eqh_active"].iloc[:3].sum() == 0
    assert result.loc[3, "eqh_detect_flag"] == 1
    assert result.loc[3, "eqh_detect_idx"] == 3.0
    assert result.loc[3, "eqh_active"] == 1
    assert result.loc[3, "eqh_active_touch_count"] == 2.0
    assert np.isclose(result.loc[3, "eqh_active_level"], 100.025)
    assert np.isclose(result.loc[3, "eqh_active_low"], 100.0)
    assert np.isclose(result.loc[3, "eqh_active_high"], 100.05)
    assert np.isclose(result.loc[3, "eqh_active_width"], 0.05)
    assert result.loc[3, "equal_highs"] == 1
    assert result.loc[3, "equal_highs_active"] == 1


def test_eqh_sweep_marks_row_and_deactivates_from_next_row() -> None:
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
            (99.9, 100.1, 99.5, 99.7),
            (99.7, 100.15, 99.4, 99.6),
            (99.6, 100.30, 99.2, 99.4),
            (99.4, 99.8, 98.9, 99.1),
        ],
        sh_conf=[0, 1, 0, 1, 0, 0],
        sh_price=[None, 100.0, None, 100.05, None, None],
        sh_origin=[None, 0, None, 2, None, None],
        sh_strength=[0.8, 0.8, 0.7, 0.7, None, None],
    )

    result = add_equal_hl(df)

    assert result.loc[4, "eqh_swept_flag"] == 1
    assert result.loc[4, "eqh_swept_idx"] == 4.0
    assert result.loc[4, "eqh_active"] == 1
    assert result.loc[4, "eqh_active_swept"] == 1
    assert result.loc[4, "eqh_active_state"] == 3
    assert result.loc[5, "eqh_active"] == 0


def test_eql_detects_and_exports_low_side_cluster() -> None:
    df = _make_eqhl_df(
        [
            (100.5, 101.0, 99.7, 100.4),
            (100.4, 100.8, 99.6, 100.2),
            (100.2, 100.6, 99.8, 100.1),
            (100.1, 100.7, 99.55, 100.3),
        ],
        sl_conf=[0, 1, 0, 1],
        sl_price=[None, 99.7, None, 99.65],
        sl_origin=[None, 0, None, 2],
        sl_strength=[0.9, 0.9, 0.8, 0.8],
    )

    result = add_equal_hl(df)

    assert result.loc[3, "eql_detect_flag"] == 1
    assert result.loc[3, "eql_active"] == 1
    assert result.loc[3, "eql_active_touch_count"] == 2.0
    assert np.isclose(result.loc[3, "eql_active_level"], 99.675)
    assert result.loc[3, "equal_lows"] == 1
    assert result.loc[3, "equal_lows_active"] == 1


def test_incompatible_touch_starts_new_cluster_and_later_two_active_clusters_can_coexist() -> (
    None
):
    df = _make_eqhl_df(
        [
            (101.0, 101.2, 100.5, 100.9),
            (100.9, 101.1, 100.4, 100.8),
            (100.8, 101.0, 100.2, 100.4),
            (100.4, 100.6, 99.9, 100.2),
            (100.2, 100.4, 99.7, 100.0),
            (100.0, 100.2, 99.6, 99.8),
        ],
        sh_conf=[0, 1, 0, 1, 1, 1],
        sh_price=[None, 101.0, None, 101.05, 100.20, 100.25],
        sh_origin=[None, 0, None, 2, 4, 5],
        sh_strength=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9],
    )

    result = add_equal_hl(df)

    assert result.loc[4, "eqh_detect_flag"] == 0
    assert result.loc[5, "eqh_detect_flag"] == 1
    assert result.loc[5, "eqh_active_count"] == 2
    assert np.isclose(result.loc[5, "eqh_active_level"], 100.225)


def test_selector_prefers_nearest_cluster_to_current_close() -> None:
    df = _make_eqhl_df(
        [
            (101.0, 101.2, 100.5, 100.9),
            (100.9, 101.1, 100.4, 100.8),
            (100.8, 101.0, 100.2, 100.4),
            (100.4, 100.6, 99.9, 100.2),
            (100.2, 100.4, 99.7, 100.0),
            (100.0, 100.2, 99.6, 100.18),
            (100.18, 100.22, 99.8, 100.20),
        ],
        sh_conf=[0, 1, 0, 1, 1, 1, 0],
        sh_price=[None, 101.0, None, 101.05, 100.20, 100.25, None],
        sh_origin=[None, 0, None, 2, 4, 5, None],
        sh_strength=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9, None],
    )

    result = add_equal_hl(df)

    assert result.loc[6, "eqh_active_count"] == 2
    assert np.isclose(result.loc[6, "eqh_active_level"], 100.225)
    assert result.loc[6, "eqh_active_id"] == result.loc[5, "eqh_cluster_id_on_detect"]


def test_selector_prefers_structural_score_before_tradeable_score() -> None:
    df = _make_eqhl_df(
        [
            (101.0, 101.2, 100.5, 100.9),
            (100.9, 101.1, 100.4, 100.8),
            (100.8, 101.0, 100.2, 100.4),
            (100.4, 100.6, 99.9, 100.2),
            (100.2, 100.4, 99.7, 100.0),
            (100.0, 100.2, 99.6, 99.95),
            (99.95, 101.02, 99.8, 100.22),
        ],
        sh_conf=[0, 1, 0, 1, 1, 1, 1],
        sh_price=[None, 101.0, None, 101.05, 100.20, 100.25, 101.11],
        sh_origin=[None, 0, None, 2, 4, 5, 6],
        sh_strength=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9, 0.75],
    )

    result = add_equal_hl(df)

    assert result.loc[6, "eqh_active_count"] == 2
    assert (
        result.loc[6, "eqh_rank1_active_id"]
        == result.loc[3, "eqh_cluster_id_on_detect"]
    )
    assert (
        result.loc[6, "eqh_rank2_active_id"]
        == result.loc[5, "eqh_cluster_id_on_detect"]
    )
    # Selector leads with structural_score: rank1 must have higher structural than rank2
    assert (
        result.loc[6, "eqh_rank1_active_structural_score"]
        > result.loc[6, "eqh_rank2_active_structural_score"]
    )
    # v3: structural_score is a 25% component of tradeable_live_score, so higher structural
    # also raises tradeable — they are now aligned, not independent.
    assert (
        result.loc[6, "eqh_rank1_active_tradeable_live_score"]
        >= result.loc[6, "eqh_rank2_active_tradeable_live_score"]
    )
    assert (
        result.loc[6, "eqh_rank1_active_distance_atr"]
        > result.loc[6, "eqh_rank2_active_distance_atr"]
    )


def test_cluster_expires_when_age_exceeds_cap() -> None:
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
            (99.9, 100.1, 99.5, 99.7),
            (99.7, 100.15, 99.4, 99.6),
            (99.6, 99.9, 99.1, 99.4),
            (99.4, 99.8, 98.9, 99.1),
        ],
        sh_conf=[0, 1, 0, 1, 0, 0],
        sh_price=[None, 100.0, None, 100.05, None, None],
        sh_origin=[None, 0, None, 2, None, None],
        sh_strength=[0.8, 0.8, 0.7, 0.7, None, None],
    )

    result = add_equal_hl(df, max_active_age=1)

    assert result.loc[3, "eqh_active"] == 1
    assert result.loc[4, "eqh_active"] == 1
    assert result.loc[5, "eqh_active"] == 0
    assert result.loc[5, "eqh_inactive_idx"] == 5.0


def test_merge_is_flagged_when_new_touch_matches_multiple_clusters() -> None:
    df = _make_eqhl_df(
        [
            (101.5, 101.7, 100.9, 101.3),
            (101.3, 101.5, 100.8, 101.2),
            (101.2, 101.4, 100.7, 101.0),
            (101.0, 101.2, 100.5, 100.8),
            (100.8, 100.82, 100.2, 100.5),
            (100.5, 100.70, 99.9, 100.2),
            (100.2, 100.13, 99.8, 100.0),
        ],
        sh_conf=[0, 1, 0, 1, 1, 1, 1],
        sh_price=[None, 100.80, None, 100.84, 100.10, 100.14, 100.46],
        sh_origin=[None, 0, None, 2, 4, 5, 6],
        sh_strength=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9, 0.85],
    )

    result = add_equal_hl(df, atr_tolerance=0.40, max_cluster_width_atr=1.00)

    assert result.loc[3, "eqh_detect_flag"] == 1
    assert result.loc[5, "eqh_detect_flag"] == 1
    assert result.loc[6, "eqh_cluster_merge_flag"] == 1
    assert result.loc[6, "eqh_active_count"] == 1


def test_non_selected_swept_cluster_exports_its_own_sweep_level() -> None:
    df = _make_eqhl_df(
        [
            (92.0, 93.0, 90.2, 92.5),
            (92.5, 93.0, 90.0, 92.0),
            (92.0, 94.0, 91.5, 93.5),
            (93.5, 94.0, 90.1, 93.0),
            (93.0, 104.0, 92.5, 103.0),
            (103.0, 104.0, 90.05, 102.0),
            (102.0, 111.0, 100.1, 110.0),
            (110.0, 111.0, 100.0, 109.0),
            (109.0, 110.0, 99.5, 108.0),
        ],
        sl_conf=[0, 1, 0, 1, 0, 1, 1, 1, 0],
        sl_price=[None, 90.0, None, 90.1, None, 90.05, 100.1, 100.0, None],
        sl_origin=[None, 0, None, 2, None, 4, 6, 7, None],
        sl_strength=[0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.7, 0.7, None],
    )

    result = add_equal_hl(df)

    assert result.loc[7, "eql_detect_flag"] == 1
    assert result.loc[7, "eql_cluster_id_on_detect"] != result.loc[7, "eql_active_id"]
    assert (
        result.loc[7, "eql_rank2_active_id"]
        == result.loc[7, "eql_cluster_id_on_detect"]
    )
    assert result.loc[8, "eql_swept_flag"] == 1
    assert (
        result.loc[8, "eql_swept_cluster_id"]
        == result.loc[7, "eql_cluster_id_on_detect"]
    )
    assert np.isclose(
        result.loc[8, "eql_swept_level"], result.loc[7, "eql_level_on_detect"]
    )
    assert result.loc[8, "eql_active_id"] != result.loc[8, "eql_swept_cluster_id"]


def test_detector_is_live_safe_and_contains_no_future_or_label_columns() -> None:
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
        ],
        sh_conf=[0, 1],
        sh_price=[None, 100.0],
        sh_origin=[None, 0],
    )

    result = add_equal_hl(df)

    eq_cols = [
        col for col in result.columns if col.startswith(("eqh_", "eql_", "equal_"))
    ]
    assert all(not col.startswith("label_") for col in eq_cols)
    assert all(not col.startswith("future_") for col in eq_cols)


def test_structural_score_is_age_independent_while_tradeable_score_varies_with_distance() -> (
    None
):
    # Cluster at ~100.0. Bars 4-6 move price *toward* cluster (99.0 -> 99.8),
    # so distance_atr shrinks and v2 tradeable_live_score should also shrink.
    df = _make_eqhl_df(
        [
            (99.5, 100.0, 99.0, 99.8),
            (99.8, 100.2, 99.4, 99.9),
            (99.9, 100.1, 99.5, 99.7),
            (99.7, 100.15, 99.4, 99.0),  # bar 3: detect, price far below cluster
            (99.0, 99.5, 98.8, 99.4),  # bar 4: price approaching
            (99.4, 99.7, 99.1, 99.7),  # bar 5: closer
            (99.7, 100.0, 99.5, 99.8),  # bar 6: close to cluster level
        ],
        sh_conf=[0, 1, 0, 1, 0, 0, 0],
        sh_price=[None, 100.0, None, 100.05, None, None, None],
        sh_origin=[None, 0, None, 2, None, None, None],
        sh_strength=[0.8, 0.8, 0.7, 0.7, None, None, None],
    )

    result = add_equal_hl(df)

    assert result.loc[3, "eqh_active"] == 1
    assert result.loc[6, "eqh_active"] == 1
    assert result.loc[3, "eqh_active_formation_delay"] == 2.0
    # structural_score is age-independent (no age component in v2)
    assert np.isclose(
        result.loc[3, "eqh_active_structural_score"],
        result.loc[6, "eqh_active_structural_score"],
    )
    # v2 tradeable_live_score is distance-sensitive: as price approaches the cluster
    # (distance_atr shrinks), the score decreases
    assert (
        result.loc[6, "eqh_active_tradeable_live_score"]
        < result.loc[3, "eqh_active_tradeable_live_score"]
    )
