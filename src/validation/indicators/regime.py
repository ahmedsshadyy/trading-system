from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.validation.common.chart_core import save_figure_html

REQUIRED_REGIME_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adx_strength",
    "compression_score",
    "ema_slope_strength",
    "structure_continuity",
    "trend_confidence_norm",
    "neutral_structure_penalty",
    "regime_input_ready",
    "trend_regime_score",
    "range_regime_score",
    "transition_regime_score",
    "raw_regime",
    "raw_regime_label",
    "raw_regime_confidence",
    "raw_regime_margin",
    "regime",
    "regime_label",
    "regime_confidence",
    "regime_margin",
    "regime_is_ranging",
    "regime_is_transitional",
    "regime_is_trending",
    "regime_prev",
    "regime_changed",
    "regime_enter_ranging",
    "regime_enter_transitional",
    "regime_enter_trending",
    "bars_in_regime",
    "regime_persistence_5",
    "regime_persistence_20",
    "regime_trend_alignment",
    "regime_bias_alignment",
    "regime_strength_bucket",
    "regime_boundary_flag",
    "regime_context_caution",
    "regime_stabilized_from_raw",
    "regime_forced_transitional",
    "regime_direct_extreme_jump",
    "trend_state",
    "trend_confidence",
    "adx_14",
    "bb_width",
}

SUMMARY_COLUMNS = [
    "trend_regime_score",
    "range_regime_score",
    "transition_regime_score",
    "regime_confidence",
    "regime_margin",
]

REGIME_NAMES = {
    0: "RANGING",
    1: "TRANSITIONAL",
    2: "TRENDING",
}

TREND_STATE_NAMES = {
    -1: "BEARISH",
    0: "NEUTRAL",
    1: "BULLISH",
}

REGIME_BACKGROUND = {
    0: "rgba(65, 90, 119, 0.10)",
    1: "rgba(230, 184, 0, 0.10)",
    2: "rgba(27, 153, 139, 0.10)",
}

MISALIGNMENT_PROFILE_COLUMNS = [
    "regime_confidence",
    "regime_margin",
    "bars_in_regime",
    "adx_strength",
    "compression_score",
    "ema_slope_strength",
    "structure_continuity",
    "trend_confidence_norm",
]


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _no_inf_ok(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    values = numeric.to_numpy(dtype=float)
    return bool(np.isfinite(values[~np.isnan(values)]).all())


def _valid_regime_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["regime"], errors="coerce").notna()


def _first_valid_row(series: pd.Series) -> int | None:
    mask = pd.to_numeric(series, errors="coerce").notna().to_numpy()
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def _score_bounds_ok(df: pd.DataFrame) -> bool:
    for col in SUMMARY_COLUMNS + ["raw_regime_confidence", "raw_regime_margin"]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not bool(values.ge(0).all() and values.le(1).all()):
            return False
    return True


def _state_contract_ok(df: pd.DataFrame) -> bool:
    regime = pd.to_numeric(df["regime"], errors="coerce").dropna()
    raw_regime = pd.to_numeric(df["raw_regime"], errors="coerce").dropna()
    return bool(
        set(regime.astype(int).unique()).issubset({0, 1, 2})
        and set(raw_regime.astype(int).unique()).issubset({0, 1, 2})
    )


def _one_hot_ok(df: pd.DataFrame) -> bool:
    flags = (
        df[["regime_is_ranging", "regime_is_transitional", "regime_is_trending"]]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )
    valid = _valid_regime_mask(df)
    return bool(
        flags.loc[valid].sum(axis=1).eq(1).all()
        and flags.loc[~valid].sum(axis=1).eq(0).all()
    )


def _warmup_contract(df: pd.DataFrame) -> dict[str, object]:
    ready = pd.to_numeric(df["regime_input_ready"], errors="coerce").fillna(0).eq(1)
    regime = pd.to_numeric(df["regime"], errors="coerce")
    ready_positions = np.flatnonzero(ready.to_numpy())
    first_ready = int(ready_positions[0]) if len(ready_positions) else None
    first_valid = _first_valid_row(regime)
    if first_ready is None:
        no_premature = bool(regime.isna().all())
    else:
        no_premature = bool(regime.iloc[:first_ready].isna().all())
    return {
        "first_ready_row": first_ready,
        "first_valid_regime_row": first_valid,
        "warmup_nan_rows": int(regime.isna().sum()),
        "no_premature_regime_before_ready": no_premature,
        "first_valid_matches_first_ready": first_ready == first_valid,
    }


def _transition_contract_ok(df: pd.DataFrame) -> bool:
    regime = pd.to_numeric(df["regime"], errors="coerce").astype(float)
    prev = pd.to_numeric(df["regime_prev"], errors="coerce")
    changed = pd.to_numeric(df["regime_changed"], errors="coerce").fillna(0).astype(int)
    bars_in_regime = (
        pd.to_numeric(df["bars_in_regime"], errors="coerce").fillna(0).astype(int)
    )
    enter_cols = [
        pd.to_numeric(df["regime_enter_ranging"], errors="coerce")
        .fillna(0)
        .astype(int),
        pd.to_numeric(df["regime_enter_transitional"], errors="coerce")
        .fillna(0)
        .astype(int),
        pd.to_numeric(df["regime_enter_trending"], errors="coerce")
        .fillna(0)
        .astype(int),
    ]
    enter_sum = sum(enter_cols)

    for i in range(len(df)):
        if pd.isna(regime.iloc[i]):
            if (
                bars_in_regime.iloc[i] != 0
                or changed.iloc[i] != 0
                or enter_sum.iloc[i] != 0
            ):
                return False
            continue
        if i == 0 or pd.isna(prev.iloc[i]):
            if (
                changed.iloc[i] != 0
                or bars_in_regime.iloc[i] != 1
                or enter_sum.iloc[i] != 1
            ):
                return False
            continue
        expected_changed = int(regime.iloc[i] != prev.iloc[i])
        if changed.iloc[i] != expected_changed:
            return False
        if expected_changed == 1 and (
            bars_in_regime.iloc[i] != 1 or enter_sum.iloc[i] != 1
        ):
            return False
        if expected_changed == 0 and (
            bars_in_regime.iloc[i] < 2 or enter_sum.iloc[i] != 0
        ):
            return False
    return True


def _persistence_bounds_ok(df: pd.DataFrame) -> bool:
    for col in ("regime_persistence_5", "regime_persistence_20"):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not bool(values.ge(0).all() and values.le(1).all()):
            return False
    return True


def _no_label_contamination_ok(df: pd.DataFrame) -> bool:
    return not any(col.startswith(("r_", "future_", "label_")) for col in df.columns)


def _regime_counts(df: pd.DataFrame, *, regime_col: str = "regime") -> dict[str, int]:
    regime = pd.to_numeric(df[regime_col], errors="coerce")
    return {name: int(regime.eq(code).sum()) for code, name in REGIME_NAMES.items()}


def _segment_frame(df: pd.DataFrame, *, regime_col: str = "regime") -> pd.DataFrame:
    regime = pd.to_numeric(df[regime_col], errors="coerce")
    valid = regime.notna()
    if not bool(valid.any()):
        return pd.DataFrame(columns=["group", "regime", "dwell"])
    groups = (
        (regime != regime.shift(1)) | regime.isna() | regime.shift(1).isna()
    ).cumsum()
    return (
        pd.DataFrame({"regime": regime, "group": groups})
        .loc[valid]
        .groupby("group")
        .agg(regime=("regime", "first"), dwell=("regime", "size"))
        .reset_index()
    )


def _regime_change_count(df: pd.DataFrame) -> int:
    return int(pd.to_numeric(df["regime_changed"], errors="coerce").fillna(0).sum())


def _dwell_stats(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    segment_rows = _segment_frame(df)
    out: dict[str, dict[str, float | int | None]] = {}
    for code, label in REGIME_NAMES.items():
        scoped = segment_rows.loc[segment_rows["regime"].eq(code), "dwell"]
        if scoped.empty:
            out[label] = {"count": 0, "mean": None, "median": None}
        else:
            out[label] = {
                "count": int(scoped.size),
                "mean": float(scoped.mean()),
                "median": float(scoped.median()),
            }
    return out


def _alignment_rates(df: pd.DataFrame) -> dict[str, float | None]:
    valid = _valid_regime_mask(df)
    if not bool(valid.any()):
        return {"trend_alignment_rate_pct": None, "bias_alignment_rate_pct": None}
    trend_alignment = pd.to_numeric(
        df.loc[valid, "regime_trend_alignment"], errors="coerce"
    )
    bias_alignment = pd.to_numeric(
        df.loc[valid, "regime_bias_alignment"], errors="coerce"
    )
    return {
        "trend_alignment_rate_pct": (
            float(trend_alignment.mean() * 100.0)
            if trend_alignment.notna().any()
            else None
        ),
        "bias_alignment_rate_pct": (
            float(bias_alignment.mean() * 100.0)
            if bias_alignment.notna().any()
            else None
        ),
    }


def _per_regime_alignment(df: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    valid = _valid_regime_mask(df)
    out: dict[str, dict[str, float | int | None]] = {}
    for code, label in REGIME_NAMES.items():
        scoped = df.loc[valid & pd.to_numeric(df["regime"], errors="coerce").eq(code)]
        if scoped.empty:
            out[label] = {
                "rows": 0,
                "trend_alignment_rate_pct": None,
                "bias_alignment_rate_pct": None,
            }
            continue
        trend_alignment = pd.to_numeric(
            scoped["regime_trend_alignment"], errors="coerce"
        )
        bias_alignment = pd.to_numeric(scoped["regime_bias_alignment"], errors="coerce")
        out[label] = {
            "rows": int(len(scoped)),
            "trend_alignment_rate_pct": (
                float(trend_alignment.mean() * 100.0)
                if trend_alignment.notna().any()
                else None
            ),
            "bias_alignment_rate_pct": (
                float(bias_alignment.mean() * 100.0)
                if bias_alignment.notna().any()
                else None
            ),
        }
    return out


def _confusion_matrix(
    df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    row_labels: dict[int, str],
    col_labels: dict[int, str],
) -> dict[str, dict[str, int]] | None:
    rows = pd.to_numeric(df[row_col], errors="coerce")
    cols = pd.to_numeric(df[col_col], errors="coerce")
    valid = rows.notna() & cols.notna()
    if not bool(valid.any()):
        return None
    grouped = (
        pd.DataFrame({"row": rows, "col": cols})
        .loc[valid]
        .groupby(["row", "col"])
        .size()
    )
    out: dict[str, dict[str, int]] = {
        row_labels[row_key]: {col_labels[col_key]: 0 for col_key in col_labels}
        for row_key in row_labels
    }
    for (row_value, col_value), count in grouped.items():
        out[row_labels[int(row_value)]][col_labels[int(col_value)]] = int(count)
    return out


def _extreme_consistency(df: pd.DataFrame) -> dict[str, float | int | None]:
    regime = pd.to_numeric(df["regime"], errors="coerce")
    trend_state = pd.to_numeric(df["trend_state"], errors="coerce")
    trending = regime.eq(2)
    ranging = regime.eq(0)
    return {
        "trending_rows": int(trending.sum()),
        "ranging_rows": int(ranging.sum()),
        "trending_with_non_neutral_trend_rate_pct": (
            float((trend_state.loc[trending].ne(0)).mean() * 100.0)
            if bool(trending.any())
            else None
        ),
        "ranging_with_neutral_trend_rate_pct": (
            float((trend_state.loc[ranging].eq(0)).mean() * 100.0)
            if bool(ranging.any())
            else None
        ),
    }


def _unaligned_decomposition(df: pd.DataFrame) -> dict[str, int]:
    regime = pd.to_numeric(df["regime"], errors="coerce")
    trend_state = pd.to_numeric(df["trend_state"], errors="coerce")
    bias = (
        pd.to_numeric(df["trend_bias_state"], errors="coerce")
        if "trend_bias_state" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    return {
        "trending_regime_with_neutral_trend_state": int(
            (regime.eq(2) & trend_state.eq(0)).sum()
        ),
        "ranging_regime_with_directional_trend_state": int(
            (regime.eq(0) & trend_state.ne(0) & trend_state.notna()).sum()
        ),
        "transitional_regime_with_directional_trend_state": int(
            (regime.eq(1) & trend_state.ne(0) & trend_state.notna()).sum()
        ),
        "transitional_regime_with_directional_bias_state": int(
            (regime.eq(1) & bias.ne(0) & bias.notna()).sum()
        ),
    }


def _extreme_misalignment_audit(df: pd.DataFrame) -> dict[str, float | int | None]:
    regime = pd.to_numeric(df["regime"], errors="coerce")
    trend_state = pd.to_numeric(df["trend_state"], errors="coerce")
    ranging = regime.eq(0)
    trending = regime.eq(2)

    trending_neutral = int((trending & trend_state.eq(0)).sum())
    ranging_directional = int((ranging & trend_state.ne(0) & trend_state.notna()).sum())
    trending_rows = int(trending.sum())
    ranging_rows = int(ranging.sum())

    return {
        "trending_rows": trending_rows,
        "ranging_rows": ranging_rows,
        "trending_with_neutral_trend_state_count": trending_neutral,
        "trending_with_neutral_trend_state_rate_pct": (
            float(trending_neutral / trending_rows * 100.0) if trending_rows else None
        ),
        "ranging_with_directional_trend_state_count": ranging_directional,
        "ranging_with_directional_trend_state_rate_pct": (
            float(ranging_directional / ranging_rows * 100.0) if ranging_rows else None
        ),
    }


def _subset_profile(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    scoped = df.loc[mask].copy()
    if scoped.empty:
        return {
            "rows": 0,
            "session_distribution": None,
            "trend_state_distribution": None,
            "bias_state_distribution": None,
            "current_score_profile": {},
        }

    session_distribution = None
    if "session_name" in scoped.columns:
        session_distribution = {
            str(key): int(value)
            for key, value in scoped["session_name"].value_counts(dropna=False).items()
        }

    trend_state_distribution = {
        TREND_STATE_NAMES.get(int(key), str(int(key))): int(value)
        for key, value in pd.to_numeric(scoped["trend_state"], errors="coerce")
        .value_counts(dropna=True)
        .sort_index()
        .items()
    }

    bias_state_distribution = None
    if "trend_bias_state" in scoped.columns:
        bias_state_distribution = {
            TREND_STATE_NAMES.get(int(key), str(int(key))): int(value)
            for key, value in pd.to_numeric(scoped["trend_bias_state"], errors="coerce")
            .value_counts(dropna=True)
            .sort_index()
            .items()
        }

    current_score_profile = {
        col: {
            "mean": (
                float(pd.to_numeric(scoped[col], errors="coerce").dropna().mean())
                if pd.to_numeric(scoped[col], errors="coerce").dropna().size
                else None
            ),
            "median": (
                float(pd.to_numeric(scoped[col], errors="coerce").dropna().median())
                if pd.to_numeric(scoped[col], errors="coerce").dropna().size
                else None
            ),
        }
        for col in MISALIGNMENT_PROFILE_COLUMNS
    }
    current_score_profile["boundary_flag_rate_pct"] = float(
        pd.to_numeric(scoped["regime_boundary_flag"], errors="coerce").fillna(0).mean()
        * 100.0
    )
    current_score_profile["context_caution_rate_pct"] = float(
        pd.to_numeric(scoped["regime_context_caution"], errors="coerce")
        .fillna(0)
        .mean()
        * 100.0
    )

    return {
        "rows": int(len(scoped)),
        "session_distribution": session_distribution,
        "trend_state_distribution": trend_state_distribution,
        "bias_state_distribution": bias_state_distribution,
        "current_score_profile": current_score_profile,
    }


def _extreme_misalignment_profiles(df: pd.DataFrame) -> dict[str, object]:
    regime = pd.to_numeric(df["regime"], errors="coerce")
    trend_state = pd.to_numeric(df["trend_state"], errors="coerce")
    trending_neutral_mask = regime.eq(2) & trend_state.eq(0)
    ranging_directional_mask = regime.eq(0) & trend_state.ne(0) & trend_state.notna()

    return {
        "trending_with_neutral_trend_state": _subset_profile(df, trending_neutral_mask),
        "ranging_with_directional_trend_state": _subset_profile(
            df, ranging_directional_mask
        ),
    }


def _current_regime_snapshot(df: pd.DataFrame) -> dict[str, object] | None:
    valid = df.loc[_valid_regime_mask(df)]
    if valid.empty:
        return None
    row = valid.iloc[-1]
    regime_value = int(
        pd.to_numeric(pd.Series([row["regime"]]), errors="coerce").iloc[0]
    )
    snapshot = {
        "timestamp": str(row["timestamp"]),
        "regime": REGIME_NAMES[regime_value],
        "regime_confidence": float(
            pd.to_numeric(pd.Series([row["regime_confidence"]]), errors="coerce").iloc[
                0
            ]
        ),
        "regime_margin": float(
            pd.to_numeric(pd.Series([row["regime_margin"]]), errors="coerce").iloc[0]
        ),
        "regime_boundary_flag": int(
            pd.to_numeric(pd.Series([row["regime_boundary_flag"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        ),
        "regime_context_caution": int(
            pd.to_numeric(pd.Series([row["regime_context_caution"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        ),
        "bars_in_regime": int(
            pd.to_numeric(pd.Series([row["bars_in_regime"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        ),
        "trend_state": TREND_STATE_NAMES.get(
            int(
                pd.to_numeric(pd.Series([row["trend_state"]]), errors="coerce")
                .fillna(0)
                .iloc[0]
            ),
            str(row["trend_state"]),
        ),
    }
    return snapshot


def _boundary_diagnostics(df: pd.DataFrame) -> dict[str, object]:
    valid = _valid_regime_mask(df)
    if not bool(valid.any()):
        return {
            "boundary_flag_rate_pct": None,
            "confidence_bucket_counts": {},
            "low_margin_transition_rate_pct": None,
            "context_caution_rate_pct": None,
            "caution_source_breakdown": {},
            "caution_overlap_counts": {},
        }
    boundary = (
        pd.to_numeric(df.loc[valid, "regime_boundary_flag"], errors="coerce")
        .fillna(0)
        .eq(1)
    )
    changed = (
        pd.to_numeric(df.loc[valid, "regime_changed"], errors="coerce").fillna(0).eq(1)
    )
    low_margin = pd.to_numeric(df.loc[valid, "regime_margin"], errors="coerce").lt(0.10)
    buckets = pd.to_numeric(df.loc[valid, "regime_strength_bucket"], errors="coerce")
    confidence_low = pd.to_numeric(
        df.loc[valid, "regime_confidence"], errors="coerce"
    ).lt(0.60)
    early_bars = pd.to_numeric(df.loc[valid, "bars_in_regime"], errors="coerce").le(2)
    context_caution = (
        pd.to_numeric(df.loc[valid, "regime_context_caution"], errors="coerce")
        .fillna(0)
        .eq(1)
    )
    bucket_counts = {
        str(int(k)): int(v)
        for k, v in buckets.value_counts(dropna=True).sort_index().items()
    }
    transition_rows = changed.sum()
    if transition_rows:
        low_margin_transition_rate = float(
            (low_margin & changed).sum() / transition_rows * 100.0
        )
    else:
        low_margin_transition_rate = None
    caution_source_breakdown = {
        "boundary_flag_rate_pct": float(boundary.mean() * 100.0),
        "confidence_below_0_60_rate_pct": float(confidence_low.mean() * 100.0),
        "bars_in_regime_le_2_rate_pct": float(early_bars.mean() * 100.0),
    }
    caution_overlap_counts = {
        "boundary_only": int((boundary & ~confidence_low & ~early_bars).sum()),
        "confidence_only": int((~boundary & confidence_low & ~early_bars).sum()),
        "bars_only": int((~boundary & ~confidence_low & early_bars).sum()),
        "boundary_and_confidence_only": int(
            (boundary & confidence_low & ~early_bars).sum()
        ),
        "boundary_and_bars_only": int((boundary & ~confidence_low & early_bars).sum()),
        "confidence_and_bars_only": int(
            (~boundary & confidence_low & early_bars).sum()
        ),
        "all_three": int((boundary & confidence_low & early_bars).sum()),
        "none": int((~boundary & ~confidence_low & ~early_bars).sum()),
    }
    return {
        "boundary_flag_rate_pct": float(boundary.mean() * 100.0),
        "confidence_bucket_counts": bucket_counts,
        "low_margin_transition_rate_pct": low_margin_transition_rate,
        "context_caution_rate_pct": float(context_caution.mean() * 100.0),
        "caution_source_breakdown": caution_source_breakdown,
        "caution_overlap_counts": caution_overlap_counts,
    }


def _transition_matrix(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    regime = pd.to_numeric(df["regime"], errors="coerce")
    prev = pd.to_numeric(df["regime_prev"], errors="coerce")
    valid = regime.notna() & prev.notna()
    grouped = (
        pd.DataFrame({"prev": prev, "curr": regime})
        .loc[valid]
        .groupby(["prev", "curr"])
        .size()
    )
    out: dict[str, dict[str, int]] = {
        REGIME_NAMES[prev_key]: {REGIME_NAMES[curr_key]: 0 for curr_key in REGIME_NAMES}
        for prev_key in REGIME_NAMES
    }
    for (prev_value, curr_value), count in grouped.items():
        out[REGIME_NAMES[int(prev_value)]][REGIME_NAMES[int(curr_value)]] = int(count)
    return out


def _flicker_diagnostics(df: pd.DataFrame) -> dict[str, float | int | None]:
    valid = _valid_regime_mask(df)
    if not bool(valid.any()):
        return {
            "changes_involving_transitional_rate_pct": None,
            "direct_extreme_jump_count": 0,
            "direct_extreme_jump_rate_pct": None,
            "single_bar_segment_count": 0,
            "single_bar_segment_rate_pct": None,
            "two_bar_segment_count": 0,
            "two_bar_segment_rate_pct": None,
        }

    regime = pd.to_numeric(df["regime"], errors="coerce")
    prev = pd.to_numeric(df["regime_prev"], errors="coerce")
    changed = pd.to_numeric(df["regime_changed"], errors="coerce").fillna(0).eq(1)
    segment_rows = _segment_frame(df)
    transition_rows = int(changed.sum())
    involving_transitional = int((changed & (regime.eq(1) | prev.eq(1))).sum())
    direct_extreme = int(
        pd.to_numeric(df["regime_direct_extreme_jump"], errors="coerce").fillna(0).sum()
    )
    single_bar = int(segment_rows["dwell"].eq(1).sum()) if not segment_rows.empty else 0
    two_bar = int(segment_rows["dwell"].eq(2).sum()) if not segment_rows.empty else 0
    segment_count = int(len(segment_rows))

    return {
        "changes_involving_transitional_rate_pct": (
            float(involving_transitional / transition_rows * 100.0)
            if transition_rows
            else None
        ),
        "direct_extreme_jump_count": direct_extreme,
        "direct_extreme_jump_rate_pct": (
            float(direct_extreme / transition_rows * 100.0) if transition_rows else None
        ),
        "single_bar_segment_count": single_bar,
        "single_bar_segment_rate_pct": (
            float(single_bar / segment_count * 100.0) if segment_count else None
        ),
        "two_bar_segment_count": two_bar,
        "two_bar_segment_rate_pct": (
            float(two_bar / segment_count * 100.0) if segment_count else None
        ),
    }


def _raw_vs_stabilized_audit(df: pd.DataFrame) -> dict[str, object]:
    valid = _valid_regime_mask(df)
    raw_valid = pd.to_numeric(df["raw_regime"], errors="coerce").notna()
    changed = pd.to_numeric(df["regime_stabilized_from_raw"], errors="coerce").fillna(0)
    forced = pd.to_numeric(df["regime_forced_transitional"], errors="coerce").fillna(0)
    direct = pd.to_numeric(df["regime_direct_extreme_jump"], errors="coerce").fillna(0)
    raw_prev = pd.to_numeric(df["raw_regime"], errors="coerce").shift(1)
    raw_now = pd.to_numeric(df["raw_regime"], errors="coerce")
    raw_extreme_flip = (
        raw_prev.notna()
        & raw_now.notna()
        & raw_prev.ne(raw_now)
        & raw_prev.isin([0, 2])
        & raw_now.isin([0, 2])
    )
    prevented = int((raw_extreme_flip & forced.eq(1)).sum())
    return {
        "raw_regime_counts": _regime_counts(df, regime_col="raw_regime"),
        "stabilized_regime_counts": _regime_counts(df, regime_col="regime"),
        "stabilized_rows": int(changed.sum()),
        "regime_stabilized_from_raw_rate_pct": (
            float(changed.loc[valid].mean() * 100.0) if bool(valid.any()) else None
        ),
        "forced_transitional_rate_pct": (
            float(forced.loc[valid].mean() * 100.0) if bool(valid.any()) else None
        ),
        "direct_extreme_jump_count": int(direct.sum()),
        "direct_extreme_jump_rate_pct": (
            float(direct.loc[valid].mean() * 100.0) if bool(valid.any()) else None
        ),
        "raw_extreme_flips_prevented": prevented,
        "raw_valid_row_count": int(raw_valid.sum()),
    }


def _regime_by_session(df: pd.DataFrame) -> dict[str, dict[str, int]] | None:
    if "session_name" not in df.columns:
        return None
    regime = pd.to_numeric(df["regime"], errors="coerce")
    valid = regime.notna()
    if not bool(valid.any()):
        return None
    grouped = (
        pd.DataFrame({"session_name": df["session_name"], "regime": regime})
        .loc[valid]
        .groupby(["session_name", "regime"])
        .size()
    )
    out: dict[str, dict[str, int]] = {}
    for (session_name, regime_code), count in grouped.items():
        out.setdefault(str(session_name), {})[REGIME_NAMES[int(regime_code)]] = int(
            count
        )
    return out


def _build_regime_figure(df: pd.DataFrame, *, title: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.48, 0.24, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )

    regime = pd.to_numeric(df["regime"], errors="coerce")
    start = None
    current = None
    for i, value in enumerate(regime):
        if pd.isna(value):
            if start is not None and current is not None:
                fig.add_vrect(
                    x0=df["timestamp"].iloc[start],
                    x1=df["timestamp"].iloc[i - 1],
                    fillcolor=REGIME_BACKGROUND[int(current)],
                    line_width=0,
                    opacity=0.35,
                    row="all",
                    col=1,
                )
            start = None
            current = None
            continue
        if start is None:
            start = i
            current = value
            continue
        if value != current:
            fig.add_vrect(
                x0=df["timestamp"].iloc[start],
                x1=df["timestamp"].iloc[i - 1],
                fillcolor=REGIME_BACKGROUND[int(current)],
                line_width=0,
                opacity=0.35,
                row="all",
                col=1,
            )
            start = i
            current = value
    if start is not None and current is not None:
        fig.add_vrect(
            x0=df["timestamp"].iloc[start],
            x1=df["timestamp"].iloc[len(df) - 1],
            fillcolor=REGIME_BACKGROUND[int(current)],
            line_width=0,
            opacity=0.35,
            row="all",
            col=1,
        )

    changes = df.loc[
        pd.to_numeric(df["regime_changed"], errors="coerce").fillna(0).eq(1)
    ]
    if not changes.empty:
        fig.add_trace(
            go.Scatter(
                x=changes["timestamp"],
                y=changes["close"],
                mode="markers",
                name="regime_changed",
                marker=dict(symbol="x", size=10, color="black"),
            ),
            row=1,
            col=1,
        )

    raw_change = (
        pd.to_numeric(df["raw_regime"], errors="coerce").ne(
            pd.to_numeric(df["regime"], errors="coerce")
        )
        & pd.to_numeric(df["regime"], errors="coerce").notna()
    )
    if bool(raw_change.any()):
        raw_scoped = df.loc[raw_change]
        fig.add_trace(
            go.Scatter(
                x=raw_scoped["timestamp"],
                y=raw_scoped["close"],
                mode="markers",
                name="raw!=final",
                marker=dict(symbol="circle-open", size=8, color="#5f0f40"),
            ),
            row=1,
            col=1,
        )

    boundary = df.loc[
        pd.to_numeric(df["regime_boundary_flag"], errors="coerce").fillna(0).eq(1)
    ]
    if not boundary.empty:
        fig.add_trace(
            go.Scatter(
                x=boundary["timestamp"],
                y=boundary["close"],
                mode="markers",
                name="boundary",
                marker=dict(symbol="diamond", size=8, color="#ffb703"),
            ),
            row=1,
            col=1,
        )

    forced = df.loc[
        pd.to_numeric(df["regime_forced_transitional"], errors="coerce").fillna(0).eq(1)
    ]
    if not forced.empty:
        fig.add_trace(
            go.Scatter(
                x=forced["timestamp"],
                y=forced["close"],
                mode="markers",
                name="forced_transitional",
                marker=dict(symbol="triangle-up", size=9, color="#d00000"),
            ),
            row=1,
            col=1,
        )

    trending_neutral = df.loc[
        pd.to_numeric(df["regime"], errors="coerce").eq(2)
        & pd.to_numeric(df["trend_state"], errors="coerce").eq(0)
    ]
    if not trending_neutral.empty:
        fig.add_trace(
            go.Scatter(
                x=trending_neutral["timestamp"],
                y=trending_neutral["close"],
                mode="markers",
                name="trend_regime_neutral_trend",
                marker=dict(symbol="square", size=8, color="#118ab2"),
            ),
            row=1,
            col=1,
        )

    ranging_directional = df.loc[
        pd.to_numeric(df["regime"], errors="coerce").eq(0)
        & pd.to_numeric(df["trend_state"], errors="coerce").ne(0)
        & pd.to_numeric(df["trend_state"], errors="coerce").notna()
    ]
    if not ranging_directional.empty:
        fig.add_trace(
            go.Scatter(
                x=ranging_directional["timestamp"],
                y=ranging_directional["close"],
                mode="markers",
                name="range_regime_directional_trend",
                marker=dict(symbol="square-open", size=8, color="#ef476f"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["adx_14"], mode="lines", name="adx_14"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["timestamp"], y=df["bb_width"], mode="lines", name="bb_width"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=pd.to_numeric(df["trend_state"], errors="coerce"),
            mode="lines",
            name="trend_state",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=pd.to_numeric(df["raw_regime"], errors="coerce"),
            mode="lines",
            name="raw_regime",
            line=dict(color="#6a4c93", width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )
    for col in ("regime_confidence", "regime_margin"):
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col),
            row=3,
            col=1,
        )

    current_snapshot = _current_regime_snapshot(df)
    if current_snapshot is not None:
        fig.add_annotation(
            x=df["timestamp"].iloc[-1],
            y=1.0,
            xref="x",
            yref="paper",
            text=(
                f"Current Regime: {current_snapshot['regime']}<br>"
                f"Conf {current_snapshot['regime_confidence']:.2f} | "
                f"Margin {current_snapshot['regime_margin']:.2f}<br>"
                f"Trend {current_snapshot['trend_state']} | "
                f"Bars {current_snapshot['bars_in_regime']} | "
                f"Caution {current_snapshot['regime_context_caution']}"
            ),
            showarrow=False,
            xanchor="right",
            yanchor="top",
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#333333",
            borderwidth=1,
            font=dict(size=11),
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1100,
        xaxis_rangeslider_visible=False,
    )
    return fig


def validate_regime(
    df: pd.DataFrame,
    *,
    summary_df: pd.DataFrame | None = None,
    live_df: pd.DataFrame | None = None,
    research_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Regime Validation",
    synthetic_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    audit_df = (
        summary_df
        if summary_df is not None
        else (research_df if research_df is not None else df)
    )
    missing = sorted(REQUIRED_REGIME_COLUMNS - set(audit_df.columns))
    if missing:
        raise ValueError(f"validate_regime: missing required columns: {missing}")

    live_research_parity_ok = None
    if live_df is not None and research_df is not None:
        non_research_cols = [
            col for col in research_df.columns if not col.startswith("r_")
        ]
        live_research_parity_ok = bool(
            live_df[non_research_cols].equals(research_df[non_research_cols])
        )

    summary = {
        "row_count": int(len(audit_df)),
        "valid_regime_row_count": int(_valid_regime_mask(audit_df).sum()),
        "downstream_caution_contract": {
            "treat_regime_boundary_flag_eq_1_as_degraded_context": True,
            "treat_regime_confidence_below_0_60_as_degraded_context": True,
            "treat_bars_in_regime_le_2_as_degraded_context": True,
            "canonical_flag": "regime_context_caution",
        },
        "warmup": _warmup_contract(audit_df),
        "value_counts": _regime_counts(audit_df),
        "regime_change_count": _regime_change_count(audit_df),
        "dwell_stats": _dwell_stats(audit_df),
        "alignment_rates": _alignment_rates(audit_df),
        "per_regime_alignment": _per_regime_alignment(audit_df),
        "extreme_consistency": _extreme_consistency(audit_df),
        "unaligned_decomposition": _unaligned_decomposition(audit_df),
        "extreme_misalignment_audit": _extreme_misalignment_audit(audit_df),
        "extreme_misalignment_profiles": _extreme_misalignment_profiles(audit_df),
        "current_regime_snapshot": _current_regime_snapshot(audit_df),
        "boundary_diagnostics": _boundary_diagnostics(audit_df),
        "transition_matrix": _transition_matrix(audit_df),
        "flicker_diagnostics": _flicker_diagnostics(audit_df),
        "raw_vs_stabilized_audit": _raw_vs_stabilized_audit(audit_df),
        "checks": {
            "required_columns_present": True,
            "no_inf_values": _no_inf_ok(audit_df),
            "score_bounds_ok": _score_bounds_ok(audit_df),
            "regime_values_ok": _state_contract_ok(audit_df),
            "one_hot_exclusive_ok": _one_hot_ok(audit_df),
            "transition_contract_ok": _transition_contract_ok(audit_df),
            "persistence_bounds_ok": _persistence_bounds_ok(audit_df),
            "live_research_parity_ok": live_research_parity_ok,
            "no_label_contamination_ok": _no_label_contamination_ok(
                live_df if live_df is not None else audit_df
            ),
        },
        "summary_stats": {
            col: _continuous_stats(audit_df[col]) for col in SUMMARY_COLUMNS
        },
        "trend_state_confusion_matrix": _confusion_matrix(
            audit_df,
            row_col="regime",
            col_col="trend_state",
            row_labels=REGIME_NAMES,
            col_labels=TREND_STATE_NAMES,
        ),
    }
    if "trend_bias_state" in audit_df.columns:
        summary["trend_bias_confusion_matrix"] = _confusion_matrix(
            audit_df,
            row_col="regime",
            col_col="trend_bias_state",
            row_labels=REGIME_NAMES,
            col_labels=TREND_STATE_NAMES,
        )
    session_counts = _regime_by_session(audit_df)
    if session_counts is not None:
        summary["regime_by_session"] = session_counts
    if synthetic_summary is not None:
        summary["synthetic_fixture_summary"] = synthetic_summary

    html_path = None
    if outpath is not None:
        fig = _build_regime_figure(df, title=title)
        html_path = save_figure_html(fig, outpath)

    return {"summary": summary, "html_path": html_path}
