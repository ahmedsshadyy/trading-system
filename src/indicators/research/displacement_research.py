from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators._helpers.validators import require_columns, require_ohlc

EPS = 1e-12
DEFAULT_VOLUME_RATIO_PERIOD = 20
EXCURSION_RATIO_DEN_FLOOR = 0.10
EXCURSION_RATIO_CAP = 10.0

DEFAULT_DISPLACEMENT_RESEARCH_HORIZONS = (1, 2, 3, 5, 10, 20)
FOLLOW_THROUGH_HORIZONS = (1, 2, 3, 5, 10, 20)
EXCURSION_HORIZONS = DEFAULT_DISPLACEMENT_RESEARCH_HORIZONS
RETEST_HORIZONS = (3, 5, 10)
OUTCOME_HORIZONS = (3, 5, 10, 20)
CONTINUATION_THRESHOLDS: dict[int, tuple[float, ...]] = {
    1: (0.5, 1.0),
    3: (0.5, 1.0, 1.5),
    5: (0.5, 1.0, 1.5),
    10: (1.0, 1.5, 2.0),
}

DISPLACEMENT_EVENT_COLUMNS = [
    "displacement_event_id",
    "displacement_detect_idx",
    "displacement_detect_ts",
    "displacement_side",
    "displacement_direction",
    "displacement_open_on_detect",
    "displacement_high_on_detect",
    "displacement_low_on_detect",
    "displacement_close_on_detect",
    "displacement_atr_on_detect",
]

DISPLACEMENT_GEOMETRY_COLUMNS = [
    "displacement_body_atr",
    "displacement_range_atr",
    "displacement_signed_body_atr",
    "displacement_body_frac",
    "displacement_upper_wick_frac",
    "displacement_lower_wick_frac",
    "displacement_opposite_wick_frac",
    "displacement_close_to_extreme_frac",
    "displacement_score",
]

DISPLACEMENT_CONTEXT_COLUMNS = [
    "displacement_trend_state_on_event",
    "displacement_trend_bias_state_on_event",
    "displacement_bos_direction_on_event",
    "displacement_choch_direction_on_event",
    "displacement_session_on_event",
    "displacement_regime_on_event",
    "displacement_volume_ratio_on_event",
    "displacement_adx_on_event",
    "displacement_rsi_on_event",
]

DISPLACEMENT_FOLLOW_THROUGH_COLUMNS = [
    *(f"displacement_hold_{h}" for h in FOLLOW_THROUGH_HORIZONS),
    *(f"displacement_failed_{h}" for h in FOLLOW_THROUGH_HORIZONS),
    *(f"displacement_reversal_{h}" for h in FOLLOW_THROUGH_HORIZONS),
]

DISPLACEMENT_EXCURSION_COLUMNS = [
    *(f"displacement_mfe_{h}_atr" for h in EXCURSION_HORIZONS),
    *(f"displacement_mae_{h}_atr" for h in EXCURSION_HORIZONS),
    *(f"displacement_excursion_ratio_{h}" for h in EXCURSION_HORIZONS),
    *(f"displacement_excursion_ratio_{h}_capped" for h in EXCURSION_HORIZONS),
]

DISPLACEMENT_CONTINUATION_COLUMNS = [
    f"displacement_continuation_{h}_{threshold:.1f}atr"
    for h, thresholds in CONTINUATION_THRESHOLDS.items()
    for threshold in thresholds
]

DISPLACEMENT_RETEST_COLUMNS = [
    *(f"displacement_retest_ever_{h}" for h in RETEST_HORIZONS),
    "displacement_first_retest_idx",
    "displacement_first_retest_delay",
    "displacement_first_retest_depth_frac",
    *(f"displacement_retest_count_{h}" for h in RETEST_HORIZONS),
    *(f"displacement_hold_after_retest_{h}" for h in RETEST_HORIZONS),
    *(f"displacement_mfe_from_retest_{h}_atr" for h in RETEST_HORIZONS),
    *(f"displacement_mae_from_retest_{h}_atr" for h in RETEST_HORIZONS),
]

DISPLACEMENT_OUTCOME_COLUMNS = [
    f"displacement_final_outcome_{h}" for h in OUTCOME_HORIZONS
]

DISPLACEMENT_SCORE_COLUMNS = [
    "displacement_r_has_valid_retest_reaction",
    "displacement_quality_score",
    "displacement_follow_through_score",
    "displacement_tradeable_score",
    "displacement_failure_severity_score",
]

DISPLACEMENT_RESEARCH_COLUMNS = (
    DISPLACEMENT_EVENT_COLUMNS
    + DISPLACEMENT_GEOMETRY_COLUMNS
    + DISPLACEMENT_CONTEXT_COLUMNS
    + DISPLACEMENT_FOLLOW_THROUGH_COLUMNS
    + DISPLACEMENT_EXCURSION_COLUMNS
    + DISPLACEMENT_CONTINUATION_COLUMNS
    + DISPLACEMENT_RETEST_COLUMNS
    + DISPLACEMENT_OUTCOME_COLUMNS
    + ["displacement_retest_reaction_score"]
    + DISPLACEMENT_SCORE_COLUMNS
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    if num < 0:
        return np.nan
    return float(num / max(den, EXCURSION_RATIO_DEN_FLOOR))


def _cap_ratio(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(min(value, EXCURSION_RATIO_CAP))


def _non_negative(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(max(value, 0.0))


def _renormalized_weighted_score(
    components: list[tuple[float, float]],
) -> float:
    finite = [(weight, value) for weight, value in components if np.isfinite(value)]
    if not finite:
        return np.nan
    weight_sum = sum(weight for weight, _ in finite)
    if weight_sum <= 0:
        return np.nan
    return float(sum(weight * value for weight, value in finite) / weight_sum)


def _series_at_positions(
    df: pd.DataFrame,
    positions: np.ndarray,
    primary: str,
    *,
    fallback: str | None = None,
    dtype: type | str = float,
    default: float | str = np.nan,
    as_string: bool = False,
) -> np.ndarray:
    col = primary
    if col not in df.columns and fallback is not None and fallback in df.columns:
        col = fallback
    if col not in df.columns:
        if as_string:
            return np.full(len(positions), str(default), dtype=object)
        return np.full(len(positions), default, dtype=float)
    series = df.iloc[positions][col]
    if as_string:
        return series.astype(str).to_numpy(dtype=object)
    return series.to_numpy(dtype=dtype)


def _volume_ratio_array(
    df: pd.DataFrame,
    *,
    period: int = DEFAULT_VOLUME_RATIO_PERIOD,
) -> np.ndarray:
    if "vol_ratio" in df.columns:
        return df["vol_ratio"].to_numpy(dtype=float)
    return np.full(len(df), np.nan, dtype=float)


def _bucket_body_atr(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 1.5:
        return "<1.5"
    if value < 2.0:
        return "1.5_2.0"
    if value < 3.0:
        return "2.0_3.0"
    return "3.0+"


def _bucket_body_frac(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.60:
        return "<0.60"
    if value < 0.70:
        return "0.60_0.70"
    if value < 0.80:
        return "0.70_0.80"
    return "0.80+"


def _bucket_close_to_extreme(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 0.05:
        return "<=0.05"
    if value <= 0.10:
        return "0.05_0.10"
    if value <= 0.20:
        return "0.10_0.20"
    return ">0.20"


def _extract_displacement_events(
    df: pd.DataFrame,
    *,
    atr_length: int,
) -> pd.DataFrame:
    require_ohlc(df, caller="build_displacement_research_table")
    require_columns(
        df,
        {
            "displacement_flag",
            "displacement_direction",
            *DISPLACEMENT_GEOMETRY_COLUMNS,
        },
        caller="build_displacement_research_table",
    )

    detect_positions = np.flatnonzero(
        df["displacement_flag"].to_numpy(dtype=np.int8) == 1
    )
    if len(detect_positions) == 0:
        return pd.DataFrame(columns=DISPLACEMENT_RESEARCH_COLUMNS)

    atr = get_atr_array(df, length=atr_length)
    volume_ratio = _volume_ratio_array(df)
    direction = df.iloc[detect_positions]["displacement_direction"].to_numpy(
        dtype=np.int8
    )
    events = pd.DataFrame(
        {
            "displacement_event_id": np.arange(1, len(detect_positions) + 1, dtype=int),
            "displacement_detect_idx": detect_positions.astype(int),
            "displacement_detect_ts": (
                pd.to_datetime(df.iloc[detect_positions]["timestamp"], utc=True)
                if "timestamp" in df.columns
                else pd.Series(pd.NaT, index=np.arange(len(detect_positions)))
            ),
            "displacement_side": np.where(direction > 0, "bull", "bear"),
            "displacement_direction": direction.astype(np.int8),
            "displacement_open_on_detect": df.iloc[detect_positions]["open"].to_numpy(
                dtype=float
            ),
            "displacement_high_on_detect": df.iloc[detect_positions]["high"].to_numpy(
                dtype=float
            ),
            "displacement_low_on_detect": df.iloc[detect_positions]["low"].to_numpy(
                dtype=float
            ),
            "displacement_close_on_detect": df.iloc[detect_positions]["close"].to_numpy(
                dtype=float
            ),
            "displacement_atr_on_detect": atr[detect_positions].astype(float),
        }
    )

    for col in DISPLACEMENT_GEOMETRY_COLUMNS:
        events[col] = df.iloc[detect_positions][col].to_numpy(dtype=float)

    events["displacement_trend_state_on_event"] = _series_at_positions(
        df, detect_positions, "trend_state"
    )
    events["displacement_trend_bias_state_on_event"] = _series_at_positions(
        df, detect_positions, "trend_bias_state", fallback="trend_bias"
    )
    events["displacement_bos_direction_on_event"] = _series_at_positions(
        df, detect_positions, "bos_direction"
    )
    events["displacement_choch_direction_on_event"] = _series_at_positions(
        df, detect_positions, "choch_direction"
    )
    events["displacement_session_on_event"] = _series_at_positions(
        df,
        detect_positions,
        "session",
        as_string=True,
        default="unknown",
    )
    events["displacement_regime_on_event"] = _series_at_positions(
        df, detect_positions, "regime"
    )
    events["displacement_volume_ratio_on_event"] = volume_ratio[detect_positions]
    events["displacement_adx_on_event"] = _series_at_positions(
        df, detect_positions, "adx_14"
    )
    events["displacement_rsi_on_event"] = _series_at_positions(
        df, detect_positions, "rsi_14"
    )

    return events


def build_displacement_research_table(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    horizons: tuple[int, ...] = DEFAULT_DISPLACEMENT_RESEARCH_HORIZONS,
) -> pd.DataFrame:
    normalized = normalize_candle_schema(df, require_volume=False)
    events = _extract_displacement_events(normalized, atr_length=atr_length)
    if events.empty:
        return events

    horizons = tuple(sorted(set(int(h) for h in horizons)))
    missing_excursion = set(EXCURSION_HORIZONS) - set(horizons)
    if missing_excursion:
        raise ValueError(
            f"build_displacement_research_table requires horizons covering "
            f"{sorted(EXCURSION_HORIZONS)}, missing {sorted(missing_excursion)}"
        )

    o = normalized["open"].to_numpy(dtype=float)
    h = normalized["high"].to_numpy(dtype=float)
    l = normalized["low"].to_numpy(dtype=float)
    c = normalized["close"].to_numpy(dtype=float)
    atr = get_atr_array(normalized, length=atr_length)

    output_rows: list[dict[str, object]] = []
    n = len(normalized)

    for row in events.itertuples(index=False):
        event: dict[str, object] = dict(row._asdict())
        detect_idx = int(event["displacement_detect_idx"])
        direction = int(event["displacement_direction"])
        side = str(event["displacement_side"])
        detect_high = float(event["displacement_high_on_detect"])
        detect_low = float(event["displacement_low_on_detect"])
        detect_close = float(event["displacement_close_on_detect"])
        detect_midpoint = 0.5 * (detect_high + detect_low)
        detect_range = max(detect_high - detect_low, EPS)
        atr_detect = float(event["displacement_atr_on_detect"])

        future_slice_all = slice(detect_idx + 1, n)
        future_h_all = h[future_slice_all]
        future_l_all = l[future_slice_all]

        if side == "bull":
            retest_zone_mask_all = (future_l_all <= detect_midpoint) & (
                future_l_all >= detect_low
            )
        else:
            retest_zone_mask_all = (future_h_all >= detect_midpoint) & (
                future_h_all <= detect_high
            )

        retest_offsets_all = np.flatnonzero(retest_zone_mask_all)
        if retest_offsets_all.size > 0:
            first_retest_idx = detect_idx + 1 + int(retest_offsets_all[0])
            event["displacement_first_retest_idx"] = float(first_retest_idx)
            event["displacement_first_retest_delay"] = float(
                first_retest_idx - detect_idx
            )
            if side == "bull":
                depth = (detect_high - l[first_retest_idx]) / detect_range
            else:
                depth = (h[first_retest_idx] - detect_low) / detect_range
            event["displacement_first_retest_depth_frac"] = _clip01(depth)
        else:
            first_retest_idx = None
            event["displacement_first_retest_idx"] = np.nan
            event["displacement_first_retest_delay"] = np.nan
            event["displacement_first_retest_depth_frac"] = np.nan

        for horizon in EXCURSION_HORIZONS:
            has_horizon = detect_idx + horizon < n
            mfe = np.nan
            mae = np.nan
            ratio = np.nan
            hold = np.nan
            failed = np.nan
            reversal = np.nan

            if has_horizon and np.isfinite(atr_detect) and atr_detect > 0:
                detect_stop = detect_idx + horizon + 1
                future_h = h[detect_idx + 1 : detect_stop]
                future_l = l[detect_idx + 1 : detect_stop]
                future_c = c[detect_idx + 1 : detect_stop]
                reverse_extension = np.nan

                if side == "bull":
                    mfe = _non_negative(
                        (np.nanmax(future_h) - detect_close) / max(atr_detect, EPS)
                    )
                    mae = _non_negative(
                        (detect_close - np.nanmin(future_l)) / max(atr_detect, EPS)
                    )
                    hold = bool(~np.any(future_c < detect_midpoint))
                    failed = bool(np.any(future_c < detect_low))
                    reverse_extension = _non_negative(
                        (detect_low - np.nanmin(future_l)) / max(atr_detect, EPS)
                    )
                else:
                    mfe = _non_negative(
                        (detect_close - np.nanmin(future_l)) / max(atr_detect, EPS)
                    )
                    mae = _non_negative(
                        (np.nanmax(future_h) - detect_close) / max(atr_detect, EPS)
                    )
                    hold = bool(~np.any(future_c > detect_midpoint))
                    failed = bool(np.any(future_c > detect_high))
                    reverse_extension = _non_negative(
                        (np.nanmax(future_h) - detect_high) / max(atr_detect, EPS)
                    )

                ratio = _safe_ratio(mfe, mae)
                reversal = bool(
                    failed
                    and np.isfinite(reverse_extension)
                    and reverse_extension >= 0.5
                )

            event[f"displacement_mfe_{horizon}_atr"] = mfe
            event[f"displacement_mae_{horizon}_atr"] = mae
            event[f"displacement_excursion_ratio_{horizon}"] = ratio
            event[f"displacement_excursion_ratio_{horizon}_capped"] = _cap_ratio(ratio)

            if horizon in FOLLOW_THROUGH_HORIZONS:
                event[f"displacement_hold_{horizon}"] = hold
                event[f"displacement_failed_{horizon}"] = failed
                event[f"displacement_reversal_{horizon}"] = reversal

            thresholds = CONTINUATION_THRESHOLDS.get(horizon)
            if thresholds is not None:
                for threshold in thresholds:
                    key = f"displacement_continuation_{horizon}_{threshold:.1f}atr"
                    event[key] = (
                        bool(np.isfinite(mfe) and mfe >= threshold)
                        if has_horizon
                        else np.nan
                    )

            if horizon in RETEST_HORIZONS:
                if has_horizon:
                    retest_window_mask = retest_zone_mask_all[:horizon]
                    retest_starts = np.flatnonzero(
                        retest_window_mask & ~np.r_[False, retest_window_mask[:-1]]
                    )
                    event[f"displacement_retest_ever_{horizon}"] = bool(
                        first_retest_idx is not None
                        and (first_retest_idx - detect_idx) <= horizon
                    )
                    event[f"displacement_retest_count_{horizon}"] = int(
                        len(retest_starts)
                    )
                else:
                    event[f"displacement_retest_ever_{horizon}"] = np.nan
                    event[f"displacement_retest_count_{horizon}"] = np.nan

                if first_retest_idx is not None and first_retest_idx + horizon < n:
                    retest_close = float(c[first_retest_idx])
                    atr_retest = float(atr[first_retest_idx])
                    if not np.isfinite(atr_retest) or atr_retest <= 0:
                        atr_retest = atr_detect
                    retest_stop = first_retest_idx + horizon + 1
                    retest_h = h[first_retest_idx + 1 : retest_stop]
                    retest_l = l[first_retest_idx + 1 : retest_stop]
                    retest_c = c[first_retest_idx + 1 : retest_stop]
                    if side == "bull":
                        mfe_retest = _non_negative(
                            (np.nanmax(retest_h) - retest_close) / max(atr_retest, EPS)
                        )
                        mae_retest = _non_negative(
                            (retest_close - np.nanmin(retest_l)) / max(atr_retest, EPS)
                        )
                        hold_after_retest = bool(~np.any(retest_c < detect_low))
                    else:
                        mfe_retest = _non_negative(
                            (retest_close - np.nanmin(retest_l)) / max(atr_retest, EPS)
                        )
                        mae_retest = _non_negative(
                            (np.nanmax(retest_h) - retest_close) / max(atr_retest, EPS)
                        )
                        hold_after_retest = bool(~np.any(retest_c > detect_high))
                    event[f"displacement_hold_after_retest_{horizon}"] = (
                        hold_after_retest
                    )
                    event[f"displacement_mfe_from_retest_{horizon}_atr"] = mfe_retest
                    event[f"displacement_mae_from_retest_{horizon}_atr"] = mae_retest
                else:
                    event[f"displacement_hold_after_retest_{horizon}"] = np.nan
                    event[f"displacement_mfe_from_retest_{horizon}_atr"] = np.nan
                    event[f"displacement_mae_from_retest_{horizon}_atr"] = np.nan

            if horizon in OUTCOME_HORIZONS:
                if not has_horizon:
                    event[f"displacement_final_outcome_{horizon}"] = (
                        "insufficient_horizon"
                    )
                elif bool(event.get(f"displacement_reversal_{horizon}", False)):
                    event[f"displacement_final_outcome_{horizon}"] = "reversed"
                elif bool(event.get(f"displacement_failed_{horizon}", False)):
                    event[f"displacement_final_outcome_{horizon}"] = "failed"
                elif (
                    np.isfinite(mfe) and np.isfinite(mae) and mfe >= 1.5 and mae <= 0.5
                ):
                    event[f"displacement_final_outcome_{horizon}"] = (
                        "strong_continuation"
                    )
                elif np.isfinite(mfe) and mfe >= 0.5:
                    event[f"displacement_final_outcome_{horizon}"] = "weak_continuation"
                else:
                    event[f"displacement_final_outcome_{horizon}"] = "chop"

        quality_score = _clip01(
            0.35 * _clip01(event["displacement_body_atr"] / 2.0)
            + 0.25 * _clip01(event["displacement_body_frac"] / 0.8)
            + 0.20 * (1.0 - _clip01(event["displacement_close_to_extreme_frac"] / 0.2))
            + 0.10 * (1.0 - _clip01(event["displacement_opposite_wick_frac"] / 0.2))
            + 0.10 * _clip01(event["displacement_range_atr"] / 2.5)
        )

        mfe_5 = event.get("displacement_mfe_5_atr", np.nan)
        mae_5 = event.get("displacement_mae_5_atr", np.nan)
        hold_5 = event.get("displacement_hold_5", np.nan)
        cont_5 = event.get("displacement_continuation_5_1.0atr", np.nan)
        if np.isfinite(mfe_5) and np.isfinite(mae_5):
            follow_through_score = _clip01(
                0.45 * _clip01(mfe_5 / 2.0)
                + 0.25 * (1.0 - _clip01(mae_5 / 1.5))
                + 0.15 * (1.0 if hold_5 is True else 0.0)
                + 0.15 * (1.0 if cont_5 is True else 0.0)
            )
        else:
            follow_through_score = np.nan

        mfe_from_retest_5 = event.get("displacement_mfe_from_retest_5_atr", np.nan)
        mae_from_retest_5 = event.get("displacement_mae_from_retest_5_atr", np.nan)
        hold_after_retest_5 = event.get("displacement_hold_after_retest_5", np.nan)
        if np.isfinite(mfe_from_retest_5) and np.isfinite(mae_from_retest_5):
            retest_reaction_score = _clip01(
                0.50 * _clip01(mfe_from_retest_5 / 1.5)
                + 0.25 * (1.0 - _clip01(mae_from_retest_5 / 1.0))
                + 0.25 * (1.0 if hold_after_retest_5 is True else 0.0)
            )
        else:
            retest_reaction_score = np.nan

        has_valid_retest_reaction = bool(np.isfinite(retest_reaction_score))
        tradeable_score = _renormalized_weighted_score(
            [
                (0.45, quality_score),
                (0.35, follow_through_score),
                (0.20, retest_reaction_score),
            ]
        )

        failed_5 = event.get("displacement_failed_5", np.nan)
        reversal_5 = event.get("displacement_reversal_5", np.nan)
        if np.isfinite(mfe_5) and np.isfinite(mae_5):
            failure_severity_score = _clip01(
                0.40 * (1.0 if failed_5 is True else 0.0)
                + 0.25 * _clip01(mae_5 / 1.5)
                + 0.20 * (1.0 - _clip01(mfe_5 / 1.0))
                + 0.15 * (1.0 if reversal_5 is True else 0.0)
            )
        else:
            failure_severity_score = np.nan

        event["displacement_r_has_valid_retest_reaction"] = has_valid_retest_reaction
        event["displacement_retest_reaction_score"] = retest_reaction_score
        event["displacement_quality_score"] = quality_score
        event["displacement_follow_through_score"] = follow_through_score
        event["displacement_tradeable_score"] = tradeable_score
        event["displacement_failure_severity_score"] = failure_severity_score

        output_rows.append(event)

    out = (
        pd.DataFrame(output_rows)
        .sort_values(["displacement_detect_idx", "displacement_event_id"])
        .reset_index(drop=True)
    )
    return out


def summarize_displacement_research(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {"event_count": 0}

    def _bool_rate(col: str) -> float:
        series = events[col]
        series = series[series.notna()]
        return float(series.mean()) if not series.empty else np.nan

    def _continuous_stats(col: str) -> dict[str, float | int]:
        clean = pd.to_numeric(events[col], errors="coerce").dropna()
        if clean.empty:
            return {
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            }
        return {
            "count": int(clean.count()),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std(ddof=0)),
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

    def _breakdown(group_col: str) -> dict[str, dict[str, float | int]]:
        frame = events.copy()
        if group_col == "displacement_body_atr_bucket":
            frame[group_col] = frame["displacement_body_atr"].map(_bucket_body_atr)
        elif group_col == "displacement_body_frac_bucket":
            frame[group_col] = frame["displacement_body_frac"].map(_bucket_body_frac)
        elif group_col == "displacement_close_to_extreme_bucket":
            frame[group_col] = frame["displacement_close_to_extreme_frac"].map(
                _bucket_close_to_extreme
            )
        grouped = frame.groupby(group_col, dropna=False)
        out: dict[str, dict[str, float | int]] = {}
        for key, group in grouped:
            label = "NaN" if pd.isna(key) else str(key)
            out[label] = {
                "event_count": int(len(group)),
                "quality_score_mean": float(group["displacement_quality_score"].mean()),
                "tradeable_score_mean": float(
                    group["displacement_tradeable_score"].mean()
                ),
                "hold_5_rate": (
                    float(group["displacement_hold_5"].dropna().mean())
                    if group["displacement_hold_5"].notna().any()
                    else np.nan
                ),
                "failed_5_rate": (
                    float(group["displacement_failed_5"].dropna().mean())
                    if group["displacement_failed_5"].notna().any()
                    else np.nan
                ),
                "continuation_5_1.0atr_rate": (
                    float(group["displacement_continuation_5_1.0atr"].dropna().mean())
                    if group["displacement_continuation_5_1.0atr"].notna().any()
                    else np.nan
                ),
            }
        return out

    hold_rates = {
        str(h): _bool_rate(f"displacement_hold_{h}") for h in FOLLOW_THROUGH_HORIZONS
    }
    failure_rates = {
        str(h): _bool_rate(f"displacement_failed_{h}") for h in FOLLOW_THROUGH_HORIZONS
    }
    reversal_rates = {
        str(h): _bool_rate(f"displacement_reversal_{h}")
        for h in FOLLOW_THROUGH_HORIZONS
    }
    continuation_rates = {
        f"{h}_{threshold:.1f}atr": _bool_rate(
            f"displacement_continuation_{h}_{threshold:.1f}atr"
        )
        for h, thresholds in CONTINUATION_THRESHOLDS.items()
        for threshold in thresholds
    }
    outcome_distributions = {
        str(h): (
            events[f"displacement_final_outcome_{h}"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        for h in OUTCOME_HORIZONS
    }

    outcome_reconciliation: dict[str, dict[str, bool | int]] = {}
    for horizon in OUTCOME_HORIZONS:
        outcome_col = f"displacement_final_outcome_{horizon}"
        failed_col = f"displacement_failed_{horizon}"
        reversal_col = f"displacement_reversal_{horizon}"
        hold_col = f"displacement_hold_{horizon}"
        sufficient_mask = events[outcome_col] != "insufficient_horizon"
        failed_mask = events[outcome_col] == "failed"
        reversed_mask = events[outcome_col] == "reversed"
        valid_labels = {
            "strong_continuation",
            "weak_continuation",
            "chop",
            "failed",
            "reversed",
            "insufficient_horizon",
        }
        outcome_reconciliation[str(horizon)] = {
            "final_outcomes_sum_to_event_count": int(
                sum(outcome_distributions[str(horizon)].values())
            )
            == int(len(events)),
            "each_event_has_exactly_one_final_outcome": bool(
                events[outcome_col].isin(valid_labels).all()
            ),
            "each_event_with_sufficient_horizon_has_non_null_final_outcome": bool(
                events.loc[sufficient_mask, outcome_col].notna().all()
            ),
            "failed_events_reconcile_with_failed_flag": bool(
                events.loc[failed_mask, failed_col].eq(True).all()
                and events.loc[failed_mask, reversal_col].eq(False).all()
            ),
            "reversed_events_reconcile_with_reversal_flag": bool(
                events.loc[reversed_mask, reversal_col].eq(True).all()
                and events.loc[reversed_mask, failed_col].eq(True).all()
            ),
            "strong_continuation_implies_not_failed": bool(
                events.loc[events[outcome_col] == "strong_continuation", failed_col]
                .eq(False)
                .all()
            ),
            "weak_continuation_implies_not_failed": bool(
                events.loc[events[outcome_col] == "weak_continuation", failed_col]
                .eq(False)
                .all()
            ),
            "chop_implies_not_failed": bool(
                events.loc[events[outcome_col] == "chop", failed_col].eq(False).all()
            ),
            "sufficient_horizon_count": int(sufficient_mask.sum()),
            "hold_column_present": hold_col in events.columns,
        }

    no_retest_mask = events["displacement_first_retest_idx"].isna()
    invalid_retest_reaction_mask = ~no_retest_mask & ~events[
        "displacement_r_has_valid_retest_reaction"
    ].astype(bool)

    negative_excursion_counts = {
        "mfe": int(
            sum(
                (
                    pd.to_numeric(events[f"displacement_mfe_{h}_atr"], errors="coerce")
                    < 0
                )
                .fillna(False)
                .sum()
                for h in EXCURSION_HORIZONS
            )
        ),
        "mae": int(
            sum(
                (
                    pd.to_numeric(events[f"displacement_mae_{h}_atr"], errors="coerce")
                    < 0
                )
                .fillna(False)
                .sum()
                for h in EXCURSION_HORIZONS
            )
        ),
        "ratio": int(
            sum(
                (
                    pd.to_numeric(
                        events[f"displacement_excursion_ratio_{h}"], errors="coerce"
                    )
                    < 0
                )
                .fillna(False)
                .sum()
                for h in EXCURSION_HORIZONS
            )
        ),
    }

    return {
        "event_count": int(len(events)),
        "bull_count": int((events["displacement_direction"] == 1).sum()),
        "bear_count": int((events["displacement_direction"] == -1).sum()),
        "recommended_excursion_ratio_variant": "capped",
        "hold_rates": hold_rates,
        "failure_rates": failure_rates,
        "reversal_rates": reversal_rates,
        "continuation_rates": continuation_rates,
        "retest_frequency": {
            str(h): _bool_rate(f"displacement_retest_ever_{h}") for h in RETEST_HORIZONS
        },
        "retest_reaction_quality": _continuous_stats(
            "displacement_retest_reaction_score"
        ),
        "retest_reaction_quality_count": int(
            pd.to_numeric(events["displacement_retest_reaction_score"], errors="coerce")
            .notna()
            .sum()
        ),
        "no_retest_count": int(no_retest_mask.sum()),
        "invalid_retest_reaction_count": int(invalid_retest_reaction_mask.sum()),
        "outcome_distributions": outcome_distributions,
        "score_stats": {
            "quality": _continuous_stats("displacement_quality_score"),
            "follow_through": _continuous_stats("displacement_follow_through_score"),
            "tradeable": _continuous_stats("displacement_tradeable_score"),
            "failure_severity": _continuous_stats(
                "displacement_failure_severity_score"
            ),
        },
        "excursion_stats": {
            "mfe_5_atr": _continuous_stats("displacement_mfe_5_atr"),
            "mae_5_atr": _continuous_stats("displacement_mae_5_atr"),
            "excursion_ratio_5_raw": _continuous_stats(
                "displacement_excursion_ratio_5"
            ),
            "excursion_ratio_5_capped": _continuous_stats(
                "displacement_excursion_ratio_5_capped"
            ),
        },
        "breakdowns": {
            "trend_state_on_event": _breakdown("displacement_trend_state_on_event"),
            "session_on_event": _breakdown("displacement_session_on_event"),
            "regime_on_event": _breakdown("displacement_regime_on_event"),
            "body_atr_bucket": _breakdown("displacement_body_atr_bucket"),
            "body_frac_bucket": _breakdown("displacement_body_frac_bucket"),
            "close_to_extreme_bucket": _breakdown(
                "displacement_close_to_extreme_bucket"
            ),
        },
        "consistency_checks": {
            "unique_event_ids": bool(events["displacement_event_id"].is_unique),
            "one_row_per_detect_idx": bool(events["displacement_detect_idx"].is_unique),
            "event_count_matches_unique_detect_idx": int(len(events))
            == int(events["displacement_detect_idx"].nunique()),
            "non_negative_mfe": negative_excursion_counts["mfe"] == 0,
            "non_negative_mae": negative_excursion_counts["mae"] == 0,
            "non_negative_excursion_ratio": negative_excursion_counts["ratio"] == 0,
        },
        "negative_excursion_counts": negative_excursion_counts,
        "outcome_reconciliation": outcome_reconciliation,
    }
