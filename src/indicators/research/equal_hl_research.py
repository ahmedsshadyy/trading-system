from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators._helpers.arrays import get_atr_array
from src.indicators._helpers.validators import require_columns, require_ohlc
from src.indicators.smc.equal_hl import EQHL_STATE_ACTIVE

EPS = 1e-12
DEFAULT_EQUAL_HL_RESEARCH_HORIZONS = (1, 2, 3, 5, 10, 20)
FOLLOW_THROUGH_HORIZONS = (1, 2, 3, 5, 10)
EXCURSION_HORIZONS = DEFAULT_EQUAL_HL_RESEARCH_HORIZONS
OUTCOME_HORIZONS = (3, 5, 10, 20)
RETEST_HORIZONS = (3, 5, 10)
CONTINUATION_THRESHOLDS: dict[int, tuple[float, ...]] = {
    1: (0.5, 1.0),
    3: (0.5, 1.0, 1.5),
    5: (0.5, 1.0, 1.5),
    10: (1.0, 1.5, 2.0),
}
WIDTH_REF = 1.0
AGE_CAP = 200.0
SPAN_CAP = 120.0
RETEST_SCORE_HORIZON = 5

EQHL_RESEARCH_EVENT_COLUMNS = [
    "eqhl_event_id",
    "eqhl_r_detect_idx",
    "eqhl_r_detect_ts",
    "eqhl_r_side",
    "eqhl_r_direction",
    "eqhl_r_cluster_id",
    "eqhl_r_origin_idx",
    "eqhl_r_first_member_detect_idx",
    "eqhl_r_last_member_detect_idx",
]

EQHL_RESEARCH_CORE_COLUMNS = [
    "eqhl_r_level",
    "eqhl_r_low",
    "eqhl_r_high",
    "eqhl_r_width",
    "eqhl_r_width_atr",
    "eqhl_r_touch_count",
    "eqhl_r_structural_score",
    "eqhl_r_tradeable_live_score_on_detect",
    "eqhl_r_score",
    "eqhl_r_span",
    "eqhl_r_formation_delay",
    "eqhl_r_age_on_detect",
    "eqhl_r_state_on_detect",
]

EQHL_RESEARCH_CONTEXT_COLUMNS = [
    "eqhl_r_trend_state_on_event",
    "eqhl_r_trend_bias_state_on_event",
    "eqhl_r_bos_direction_on_event",
    "eqhl_r_choch_direction_on_event",
    "eqhl_r_session_on_event",
    "eqhl_r_regime_on_event",
    "eqhl_r_volume_ratio_on_event",
    "eqhl_r_adx_on_event",
    "eqhl_r_rsi_on_event",
]

EQHL_RESEARCH_FOLLOW_THROUGH_COLUMNS = [
    *(f"eqhl_r_hold_{h}" for h in FOLLOW_THROUGH_HORIZONS),
    *(f"eqhl_r_failed_{h}" for h in FOLLOW_THROUGH_HORIZONS),
    *(f"eqhl_r_swept_{h}" for h in FOLLOW_THROUGH_HORIZONS),
]

EQHL_RESEARCH_EXCURSION_COLUMNS = [
    *(f"eqhl_r_mfe_{h}_atr" for h in EXCURSION_HORIZONS),
    *(f"eqhl_r_mae_{h}_atr" for h in EXCURSION_HORIZONS),
    *(f"eqhl_r_excursion_ratio_{h}" for h in EXCURSION_HORIZONS),
]

EQHL_RESEARCH_CONTINUATION_COLUMNS = [
    f"eqhl_r_continuation_{h}_{threshold:.1f}atr"
    for h, thresholds in CONTINUATION_THRESHOLDS.items()
    for threshold in thresholds
]

EQHL_RESEARCH_RETEST_COLUMNS = [
    *(f"eqhl_r_retest_ever_{h}" for h in RETEST_HORIZONS),
    "eqhl_r_first_retest_idx",
    "eqhl_r_first_retest_delay",
    *(f"eqhl_r_retest_count_{h}" for h in RETEST_HORIZONS),
    *(f"eqhl_r_mfe_from_retest_{h}_atr" for h in RETEST_HORIZONS),
    *(f"eqhl_r_mae_from_retest_{h}_atr" for h in RETEST_HORIZONS),
    *(f"eqhl_r_hold_after_retest_{h}" for h in RETEST_HORIZONS),
]

EQHL_RESEARCH_OUTCOME_COLUMNS = [f"eqhl_r_final_outcome_{h}" for h in OUTCOME_HORIZONS]

EQHL_RESEARCH_SCORE_COLUMNS = [
    "eqhl_r_retest_reaction_score",
    "eqhl_r_quality_score",
    "eqhl_r_follow_through_score",
    "eqhl_r_tradeable_score",
    "eqhl_r_failure_severity_score",
]

EQHL_RESEARCH_COLUMNS = (
    EQHL_RESEARCH_EVENT_COLUMNS
    + EQHL_RESEARCH_CORE_COLUMNS
    + EQHL_RESEARCH_CONTEXT_COLUMNS
    + EQHL_RESEARCH_FOLLOW_THROUGH_COLUMNS
    + EQHL_RESEARCH_EXCURSION_COLUMNS
    + EQHL_RESEARCH_CONTINUATION_COLUMNS
    + EQHL_RESEARCH_RETEST_COLUMNS
    + EQHL_RESEARCH_OUTCOME_COLUMNS
    + EQHL_RESEARCH_SCORE_COLUMNS
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or num < 0:
        return np.nan
    return float(num / max(den, EPS))


def _non_negative(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(max(value, 0.0))


def _renormalized_weighted_score(components: list[tuple[float, float]]) -> float:
    finite = [(weight, value) for weight, value in components if np.isfinite(value)]
    if not finite:
        return np.nan
    weight_sum = sum(weight for weight, _ in finite)
    if weight_sum <= 0:
        return np.nan
    return float(sum(weight * value for weight, value in finite) / weight_sum)


def _continuous_stats(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
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


def _bool_rate(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _bucket_touch_count(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 2:
        return "2"
    if value == 3:
        return "3"
    if value == 4:
        return "4"
    return "5+"


def _bucket_width(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.10:
        return "<0.10"
    if value < 0.25:
        return "0.10_0.25"
    if value < 0.50:
        return "0.25_0.50"
    return "0.50+"


def _bucket_age(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 10:
        return "0_10"
    if value <= 25:
        return "11_25"
    if value <= 50:
        return "26_50"
    return "51+"


def _extract_events(df: pd.DataFrame) -> pd.DataFrame:
    require_ohlc(df, caller="build_equal_hl_research_table")
    require_columns(
        df,
        {
            "eqh_detect_flag",
            "eql_detect_flag",
            "eqh_event_id",
            "eql_event_id",
            "eqh_cluster_id_on_detect",
            "eql_cluster_id_on_detect",
        },
        caller="build_equal_hl_research_table",
    )

    eqh_positions = np.flatnonzero(
        pd.to_numeric(df["eqh_detect_flag"], errors="coerce").fillna(0).to_numpy() == 1
    )
    eql_positions = np.flatnonzero(
        pd.to_numeric(df["eql_detect_flag"], errors="coerce").fillna(0).to_numpy() == 1
    )

    events: list[dict[str, object]] = []
    for positions, prefix, side, direction in (
        (eqh_positions, "eqh", "eqh", -1),
        (eql_positions, "eql", "eql", 1),
    ):
        for pos in positions:
            row = df.iloc[int(pos)]
            events.append(
                {
                    "eqhl_event_id": int(row[f"{prefix}_event_id"]),
                    "eqhl_r_detect_idx": int(pos),
                    "eqhl_r_detect_ts": (
                        pd.to_datetime(row["timestamp"], utc=True)
                        if "timestamp" in df.columns
                        else pd.NaT
                    ),
                    "eqhl_r_side": side,
                    "eqhl_r_direction": direction,
                    "eqhl_r_cluster_id": int(row[f"{prefix}_cluster_id_on_detect"]),
                    "eqhl_r_origin_idx": float(row[f"{prefix}_origin_idx"]),
                    "eqhl_r_first_member_detect_idx": float(
                        row[f"{prefix}_first_member_detect_idx"]
                    ),
                    "eqhl_r_last_member_detect_idx": float(
                        row[f"{prefix}_last_member_detect_idx"]
                    ),
                    "eqhl_r_level": float(row[f"{prefix}_level_on_detect"]),
                    "eqhl_r_low": float(row[f"{prefix}_low_on_detect"]),
                    "eqhl_r_high": float(row[f"{prefix}_high_on_detect"]),
                    "eqhl_r_width": float(row[f"{prefix}_width_on_detect"]),
                    "eqhl_r_width_atr": float(row[f"{prefix}_width_atr_on_detect"]),
                    "eqhl_r_touch_count": float(
                        row[f"{prefix}_member_count_on_detect"]
                    ),
                    "eqhl_r_structural_score": float(
                        row.get(
                            f"{prefix}_structural_score_on_detect",
                            row[f"{prefix}_score_on_detect"],
                        )
                    ),
                    "eqhl_r_tradeable_live_score_on_detect": float(
                        row.get(
                            f"{prefix}_tradeable_live_score_on_detect",
                            row[f"{prefix}_score_on_detect"],
                        )
                    ),
                    "eqhl_r_score": float(row[f"{prefix}_score_on_detect"]),
                    "eqhl_r_span": max(
                        int(float(row[f"{prefix}_last_member_detect_idx"]))
                        - int(float(row[f"{prefix}_first_member_detect_idx"])),
                        0,
                    ),
                    "eqhl_r_formation_delay": max(
                        int(pos) - int(float(row[f"{prefix}_first_member_detect_idx"])),
                        0,
                    ),
                    "eqhl_r_age_on_detect": max(
                        int(pos) - int(float(row[f"{prefix}_first_member_detect_idx"])),
                        0,
                    ),
                    "eqhl_r_state_on_detect": float(EQHL_STATE_ACTIVE),
                }
            )
    events_df = pd.DataFrame(events)
    if events_df.empty:
        return pd.DataFrame(columns=EQHL_RESEARCH_COLUMNS)
    events_df = events_df.sort_values(
        ["eqhl_r_detect_idx", "eqhl_event_id"]
    ).reset_index(drop=True)
    events_df["eqhl_event_id"] = np.arange(1, len(events_df) + 1, dtype=int)
    return events_df


def _future_window(
    array: np.ndarray,
    *,
    detect_idx: int,
    horizon: int,
) -> np.ndarray:
    start = detect_idx + 1
    stop = detect_idx + horizon + 1
    return array[start:stop]


def _zone_overlap(
    high_value: float, low_value: float, zone_low: float, zone_high: float
) -> bool:
    return bool(high_value >= zone_low and low_value <= zone_high)


def build_equal_hl_research_table(
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_EQUAL_HL_RESEARCH_HORIZONS,
    atr_length: int = 14,
) -> pd.DataFrame:
    events = _extract_events(df)
    if events.empty:
        return events

    atr = get_atr_array(df, atr_length).astype(float, copy=False)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    timestamp = (
        pd.to_datetime(df["timestamp"], utc=True) if "timestamp" in df.columns else None
    )

    vol_ratio = (
        pd.to_numeric(df["vol_ratio"], errors="coerce")
        if "vol_ratio" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    trend_state = (
        pd.to_numeric(df["trend_state"], errors="coerce")
        if "trend_state" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    trend_bias = (
        pd.to_numeric(df["trend_bias_state"], errors="coerce")
        if "trend_bias_state" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    bos_direction = (
        pd.to_numeric(df["bos_direction"], errors="coerce")
        if "bos_direction" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    choch_direction = (
        pd.to_numeric(df["choch_direction"], errors="coerce")
        if "choch_direction" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    regime = (
        pd.to_numeric(df["regime"], errors="coerce")
        if "regime" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    adx = (
        pd.to_numeric(df["adx"], errors="coerce")
        if "adx" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    rsi = (
        pd.to_numeric(df["rsi"], errors="coerce")
        if "rsi" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    session = (
        df["session"].astype(str)
        if "session" in df.columns
        else pd.Series("unknown", index=df.index, dtype=object)
    )

    rows: list[dict[str, object]] = []
    for event in events.to_dict("records"):
        detect_idx = int(event["eqhl_r_detect_idx"])
        direction = int(event["eqhl_r_direction"])
        zone_low = float(event["eqhl_r_low"])
        zone_high = float(event["eqhl_r_high"])
        detect_close = float(close[detect_idx])
        atr_detect = float(atr[detect_idx]) if detect_idx < len(atr) else np.nan

        event["eqhl_r_detect_ts"] = (
            timestamp.iloc[detect_idx] if timestamp is not None else pd.NaT
        )
        event["eqhl_r_trend_state_on_event"] = float(trend_state.iloc[detect_idx])
        event["eqhl_r_trend_bias_state_on_event"] = float(trend_bias.iloc[detect_idx])
        event["eqhl_r_bos_direction_on_event"] = float(bos_direction.iloc[detect_idx])
        event["eqhl_r_choch_direction_on_event"] = float(
            choch_direction.iloc[detect_idx]
        )
        event["eqhl_r_session_on_event"] = str(session.iloc[detect_idx])
        event["eqhl_r_regime_on_event"] = float(regime.iloc[detect_idx])
        event["eqhl_r_volume_ratio_on_event"] = float(vol_ratio.iloc[detect_idx])
        event["eqhl_r_adx_on_event"] = float(adx.iloc[detect_idx])
        event["eqhl_r_rsi_on_event"] = float(rsi.iloc[detect_idx])

        first_retest_idx = np.nan
        first_retest_delay = np.nan

        for horizon in horizons:
            if detect_idx + horizon >= len(df):
                if horizon in FOLLOW_THROUGH_HORIZONS:
                    event[f"eqhl_r_hold_{horizon}"] = np.nan
                    event[f"eqhl_r_failed_{horizon}"] = np.nan
                    event[f"eqhl_r_swept_{horizon}"] = np.nan
                event[f"eqhl_r_mfe_{horizon}_atr"] = np.nan
                event[f"eqhl_r_mae_{horizon}_atr"] = np.nan
                event[f"eqhl_r_excursion_ratio_{horizon}"] = np.nan
                continue

            future_high = _future_window(high, detect_idx=detect_idx, horizon=horizon)
            future_low = _future_window(low, detect_idx=detect_idx, horizon=horizon)
            future_close = _future_window(close, detect_idx=detect_idx, horizon=horizon)

            if direction == -1:
                hold = bool(np.all(future_close <= zone_high))
                failed = bool(np.any(future_close > zone_high))
                swept = bool(np.any(future_high > zone_high))
                mfe = _non_negative(detect_close - float(np.min(future_low)))
                mae = _non_negative(float(np.max(future_high)) - detect_close)
            else:
                hold = bool(np.all(future_close >= zone_low))
                failed = bool(np.any(future_close < zone_low))
                swept = bool(np.any(future_low < zone_low))
                mfe = _non_negative(float(np.max(future_high)) - detect_close)
                mae = _non_negative(detect_close - float(np.min(future_low)))

            if horizon in FOLLOW_THROUGH_HORIZONS:
                event[f"eqhl_r_hold_{horizon}"] = int(hold)
                event[f"eqhl_r_failed_{horizon}"] = int(failed)
                event[f"eqhl_r_swept_{horizon}"] = int(swept)

            event[f"eqhl_r_mfe_{horizon}_atr"] = _safe_ratio(mfe, atr_detect)
            event[f"eqhl_r_mae_{horizon}_atr"] = _safe_ratio(mae, atr_detect)
            event[f"eqhl_r_excursion_ratio_{horizon}"] = _safe_ratio(
                event[f"eqhl_r_mfe_{horizon}_atr"],
                event[f"eqhl_r_mae_{horizon}_atr"],
            )

        for horizon, thresholds in CONTINUATION_THRESHOLDS.items():
            mfe_value = float(event.get(f"eqhl_r_mfe_{horizon}_atr", np.nan))
            for threshold in thresholds:
                event[f"eqhl_r_continuation_{horizon}_{threshold:.1f}atr"] = (
                    int(np.isfinite(mfe_value) and mfe_value >= threshold)
                    if np.isfinite(mfe_value)
                    else np.nan
                )

        retest_mask_all: list[bool] = []
        for idx in range(detect_idx + 1, len(df)):
            overlap = _zone_overlap(high[idx], low[idx], zone_low, zone_high)
            if direction == -1:
                valid_retest = overlap and close[idx] <= zone_high
            else:
                valid_retest = overlap and close[idx] >= zone_low
            retest_mask_all.append(bool(valid_retest))
            if np.isnan(first_retest_idx) and valid_retest:
                first_retest_idx = float(idx)
                first_retest_delay = float(idx - detect_idx)

        event["eqhl_r_first_retest_idx"] = first_retest_idx
        event["eqhl_r_first_retest_delay"] = first_retest_delay

        for horizon in RETEST_HORIZONS:
            if detect_idx + horizon >= len(df):
                event[f"eqhl_r_retest_ever_{horizon}"] = np.nan
                event[f"eqhl_r_retest_count_{horizon}"] = np.nan
                event[f"eqhl_r_mfe_from_retest_{horizon}_atr"] = np.nan
                event[f"eqhl_r_mae_from_retest_{horizon}_atr"] = np.nan
                event[f"eqhl_r_hold_after_retest_{horizon}"] = np.nan
                continue

            retest_flags = np.asarray(retest_mask_all[:horizon], dtype=bool)
            event[f"eqhl_r_retest_ever_{horizon}"] = int(retest_flags.any())
            event[f"eqhl_r_retest_count_{horizon}"] = int(retest_flags.sum())

            if np.isnan(first_retest_idx) or int(first_retest_idx) + horizon >= len(df):
                event[f"eqhl_r_mfe_from_retest_{horizon}_atr"] = np.nan
                event[f"eqhl_r_mae_from_retest_{horizon}_atr"] = np.nan
                event[f"eqhl_r_hold_after_retest_{horizon}"] = np.nan
                continue

            retest_idx = int(first_retest_idx)
            retest_close = float(close[retest_idx])
            future_high = high[retest_idx + 1 : retest_idx + horizon + 1]
            future_low = low[retest_idx + 1 : retest_idx + horizon + 1]
            future_close = close[retest_idx + 1 : retest_idx + horizon + 1]

            if direction == -1:
                mfe = _non_negative(retest_close - float(np.min(future_low)))
                mae = _non_negative(float(np.max(future_high)) - retest_close)
                hold = bool(np.all(future_close <= zone_high))
            else:
                mfe = _non_negative(float(np.max(future_high)) - retest_close)
                mae = _non_negative(retest_close - float(np.min(future_low)))
                hold = bool(np.all(future_close >= zone_low))

            event[f"eqhl_r_mfe_from_retest_{horizon}_atr"] = _safe_ratio(
                mfe, atr_detect
            )
            event[f"eqhl_r_mae_from_retest_{horizon}_atr"] = _safe_ratio(
                mae, atr_detect
            )
            event[f"eqhl_r_hold_after_retest_{horizon}"] = int(hold)

        for horizon in OUTCOME_HORIZONS:
            if detect_idx + horizon >= len(df):
                event[f"eqhl_r_final_outcome_{horizon}"] = "insufficient_horizon"
                continue

            failed = (
                bool(event.get(f"eqhl_r_failed_{min(horizon, 10)}", 0))
                if horizon in FOLLOW_THROUGH_HORIZONS
                else bool(
                    np.any(
                        _future_window(close, detect_idx=detect_idx, horizon=horizon)
                        > zone_high
                    )
                    if direction == -1
                    else np.any(
                        _future_window(close, detect_idx=detect_idx, horizon=horizon)
                        < zone_low
                    )
                )
            )
            swept = (
                bool(event.get(f"eqhl_r_swept_{min(horizon, 10)}", 0))
                if horizon in FOLLOW_THROUGH_HORIZONS
                else bool(
                    np.any(
                        _future_window(high, detect_idx=detect_idx, horizon=horizon)
                        > zone_high
                    )
                    if direction == -1
                    else np.any(
                        _future_window(low, detect_idx=detect_idx, horizon=horizon)
                        < zone_low
                    )
                )
            )
            hold = (
                bool(event.get(f"eqhl_r_hold_{min(horizon, 10)}", 0))
                if horizon in FOLLOW_THROUGH_HORIZONS
                else not failed
            )
            mfe_value = float(event.get(f"eqhl_r_mfe_{horizon}_atr", np.nan))
            retest_count = float(
                event.get(f"eqhl_r_retest_count_{min(horizon, 10)}", np.nan)
            )

            if failed:
                outcome = "failed"
            elif swept:
                outcome = "swept"
            elif (
                hold
                and np.isfinite(mfe_value)
                and mfe_value >= 1.0
                and retest_count == 0
            ):
                outcome = "clean_reaction"
            elif np.isfinite(mfe_value) and mfe_value >= 1.0:
                outcome = "dirty_reaction"
            elif np.isfinite(mfe_value) and mfe_value >= 0.5:
                outcome = "weak_reaction"
            else:
                outcome = "chop"
            event[f"eqhl_r_final_outcome_{horizon}"] = outcome

        quality_score = float(
            0.30 * _clip01(float(event["eqhl_r_touch_count"]) / 4.0)
            + 0.25 * (1.0 - _clip01(float(event["eqhl_r_width_atr"]) / WIDTH_REF))
            + 0.20 * _clip01(float(event["eqhl_r_structural_score"]))
            + 0.15 * (1.0 - _clip01(float(event["eqhl_r_formation_delay"]) / AGE_CAP))
            + 0.10 * (1.0 - _clip01(float(event["eqhl_r_span"]) / SPAN_CAP))
        )
        follow_through_score = float(
            0.45 * _clip01(float(event.get("eqhl_r_mfe_5_atr", np.nan)) / 2.0)
            + 0.25 * (1.0 - _clip01(float(event.get("eqhl_r_mae_5_atr", np.nan)) / 1.5))
            + 0.15 * float(event.get("eqhl_r_hold_5", 0) or 0)
            + 0.15 * float(event.get("eqhl_r_continuation_5_1.0atr", 0) or 0)
        )

        retest_reaction_score = _renormalized_weighted_score(
            [
                (
                    0.45,
                    _clip01(
                        float(event.get("eqhl_r_mfe_from_retest_5_atr", np.nan)) / 1.5
                    ),
                ),
                (
                    0.35,
                    1.0
                    - _clip01(
                        float(event.get("eqhl_r_mae_from_retest_5_atr", np.nan)) / 1.0
                    ),
                ),
                (0.20, float(event.get("eqhl_r_hold_after_retest_5", np.nan))),
            ]
        )

        tradeable_score = _renormalized_weighted_score(
            [
                (0.45, quality_score),
                (0.35, follow_through_score),
                (0.20, retest_reaction_score),
            ]
        )
        failure_severity_score = float(
            0.40 * float(event.get("eqhl_r_failed_5", 0) or 0)
            + 0.25 * _clip01(float(event.get("eqhl_r_mae_5_atr", np.nan)) / 1.5)
            + 0.20 * (1.0 - _clip01(float(event.get("eqhl_r_mfe_5_atr", np.nan)) / 1.0))
            + 0.15 * float(event.get("eqhl_r_swept_5", 0) or 0)
        )

        event["eqhl_r_retest_reaction_score"] = retest_reaction_score
        event["eqhl_r_quality_score"] = quality_score
        event["eqhl_r_follow_through_score"] = follow_through_score
        event["eqhl_r_tradeable_score"] = tradeable_score
        event["eqhl_r_failure_severity_score"] = failure_severity_score

        rows.append(event)

    result = pd.DataFrame(rows)
    for column in EQHL_RESEARCH_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan
    return (
        result[EQHL_RESEARCH_COLUMNS]
        .sort_values(["eqhl_r_detect_idx", "eqhl_event_id"])
        .reset_index(drop=True)
    )


def summarize_equal_hl_research(research_table: pd.DataFrame) -> dict[str, object]:
    if research_table.empty:
        return {
            "event_count": 0,
            "eqh_count": 0,
            "eql_count": 0,
        }

    summary: dict[str, object] = {
        "event_count": int(len(research_table)),
        "eqh_count": int((research_table["eqhl_r_side"] == "eqh").sum()),
        "eql_count": int((research_table["eqhl_r_side"] == "eql").sum()),
        "touch_count_distribution": (
            research_table["eqhl_r_touch_count"]
            .map(_bucket_touch_count)
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "width_atr_stats": _continuous_stats(research_table["eqhl_r_width_atr"]),
        "touch_count_stats": _continuous_stats(research_table["eqhl_r_touch_count"]),
        "score_stats": _continuous_stats(research_table["eqhl_r_score"]),
        "structural_score_stats": _continuous_stats(
            research_table["eqhl_r_structural_score"]
        ),
        "tradeable_live_score_on_detect_stats": _continuous_stats(
            research_table["eqhl_r_tradeable_live_score_on_detect"]
        ),
        "quality_score_stats": _continuous_stats(
            research_table["eqhl_r_quality_score"]
        ),
        "tradeable_score_stats": _continuous_stats(
            research_table["eqhl_r_tradeable_score"]
        ),
        "failure_severity_score_stats": _continuous_stats(
            research_table["eqhl_r_failure_severity_score"]
        ),
    }

    follow_summary: dict[str, object] = {}
    for horizon in FOLLOW_THROUGH_HORIZONS:
        follow_summary[str(horizon)] = {
            "hold_rate": _bool_rate(research_table[f"eqhl_r_hold_{horizon}"]),
            "failed_rate": _bool_rate(research_table[f"eqhl_r_failed_{horizon}"]),
            "swept_rate": _bool_rate(research_table[f"eqhl_r_swept_{horizon}"]),
        }
    summary["follow_through_rates"] = follow_summary

    continuation_summary: dict[str, float] = {}
    for horizon, thresholds in CONTINUATION_THRESHOLDS.items():
        for threshold in thresholds:
            column = f"eqhl_r_continuation_{horizon}_{threshold:.1f}atr"
            continuation_summary[column] = _bool_rate(research_table[column])
    summary["continuation_rates"] = continuation_summary

    outcome_summary: dict[str, object] = {}
    for horizon in OUTCOME_HORIZONS:
        column = f"eqhl_r_final_outcome_{horizon}"
        outcome_summary[str(horizon)] = (
            research_table[column].value_counts(dropna=False).sort_index().to_dict()
        )
    summary["outcome_distribution"] = outcome_summary

    bucketed = research_table.assign(
        touch_bucket=research_table["eqhl_r_touch_count"].map(_bucket_touch_count),
        width_bucket=research_table["eqhl_r_width_atr"].map(_bucket_width),
        formation_delay_bucket=research_table["eqhl_r_formation_delay"].map(
            _bucket_age
        ),
    )
    breakdowns: dict[str, object] = {}
    for name, column in (
        ("trend_state", "eqhl_r_trend_state_on_event"),
        ("session", "eqhl_r_session_on_event"),
        ("regime", "eqhl_r_regime_on_event"),
        ("touch_bucket", "touch_bucket"),
        ("width_bucket", "width_bucket"),
        ("formation_delay_bucket", "formation_delay_bucket"),
    ):
        grouped = (
            bucketed.groupby(column, dropna=False)[
                ["eqhl_r_quality_score", "eqhl_r_tradeable_score"]
            ]
            .mean()
            .round(6)
        )
        breakdowns[name] = grouped.to_dict(orient="index")
    summary["breakdowns"] = breakdowns
    return summary
