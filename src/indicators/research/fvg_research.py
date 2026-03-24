from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
CONTINUATION_THRESHOLDS_ATR = (0.5, 1.0, 1.5, 2.0)
CLEANLINESS_SUMMARY_CAP = 5.0
FINAL_OUTCOME_HORIZON = 20
CLEANLINESS_BUCKET_ORDER = ("dirty", "mixed", "clean", "very_clean", "unknown")
FINAL_OUTCOME_ORDER = ("invalidated", "full_fill", "expired", "merged", "unresolved")
CWT_ATR_THRESHOLDS = (0.5, 1.0, 1.5)
CWT_WIDTH_THRESHOLDS = (1.0, 1.5)
DEEP_PARTIAL_FILL_THRESHOLD = 0.5
LATE_TRIGGER_MIN_BARS = 15
RECOMMENDED_CWT_THRESHOLD_ATR = 1.5
RECOMMENDED_CWT_HORIZON = 10
TERMINAL_STATE_TO_LABEL = {
    0: "unresolved",
    4: "full_fill",
    5: "invalidated",
    6: "expired",
    7: "merged",
}
CORE_TERMINAL_LABEL_ORDER = (
    "full_fill",
    "invalidated",
    "expired",
    "merged",
    "unresolved",
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    return float(num / max(den, EPS))


def _bucket_width(width_atr: float) -> str:
    if not np.isfinite(width_atr):
        return "unknown"
    if width_atr < 0.25:
        return "xs"
    if width_atr < 0.50:
        return "s"
    if width_atr < 0.75:
        return "m"
    if width_atr < 1.25:
        return "l"
    return "xl"


def _bucket_cleanliness(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.20:
        return "dirty"
    if value < 0.50:
        return "mixed"
    if value < 0.90:
        return "clean"
    return "very_clean"


def _cap_cleanliness(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(min(value, CLEANLINESS_SUMMARY_CAP))


def _bucket_touch_delay(delay: float) -> str:
    if not np.isfinite(delay):
        return "untouched"
    if delay <= 1:
        return "immediate"
    if delay <= 3:
        return "fast"
    if delay <= 10:
        return "delayed"
    return "late"


def _bucket_trigger_delay(delay: float) -> str:
    if not np.isfinite(delay):
        return "unknown"
    if delay <= 3:
        return "1_3"
    if delay <= 7:
        return "4_7"
    if delay <= 12:
        return "8_12"
    return "13_20"


def _terminal_label_from_state(state: int | float) -> str:
    if not np.isfinite(state):
        return "unknown"
    return TERMINAL_STATE_TO_LABEL.get(int(state), "unknown")


def _extract_events(
    df: pd.DataFrame,
    debug_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    bull = df[df["fvg_bull_detect_flag"] == 1].copy()
    bear = df[df["fvg_bear_detect_flag"] == 1].copy()

    bull_events = pd.DataFrame(
        {
            "fvg_event_id": bull["fvg_bull_event_id"].astype(int),
            "fvg_side": "bull",
            "fvg_detect_idx": bull.index.astype(int),
            "fvg_zone_low": bull["fvg_bull_low"].astype(float),
            "fvg_zone_high": bull["fvg_bull_high"].astype(float),
            "fvg_zone_mid": bull["fvg_bull_mid"].astype(float),
            "fvg_width": bull["fvg_bull_width"].astype(float),
            "fvg_width_atr": bull["fvg_bull_width_atr"].astype(float),
            "fvg_gap_cleanliness": bull["fvg_bull_gap_cleanliness"].astype(float),
            "fvg_gap_cleanliness_capped": bull["fvg_bull_gap_cleanliness"]
            .astype(float)
            .map(_cap_cleanliness),
            "fvg_terminal_state": bull["fvg_bull_terminal_state"].astype(int),
            "fvg_first_touch_idx": bull["fvg_bull_first_touch_idx"].astype(float),
            "fvg_first_partial_fill_idx": bull[
                "fvg_bull_first_partial_fill_idx"
            ].astype(float),
            "fvg_full_fill_idx": bull["fvg_bull_full_fill_idx"].astype(float),
            "fvg_invalidation_idx": bull["fvg_bull_invalidation_idx"].astype(float),
            "fvg_expiry_idx": bull["fvg_bull_expiry_idx"].astype(float),
        }
    )
    bear_events = pd.DataFrame(
        {
            "fvg_event_id": bear["fvg_bear_event_id"].astype(int),
            "fvg_side": "bear",
            "fvg_detect_idx": bear.index.astype(int),
            "fvg_zone_low": bear["fvg_bear_low"].astype(float),
            "fvg_zone_high": bear["fvg_bear_high"].astype(float),
            "fvg_zone_mid": bear["fvg_bear_mid"].astype(float),
            "fvg_width": bear["fvg_bear_width"].astype(float),
            "fvg_width_atr": bear["fvg_bear_width_atr"].astype(float),
            "fvg_gap_cleanliness": bear["fvg_bear_gap_cleanliness"].astype(float),
            "fvg_gap_cleanliness_capped": bear["fvg_bear_gap_cleanliness"]
            .astype(float)
            .map(_cap_cleanliness),
            "fvg_terminal_state": bear["fvg_bear_terminal_state"].astype(int),
            "fvg_first_touch_idx": bear["fvg_bear_first_touch_idx"].astype(float),
            "fvg_first_partial_fill_idx": bear[
                "fvg_bear_first_partial_fill_idx"
            ].astype(float),
            "fvg_full_fill_idx": bear["fvg_bear_full_fill_idx"].astype(float),
            "fvg_invalidation_idx": bear["fvg_bear_invalidation_idx"].astype(float),
            "fvg_expiry_idx": bear["fvg_bear_expiry_idx"].astype(float),
        }
    )
    events = pd.concat([bull_events, bear_events], ignore_index=True)

    if debug_tables is not None and not debug_tables["event_table"].empty:
        debug_events = debug_tables["event_table"].rename(
            columns={
                "event_id": "fvg_event_id",
                "side": "fvg_side",
                "origin_idx": "fvg_origin_idx",
                "detect_idx": "fvg_detect_idx",
                "low": "fvg_zone_low",
                "high": "fvg_zone_high",
                "mid": "fvg_zone_mid",
                "width": "fvg_width",
                "width_atr": "fvg_width_atr",
                "terminal_state": "fvg_terminal_state",
                "merge_idx": "fvg_merge_idx",
                "terminal_idx": "fvg_terminal_idx",
            }
        )
        keep = [
            "fvg_event_id",
            "fvg_side",
            "fvg_origin_idx",
            "fvg_detect_idx",
            "fvg_merge_idx",
            "fvg_terminal_idx",
        ]
        events = events.merge(
            debug_events[keep],
            on=["fvg_event_id", "fvg_side", "fvg_detect_idx"],
            how="left",
        )
    else:
        events["fvg_origin_idx"] = events["fvg_detect_idx"] - 1
        events["fvg_merge_idx"] = np.nan
        events["fvg_terminal_idx"] = np.nan

    events["fvg_detect_ts"] = pd.to_datetime(
        df.loc[events["fvg_detect_idx"], "timestamp"].to_numpy(), utc=True
    )
    events["fvg_terminal_label"] = events["fvg_terminal_state"].map(
        _terminal_label_from_state
    )
    if "trend_state" in df.columns:
        events["fvg_r_trend_state_on_detect"] = df.loc[
            events["fvg_detect_idx"], "trend_state"
        ].to_numpy(dtype=float)
    else:
        events["fvg_r_trend_state_on_detect"] = np.nan
    if "bos_direction" in df.columns:
        events["fvg_r_bos_direction_on_detect"] = df.loc[
            events["fvg_detect_idx"], "bos_direction"
        ].to_numpy(dtype=float)
    else:
        events["fvg_r_bos_direction_on_detect"] = np.nan
    if "choch_direction" in df.columns:
        events["fvg_r_choch_direction_on_detect"] = df.loc[
            events["fvg_detect_idx"], "choch_direction"
        ].to_numpy(dtype=float)
    else:
        events["fvg_r_choch_direction_on_detect"] = np.nan
    if "displacement_candle" in df.columns:
        events["fvg_r_displacement_flag_on_detect"] = df.loc[
            events["fvg_detect_idx"], "displacement_candle"
        ].to_numpy(dtype=float)
    else:
        events["fvg_r_displacement_flag_on_detect"] = np.nan
    if "session" in df.columns:
        events["fvg_r_session_on_detect"] = (
            df.loc[events["fvg_detect_idx"], "session"].astype(str).to_numpy()
        )
    else:
        events["fvg_r_session_on_detect"] = "unknown"

    events["fvg_r_width_bucket"] = events["fvg_width_atr"].map(_bucket_width)
    events["fvg_r_cleanliness_bucket"] = events["fvg_gap_cleanliness_capped"].map(
        _bucket_cleanliness
    )
    return events.sort_values(["fvg_detect_idx", "fvg_event_id"]).reset_index(drop=True)


def build_fvg_research_table(
    df: pd.DataFrame,
    *,
    debug_tables: dict[str, pd.DataFrame] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    events = _extract_events(df, debug_tables=debug_tables)
    if events.empty:
        return events

    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr = (
        df["atr_14"].to_numpy(dtype=float)
        if "atr_14" in df.columns
        else np.full(len(df), np.nan)
    )

    output_rows: list[dict[str, object]] = []
    for row in events.itertuples(index=False):
        event: dict[str, object] = dict(row._asdict())
        detect_idx = int(event["fvg_detect_idx"])
        side = str(event["fvg_side"])
        zone_low = float(event["fvg_zone_low"])
        zone_high = float(event["fvg_zone_high"])
        zone_width = float(event["fvg_width"])
        atr_detect = float(atr[detect_idx]) if detect_idx < len(atr) else np.nan
        detect_close = float(c[detect_idx])
        final_horizon = FINAL_OUTCOME_HORIZON
        final_eval_stop = detect_idx + final_horizon
        has_sufficient_forward_horizon = final_eval_stop < len(df)
        event["fvg_r_has_sufficient_forward_horizon"] = bool(
            has_sufficient_forward_horizon
        )
        terminal_state = int(event["fvg_terminal_state"])
        terminal_label = _terminal_label_from_state(terminal_state)
        event["fvg_terminal_label"] = terminal_label
        event["fvg_r_core_full_fill_flag"] = bool(terminal_label == "full_fill")
        event["fvg_r_core_invalidated_flag"] = bool(terminal_label == "invalidated")
        event["fvg_r_core_expired_flag"] = bool(terminal_label == "expired")
        event["fvg_r_core_merged_flag"] = bool(terminal_label == "merged")
        event["fvg_r_core_still_active_at_dataset_end_flag"] = bool(
            terminal_label == "unresolved"
        )
        touch_idx = (
            int(event["fvg_first_touch_idx"])
            if np.isfinite(event["fvg_first_touch_idx"])
            else None
        )
        event["fvg_r_ever_touched"] = bool(touch_idx is not None)
        event["fvg_r_first_touch_idx"] = (
            float(touch_idx) if touch_idx is not None else np.nan
        )
        first_partial_fill_idx = (
            int(event["fvg_first_partial_fill_idx"])
            if np.isfinite(event["fvg_first_partial_fill_idx"])
            else None
        )
        event["fvg_r_ever_partial_fill"] = bool(first_partial_fill_idx is not None)

        if side == "bull":
            invalidate_boundary = zone_low
            protected_close = zone_high
        else:
            invalidate_boundary = zone_high
            protected_close = zone_low

        event["fvg_r_time_to_first_touch"] = (
            float(touch_idx - detect_idx) if touch_idx is not None else np.nan
        )
        event["fvg_r_time_to_first_partial_fill"] = (
            float(event["fvg_first_partial_fill_idx"] - detect_idx)
            if np.isfinite(event["fvg_first_partial_fill_idx"])
            else np.nan
        )
        event["fvg_r_time_to_full_fill"] = (
            float(event["fvg_full_fill_idx"] - detect_idx)
            if np.isfinite(event["fvg_full_fill_idx"])
            else np.nan
        )
        event["fvg_r_time_to_invalidation"] = (
            float(event["fvg_invalidation_idx"] - detect_idx)
            if np.isfinite(event["fvg_invalidation_idx"])
            else np.nan
        )
        event["fvg_r_touch_delay_bucket"] = _bucket_touch_delay(
            event["fvg_r_time_to_first_touch"]
        )

        if touch_idx is not None:
            if side == "bull":
                first_touch_depth = _clip01(
                    (zone_high - l[touch_idx]) / max(zone_width, EPS)
                )
            else:
                first_touch_depth = _clip01(
                    (h[touch_idx] - zone_low) / max(zone_width, EPS)
                )
        else:
            first_touch_depth = np.nan
        event["fvg_r_first_retest_depth_frac"] = first_touch_depth

        terminal_candidates = {
            "invalidated": (
                float(event["fvg_invalidation_idx"])
                if np.isfinite(event["fvg_invalidation_idx"])
                else np.nan
            ),
            "full_fill": (
                float(event["fvg_full_fill_idx"])
                if np.isfinite(event["fvg_full_fill_idx"])
                else np.nan
            ),
            "merged": (
                float(event["fvg_merge_idx"])
                if np.isfinite(event["fvg_merge_idx"])
                else np.nan
            ),
            "expired": (
                float(event["fvg_expiry_idx"])
                if np.isfinite(event["fvg_expiry_idx"])
                else np.nan
            ),
        }
        finite_terminal = {
            label: idx for label, idx in terminal_candidates.items() if np.isfinite(idx)
        }
        first_terminal_label = None
        first_terminal_idx = np.nan
        if finite_terminal:
            first_terminal_label, first_terminal_idx = min(
                finite_terminal.items(), key=lambda item: item[1]
            )
            first_terminal_idx = float(first_terminal_idx)
        event["fvg_r_first_terminal_idx"] = first_terminal_idx
        event["fvg_r_first_terminal_label"] = (
            first_terminal_label if first_terminal_label is not None else "none"
        )

        if np.isfinite(first_terminal_idx) and first_terminal_idx > detect_idx:
            term_slice = slice(detect_idx + 1, int(first_terminal_idx) + 1)
            if side == "bull":
                max_fill_before_terminal = _clip01(
                    (zone_high - np.nanmin(l[term_slice])) / max(zone_width, EPS)
                )
            else:
                max_fill_before_terminal = _clip01(
                    (np.nanmax(h[term_slice]) - zone_low) / max(zone_width, EPS)
                )
        else:
            max_fill_before_terminal = 0.0
        event["fvg_r_max_fill_pct_before_terminal"] = max_fill_before_terminal
        event["fvg_r_deep_partial_fill_before_terminal"] = bool(
            max_fill_before_terminal >= DEEP_PARTIAL_FILL_THRESHOLD
        )

        continuation_without_touch_idx = np.nan
        cwt_trigger_indices: dict[str, float] = {}
        if (
            has_sufficient_forward_horizon
            and touch_idx is None
            and np.isfinite(atr_detect)
        ):
            for future_idx in range(detect_idx + 1, final_eval_stop + 1):
                if side == "bull":
                    favorable_move = h[future_idx] - detect_close
                else:
                    favorable_move = detect_close - l[future_idx]

                for threshold in CWT_ATR_THRESHOLDS:
                    key = f"atr_{threshold:g}"
                    if key not in cwt_trigger_indices and favorable_move >= (
                        threshold * atr_detect
                    ):
                        cwt_trigger_indices[key] = float(future_idx)

                for threshold in CWT_WIDTH_THRESHOLDS:
                    key = f"width_{threshold:g}"
                    if key not in cwt_trigger_indices and favorable_move >= (
                        threshold * zone_width
                    ):
                        cwt_trigger_indices[key] = float(future_idx)

                if "atr_1" in cwt_trigger_indices and not np.isfinite(
                    continuation_without_touch_idx
                ):
                    continuation_without_touch_idx = cwt_trigger_indices["atr_1"]
                if len(cwt_trigger_indices) == len(CWT_ATR_THRESHOLDS) + len(
                    CWT_WIDTH_THRESHOLDS
                ):
                    break
        event["fvg_r_continuation_without_touch_idx"] = continuation_without_touch_idx
        event["fvg_r_time_to_continuation_without_touch"] = (
            float(continuation_without_touch_idx - detect_idx)
            if np.isfinite(continuation_without_touch_idx)
            else np.nan
        )
        event["fvg_r_cwt_trigger_delay_bucket"] = _bucket_trigger_delay(
            event["fvg_r_time_to_continuation_without_touch"]
        )
        for threshold in CWT_ATR_THRESHOLDS:
            key = f"atr_{threshold:g}"
            idx_val = cwt_trigger_indices.get(key, np.nan)
            event[f"fvg_r_cwt_trigger_idx_{key}"] = idx_val
            event[f"fvg_r_cwt_time_{key}"] = (
                float(idx_val - detect_idx) if np.isfinite(idx_val) else np.nan
            )
            event[f"fvg_r_cwt_flag_{key}"] = bool(np.isfinite(idx_val))
        for threshold in CWT_WIDTH_THRESHOLDS:
            key = f"width_{threshold:g}"
            idx_val = cwt_trigger_indices.get(key, np.nan)
            event[f"fvg_r_cwt_trigger_idx_{key}"] = idx_val
            event[f"fvg_r_cwt_time_{key}"] = (
                float(idx_val - detect_idx) if np.isfinite(idx_val) else np.nan
            )
            event[f"fvg_r_cwt_flag_{key}"] = bool(np.isfinite(idx_val))

        explicit_no_early_failure = bool(
            np.isfinite(continuation_without_touch_idx)
            and (
                not np.isfinite(first_terminal_idx)
                or first_terminal_idx > continuation_without_touch_idx
            )
        )
        event["fvg_r_cwt_flag_1atr_no_early_failure"] = explicit_no_early_failure
        event["fvg_r_cwt_recommended_flag"] = bool(
            np.isfinite(event["fvg_r_cwt_trigger_idx_atr_1.5"])
            and event["fvg_r_cwt_time_atr_1.5"] <= RECOMMENDED_CWT_HORIZON
            and (
                not np.isfinite(first_terminal_idx)
                or first_terminal_idx > event["fvg_r_cwt_trigger_idx_atr_1.5"]
            )
        )
        event["fvg_r_continuation_without_touch_flag"] = bool(
            np.isfinite(continuation_without_touch_idx) and explicit_no_early_failure
        )
        event["fvg_r_continuation_without_touch_recommended_flag"] = bool(
            event["fvg_r_cwt_recommended_flag"]
        )

        for horizon in horizons:
            detect_stop = min(len(df), detect_idx + horizon + 1)
            detect_slice = slice(detect_idx + 1, detect_stop)
            if detect_stop > detect_idx + 1:
                future_h = h[detect_slice]
                future_l = l[detect_slice]
                if side == "bull":
                    detect_mfe = (np.nanmax(future_h) - detect_close) / max(
                        atr_detect, EPS
                    )
                    detect_mae = (detect_close - np.nanmin(future_l)) / max(
                        atr_detect, EPS
                    )
                    fill_snapshot = _clip01(
                        (zone_high - future_l[-1]) / max(zone_width, EPS)
                    )
                    max_fill = _clip01(
                        (zone_high - np.nanmin(future_l)) / max(zone_width, EPS)
                    )
                else:
                    detect_mfe = (detect_close - np.nanmin(future_l)) / max(
                        atr_detect, EPS
                    )
                    detect_mae = (np.nanmax(future_h) - detect_close) / max(
                        atr_detect, EPS
                    )
                    fill_snapshot = _clip01(
                        (future_h[-1] - zone_low) / max(zone_width, EPS)
                    )
                    max_fill = _clip01(
                        (np.nanmax(future_h) - zone_low) / max(zone_width, EPS)
                    )
            else:
                detect_mfe = np.nan
                detect_mae = np.nan
                fill_snapshot = np.nan
                max_fill = np.nan

            event[f"fvg_r_fill_pct_{horizon}"] = fill_snapshot
            event[f"fvg_r_max_fill_pct_{horizon}"] = max_fill
            event[f"fvg_r_mfe_from_detect_atr_{horizon}"] = detect_mfe
            event[f"fvg_r_mae_from_detect_atr_{horizon}"] = detect_mae
            event[f"fvg_r_excursion_ratio_detect_{horizon}"] = _safe_ratio(
                detect_mfe, detect_mae
            )

            if touch_idx is not None:
                touch_close = float(c[touch_idx])
                atr_touch = (
                    float(atr[touch_idx]) if np.isfinite(atr[touch_idx]) else atr_detect
                )
                touch_stop = min(len(df), touch_idx + horizon + 1)
                touch_slice = slice(touch_idx + 1, touch_stop)
                if touch_stop > touch_idx + 1:
                    touch_h = h[touch_slice]
                    touch_l = l[touch_slice]
                    touch_c = c[touch_slice]
                    if side == "bull":
                        mfe_touch = (np.nanmax(touch_h) - touch_close) / max(
                            atr_touch, EPS
                        )
                        mae_touch = (touch_close - np.nanmin(touch_l)) / max(
                            atr_touch, EPS
                        )
                        fail = bool(np.any(touch_c < invalidate_boundary))
                    else:
                        mfe_touch = (touch_close - np.nanmin(touch_l)) / max(
                            atr_touch, EPS
                        )
                        mae_touch = (np.nanmax(touch_h) - touch_close) / max(
                            atr_touch, EPS
                        )
                        fail = bool(np.any(touch_c > invalidate_boundary))
                    hold = not fail
                else:
                    mfe_touch = np.nan
                    mae_touch = np.nan
                    hold = np.nan
                    fail = np.nan
            else:
                mfe_touch = np.nan
                mae_touch = np.nan
                hold = np.nan
                fail = np.nan

            event[f"fvg_r_mfe_from_touch_atr_{horizon}"] = mfe_touch
            event[f"fvg_r_mae_from_touch_atr_{horizon}"] = mae_touch
            event[f"fvg_r_excursion_ratio_touch_{horizon}"] = _safe_ratio(
                mfe_touch, mae_touch
            )
            event[f"fvg_r_hold_after_touch_{horizon}"] = hold
            event[f"fvg_r_fail_after_touch_{horizon}"] = fail

            if touch_idx is not None and touch_idx + horizon < len(df):
                if side == "bull":
                    touch_episode = (
                        l[touch_idx : touch_idx + horizon + 1] <= zone_high
                    ).astype(int)
                else:
                    touch_episode = (
                        h[touch_idx : touch_idx + horizon + 1] >= zone_low
                    ).astype(int)
                starts = np.flatnonzero(
                    (touch_episode == 1) & (np.r_[0, touch_episode[:-1]] == 0)
                )
                event[f"fvg_r_retest_count_{horizon}"] = int(len(starts))
                event[f"fvg_r_max_retest_depth_frac_{horizon}"] = event[
                    f"fvg_r_max_fill_pct_{horizon}"
                ]
            else:
                event[f"fvg_r_retest_count_{horizon}"] = np.nan
                event[f"fvg_r_max_retest_depth_frac_{horizon}"] = np.nan

        if touch_idx is not None:
            next_close = c[touch_idx + 1] if touch_idx + 1 < len(df) else np.nan
            event["fvg_r_immediate_reject_after_touch"] = bool(
                c[touch_idx] >= protected_close
                if side == "bull"
                else c[touch_idx] <= protected_close
            ) or bool(
                next_close >= protected_close
                if side == "bull" and np.isfinite(next_close)
                else (
                    next_close <= protected_close
                    if side == "bear" and np.isfinite(next_close)
                    else False
                )
            )
            event["fvg_r_dirty_accept_after_touch"] = (
                bool(
                    event.get("fvg_r_fail_after_touch_3") is True
                    or (
                        np.isfinite(event.get("fvg_r_retest_count_5", np.nan))
                        and event["fvg_r_retest_count_5"] >= 2
                    )
                )
                and not event["fvg_r_immediate_reject_after_touch"]
            )
        else:
            event["fvg_r_immediate_reject_after_touch"] = np.nan
            event["fvg_r_dirty_accept_after_touch"] = np.nan

        for horizon in (5, 10, 20):
            for threshold in CONTINUATION_THRESHOLDS_ATR:
                event[f"fvg_r_continuation_detect_{horizon}_{threshold:g}atr"] = bool(
                    np.isfinite(
                        event.get(f"fvg_r_mfe_from_detect_atr_{horizon}", np.nan)
                    )
                    and event[f"fvg_r_mfe_from_detect_atr_{horizon}"] >= threshold
                )
                event[f"fvg_r_continuation_touch_{horizon}_{threshold:g}atr"] = bool(
                    np.isfinite(
                        event.get(f"fvg_r_mfe_from_touch_atr_{horizon}", np.nan)
                    )
                    and event[f"fvg_r_mfe_from_touch_atr_{horizon}"] >= threshold
                )

        terminal_idx = (
            int(event["fvg_terminal_idx"])
            if np.isfinite(event["fvg_terminal_idx"])
            else len(df) - 1
        )
        if terminal_idx > detect_idx:
            term_slice = slice(detect_idx + 1, terminal_idx + 1)
            if side == "bull":
                mfe_before_terminal = (np.nanmax(h[term_slice]) - detect_close) / max(
                    atr_detect, EPS
                )
            else:
                mfe_before_terminal = (detect_close - np.nanmin(l[term_slice])) / max(
                    atr_detect, EPS
                )
        else:
            mfe_before_terminal = np.nan

        event["fvg_r_continuation_before_full_fill"] = bool(
            np.isfinite(event["fvg_full_fill_idx"])
            and np.isfinite(mfe_before_terminal)
            and mfe_before_terminal >= 1.0
        )
        event["fvg_r_continuation_before_invalidation"] = bool(
            np.isfinite(event["fvg_invalidation_idx"])
            and np.isfinite(mfe_before_terminal)
            and mfe_before_terminal >= 1.0
        )

        final_outcome = terminal_label
        event["fvg_r_final_outcome"] = final_outcome
        event["fvg_r_touched_rejected_flag"] = bool(
            event["fvg_r_immediate_reject_after_touch"] is True
            and touch_idx is not None
        )
        event["fvg_r_untouched_behavior_flag"] = bool(
            has_sufficient_forward_horizon
            and touch_idx is None
            and not event["fvg_r_continuation_without_touch_flag"]
        )
        event["fvg_r_research_eligible_flag"] = True
        event["fvg_r_research_exclusion_reason"] = np.nan
        event["fvg_r_unresolved_due_to_insufficient_forward_horizon_flag"] = bool(
            final_outcome == "unresolved" and not has_sufficient_forward_horizon
        )
        event["fvg_r_exact_one_mapping_or_exclusion_flag"] = bool(
            final_outcome in FINAL_OUTCOME_ORDER
            and pd.isna(event["fvg_r_research_exclusion_reason"])
        )

        for horizon in (5, 10, 20):
            if touch_idx is None or touch_idx + 1 >= len(df):
                event[f"fvg_r_outcome_{horizon}"] = (
                    "continuation_without_touch"
                    if event.get(f"fvg_r_mfe_from_detect_atr_{horizon}", np.nan) >= 1.0
                    else "untouched"
                )
            elif event.get(f"fvg_r_fail_after_touch_{horizon}") is True:
                event[f"fvg_r_outcome_{horizon}"] = "invalidated"
            elif (
                np.isfinite(event["fvg_full_fill_idx"])
                and (event["fvg_full_fill_idx"] - detect_idx) <= horizon
            ):
                event[f"fvg_r_outcome_{horizon}"] = "full_fill"
            elif event["fvg_r_immediate_reject_after_touch"]:
                event[f"fvg_r_outcome_{horizon}"] = "touched_rejected"
            else:
                event[f"fvg_r_outcome_{horizon}"] = "partial_fill_hold"

        event["fvg_r_tradeable_touch_reaction"] = bool(
            touch_idx is not None
            and np.isfinite(event["fvg_r_mfe_from_touch_atr_5"])
            and event["fvg_r_mfe_from_touch_atr_5"] >= 1.0
            and event["fvg_r_mae_from_touch_atr_5"] <= 0.5
            and event["fvg_r_fail_after_touch_5"] is False
        )
        event["fvg_r_strong_continuation"] = bool(
            event["fvg_r_tradeable_touch_reaction"]
            and event["fvg_r_continuation_touch_10_1atr"]
        )
        event["fvg_r_noisy_or_inefficient"] = bool(
            touch_idx is not None
            and np.isfinite(event["fvg_r_mfe_from_touch_atr_5"])
            and event["fvg_r_mfe_from_touch_atr_5"] >= 1.0
            and (
                event["fvg_r_mae_from_touch_atr_5"] > 0.75
                or event.get("fvg_r_retest_count_10", 0) >= 3
            )
        )

        touch_mfe = event.get("fvg_r_mfe_from_touch_atr_5", np.nan)
        touch_mae = event.get("fvg_r_mae_from_touch_atr_5", np.nan)
        hold5 = event.get("fvg_r_hold_after_touch_5", np.nan)
        depth = event.get("fvg_r_first_retest_depth_frac", np.nan)

        event["fvg_r_touch_reaction_score"] = _clip01(
            0.40 * _clip01((touch_mfe if np.isfinite(touch_mfe) else 0.0) / 2.0)
            + 0.25
            * (1.0 - _clip01((touch_mae if np.isfinite(touch_mae) else 1.0) / 2.0))
            + 0.20 * (1.0 if hold5 is True else 0.0)
            + 0.15 * (1.0 - _clip01(depth if np.isfinite(depth) else 1.0))
        )
        event["fvg_r_structural_respect_score"] = _clip01(
            0.30 * (1.0 if hold5 is True else 0.0)
            + 0.20
            * (1.0 if event["fvg_r_immediate_reject_after_touch"] is True else 0.0)
            + 0.20
            * (
                1.0
                - _clip01(
                    event.get("fvg_r_max_fill_pct_10", np.nan)
                    if np.isfinite(event.get("fvg_r_max_fill_pct_10", np.nan))
                    else 1.0
                )
            )
            + 0.15 * (1.0 if event["fvg_r_continuation_before_invalidation"] else 0.0)
            + 0.15
            * _clip01(
                (
                    event.get("fvg_gap_cleanliness", np.nan)
                    if np.isfinite(event.get("fvg_gap_cleanliness", np.nan))
                    else 0.0
                )
            )
        )
        event["fvg_r_tradability_score"] = _clip01(
            0.55 * event["fvg_r_touch_reaction_score"]
            + 0.25 * event["fvg_r_structural_respect_score"]
            + 0.20
            * _clip01(
                (event["fvg_width_atr"] if np.isfinite(event["fvg_width_atr"]) else 0.0)
                / 1.5
            )
        )
        invalidation_speed = (
            1.0 / max(event["fvg_r_time_to_invalidation"], 1.0)
            if np.isfinite(event["fvg_r_time_to_invalidation"])
            else 0.0
        )
        event["fvg_r_failure_severity_score"] = _clip01(
            0.35 * (1.0 if final_outcome == "invalidated" else 0.0)
            + 0.25 * _clip01((touch_mae if np.isfinite(touch_mae) else 0.0) / 2.0)
            + 0.20
            * _clip01(
                event.get("fvg_r_max_fill_pct_10", np.nan)
                if np.isfinite(event.get("fvg_r_max_fill_pct_10", np.nan))
                else 0.0
            )
            + 0.20 * invalidation_speed
        )

        output_rows.append(event)

    return (
        pd.DataFrame(output_rows)
        .sort_values(["fvg_detect_idx", "fvg_event_id"])
        .reset_index(drop=True)
    )


def summarize_fvg_research(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {"event_count": 0}

    def _bool_rate(col: str) -> float:
        ser = events[col]
        ser = ser[ser.notna()]
        return float(ser.mean()) if not ser.empty else np.nan

    def _count_rate(mask: pd.Series, denominator: int) -> dict[str, float | int]:
        count = int(mask.sum())
        return {
            "count": count,
            "rate_total_events": float(count / len(events)) if len(events) else np.nan,
            "rate_of_eligible_never_touched": (
                float(count / denominator) if denominator > 0 else np.nan
            ),
        }

    def _bucket_breakdown(
        frame: pd.DataFrame,
        bucket_col: str,
        flag_mask: pd.Series,
        eligible_mask: pd.Series,
    ) -> dict[str, dict[str, float | int]]:
        # eligible_count is the subgroup denominator, success_count is the
        # subgroup numerator, and rate is NaN when the subgroup is empty.
        scoped = frame.loc[eligible_mask].copy()
        out: dict[str, dict[str, float | int]] = {}
        for bucket in sorted(scoped[bucket_col].drop_duplicates().tolist(), key=str):
            if pd.isna(bucket):
                group = scoped[scoped[bucket_col].isna()]
            else:
                group = scoped[scoped[bucket_col] == bucket]
            eligible_count = int(len(group))
            success_count = (
                int(flag_mask.loc[group.index].sum()) if eligible_count > 0 else 0
            )
            rate = (
                float(success_count / eligible_count) if eligible_count > 0 else np.nan
            )
            label = "NaN" if pd.isna(bucket) else str(bucket)
            out[label] = {
                "eligible_count": eligible_count,
                "success_count": success_count,
                "rate": rate,
            }
        return out

    def _breakdown_count_total(breakdown: dict[str, dict[str, float | int]]) -> int:
        return int(sum(int(bucket["eligible_count"]) for bucket in breakdown.values()))

    def _crosstab_to_dict(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {
            str(row): {str(col): int(val) for col, val in row_vals.items()}
            for row, row_vals in frame.to_dict(orient="index").items()
        }

    width_bucket_hold = (
        events.groupby("fvg_r_width_bucket")["fvg_r_hold_after_touch_5"]
        .mean()
        .dropna()
        .to_dict()
    )

    # "unknown" means cleanliness was not assignable because the value was null
    # or excluded from normal bucketing by the structural null rule.
    eligible_for_cleanliness = events[
        events["fvg_r_final_outcome"] != "unresolved"
    ].copy()
    cleanliness_bucket_tradeable: dict[str, dict[str, float | int]] = {}
    for bucket in CLEANLINESS_BUCKET_ORDER:
        bucket_events = eligible_for_cleanliness[
            eligible_for_cleanliness["fvg_r_cleanliness_bucket"] == bucket
        ]
        # eligible_count is the subgroup denominator, success_count is the
        # subgroup numerator, and rate is NaN when the subgroup is empty.
        eligible_count = int(len(bucket_events))
        success_count = (
            int(bucket_events["fvg_r_tradeable_touch_reaction"].sum())
            if eligible_count > 0
            else 0
        )
        rate = float(success_count / eligible_count) if eligible_count > 0 else np.nan
        cleanliness_bucket_tradeable[bucket] = {
            "eligible_count": eligible_count,
            "success_count": success_count,
            "rate": rate,
        }

    final_outcome_counts = {
        label: int((events["fvg_r_final_outcome"] == label).sum())
        for label in FINAL_OUTCOME_ORDER
    }
    core_terminal_counts = {
        label: int((events["fvg_terminal_label"] == label).sum())
        for label in CORE_TERMINAL_LABEL_ORDER
    }

    never_touched_mask = ~events["fvg_r_ever_touched"].astype(bool)
    touched_mask = events["fvg_r_ever_touched"].astype(bool)
    unresolved_mask = events["fvg_r_final_outcome"] == "unresolved"
    eligible_never_touched_mask = never_touched_mask & events[
        "fvg_r_has_sufficient_forward_horizon"
    ].astype(bool)
    eligible_never_touched_count = int(eligible_never_touched_mask.sum())

    current_cwt_mask = events["fvg_r_continuation_without_touch_flag"].astype(bool)
    recommended_cwt_mask = events[
        "fvg_r_continuation_without_touch_recommended_flag"
    ].astype(bool)
    sensitivity_masks = {
        "current_rule": current_cwt_mask,
        "atr_0.5": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_atr_0.5"].astype(bool)
        & (
            ~np.isfinite(events["fvg_r_first_terminal_idx"])
            | (
                events["fvg_r_first_terminal_idx"]
                > events["fvg_r_cwt_trigger_idx_atr_0.5"]
            )
        ),
        "atr_1.0": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_atr_1"].astype(bool)
        & (
            ~np.isfinite(events["fvg_r_first_terminal_idx"])
            | (
                events["fvg_r_first_terminal_idx"]
                > events["fvg_r_cwt_trigger_idx_atr_1"]
            )
        ),
        "atr_1.5": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_atr_1.5"].astype(bool)
        & (
            ~np.isfinite(events["fvg_r_first_terminal_idx"])
            | (
                events["fvg_r_first_terminal_idx"]
                > events["fvg_r_cwt_trigger_idx_atr_1.5"]
            )
        ),
        "width_1.0": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_width_1"].astype(bool)
        & (
            ~np.isfinite(events["fvg_r_first_terminal_idx"])
            | (
                events["fvg_r_first_terminal_idx"]
                > events["fvg_r_cwt_trigger_idx_width_1"]
            )
        ),
        "width_1.5": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_width_1.5"].astype(bool)
        & (
            ~np.isfinite(events["fvg_r_first_terminal_idx"])
            | (
                events["fvg_r_first_terminal_idx"]
                > events["fvg_r_cwt_trigger_idx_width_1.5"]
            )
        ),
        "atr_1.0_no_early_failure": eligible_never_touched_mask
        & events["fvg_r_cwt_flag_1atr_no_early_failure"].astype(bool),
        "recommended_rule": eligible_never_touched_mask & recommended_cwt_mask,
    }

    current_cwt_events = events.loc[current_cwt_mask].copy()
    cwt_side_breakdown = _bucket_breakdown(
        events, "fvg_side", current_cwt_mask, eligible_never_touched_mask
    )
    cwt_width_breakdown = _bucket_breakdown(
        events, "fvg_r_width_bucket", current_cwt_mask, eligible_never_touched_mask
    )
    cwt_cleanliness_breakdown = _bucket_breakdown(
        events,
        "fvg_r_cleanliness_bucket",
        current_cwt_mask,
        eligible_never_touched_mask,
    )
    cwt_trigger_delay_breakdown = _bucket_breakdown(
        events,
        "fvg_r_cwt_trigger_delay_bucket",
        current_cwt_mask,
        eligible_never_touched_mask,
    )
    cwt_trend_breakdown = (
        _bucket_breakdown(
            events,
            "fvg_r_trend_state_on_detect",
            current_cwt_mask,
            eligible_never_touched_mask,
        )
        if "fvg_r_trend_state_on_detect" in events.columns
        else {}
    )
    continuation_without_touch_audit = {
        "definition": {
            "anchor": "detect_close",
            "threshold": "1.0 ATR favorable move",
            "horizon_bars": FINAL_OUTCOME_HORIZON,
            "requires_never_touched": True,
            "requires_sufficient_forward_horizon": True,
            "requires_earlier_terminal_not_to_precede_trigger": True,
            "trigger_price_basis": "future high for bull / future low for bear",
        },
        "recommended_definition": {
            "anchor": "detect_close",
            "threshold": "1.5 ATR favorable move",
            "horizon_bars": RECOMMENDED_CWT_HORIZON,
            "requires_never_touched": True,
            "requires_sufficient_forward_horizon": True,
            "requires_earlier_terminal_not_to_precede_trigger": True,
            "trigger_price_basis": "future high for bull / future low for bear",
        },
        "current_rule": _count_rate(current_cwt_mask, eligible_never_touched_count),
        "recommended_rule": _count_rate(
            sensitivity_masks["recommended_rule"], eligible_never_touched_count
        ),
        "sensitivity": {
            key: _count_rate(mask, eligible_never_touched_count)
            for key, mask in sensitivity_masks.items()
        },
        "very_small_move_audit": {
            "current_rule_not_meeting_1.5atr_count": int(
                (current_cwt_mask & ~sensitivity_masks["atr_1.5"]).sum()
            ),
            "current_rule_not_meeting_1.0x_width_count": int(
                (current_cwt_mask & ~sensitivity_masks["width_1.0"]).sum()
            ),
            "current_rule_not_meeting_1.5x_width_count": int(
                (current_cwt_mask & ~sensitivity_masks["width_1.5"]).sum()
            ),
            "current_rule_same_row_structural_tie_count": int(
                (
                    current_cwt_mask
                    & np.isfinite(events["fvg_r_first_terminal_idx"])
                    & (
                        events["fvg_r_first_terminal_idx"]
                        == events["fvg_r_cwt_trigger_idx_atr_1"]
                    )
                ).sum()
            ),
        },
        "timing": {
            "late_trigger_count": int(
                (
                    current_cwt_events["fvg_r_time_to_continuation_without_touch"]
                    >= LATE_TRIGGER_MIN_BARS
                ).sum()
            ),
            "mean_trigger_delay": (
                float(
                    current_cwt_events[
                        "fvg_r_time_to_continuation_without_touch"
                    ].mean()
                )
                if not current_cwt_events.empty
                else np.nan
            ),
            "median_trigger_delay": (
                float(
                    current_cwt_events[
                        "fvg_r_time_to_continuation_without_touch"
                    ].median()
                )
                if not current_cwt_events.empty
                else np.nan
            ),
            "p90_trigger_delay": (
                float(
                    current_cwt_events[
                        "fvg_r_time_to_continuation_without_touch"
                    ].quantile(0.9)
                )
                if not current_cwt_events.empty
                else np.nan
            ),
        },
        "breakdowns": {
            "side": cwt_side_breakdown,
            "width_bucket": cwt_width_breakdown,
            "cleanliness_bucket": cwt_cleanliness_breakdown,
            "trigger_delay_bucket": cwt_trigger_delay_breakdown,
            "trend_state_on_detect": cwt_trend_breakdown,
        },
    }

    never_touched_audit = {
        "never_touched_count": int(never_touched_mask.sum()),
        "never_touched_and_continuation_without_touch_count": int(
            current_cwt_mask.sum()
        ),
        "never_touched_and_continuation_without_touch_recommended_count": int(
            recommended_cwt_mask.sum()
        ),
        "never_touched_and_untouched_behavior_count": int(
            events["fvg_r_untouched_behavior_flag"].astype(bool).sum()
        ),
        "never_touched_and_core_expired_count": int(
            (never_touched_mask & (events["fvg_r_final_outcome"] == "expired")).sum()
        ),
        "never_touched_and_core_merged_count": int(
            (never_touched_mask & (events["fvg_r_final_outcome"] == "merged")).sum()
        ),
        "never_touched_and_research_unresolved_count": int(
            (never_touched_mask & unresolved_mask).sum()
        ),
    }
    touched_audit = {
        "touched_count": int(touched_mask.sum()),
        "touched_and_core_invalidated_count": int(
            (touched_mask & (events["fvg_r_final_outcome"] == "invalidated")).sum()
        ),
        "touched_and_core_full_fill_count": int(
            (touched_mask & (events["fvg_r_final_outcome"] == "full_fill")).sum()
        ),
        "touched_and_touched_rejected_count": int(
            (touched_mask & events["fvg_r_touched_rejected_flag"].astype(bool)).sum()
        ),
        "touched_and_core_expired_count": int(
            (touched_mask & (events["fvg_r_final_outcome"] == "expired")).sum()
        ),
        "touched_and_research_unresolved_count": int(
            (touched_mask & unresolved_mask).sum()
        ),
    }

    expired_mask = events["fvg_r_final_outcome"] == "expired"
    expired_events = events.loc[expired_mask].copy()
    expired_touched_only_mask = (
        expired_mask & touched_mask & ~events["fvg_r_ever_partial_fill"].astype(bool)
    )
    expired_partial_fill_mask = expired_mask & events["fvg_r_ever_partial_fill"].astype(
        bool
    )
    expired_deep_partial_fill_mask = expired_mask & events[
        "fvg_r_deep_partial_fill_before_terminal"
    ].astype(bool)
    expired_ages = (
        events.loc[expired_mask, "fvg_expiry_idx"]
        - events.loc[expired_mask, "fvg_detect_idx"]
    )
    expiry_audit = {
        "expired_count": int(expired_mask.sum()),
        "expired_never_touched_count": int((expired_mask & never_touched_mask).sum()),
        "expired_touched_only_count": int(expired_touched_only_mask.sum()),
        "expired_partial_fill_count": int(expired_partial_fill_mask.sum()),
        "expired_deep_partial_fill_count": int(expired_deep_partial_fill_mask.sum()),
        "expired_after_first_touch_count": int((expired_mask & touched_mask).sum()),
        "expired_after_partial_fill_count": int(expired_partial_fill_mask.sum()),
        "expired_age_distribution": {
            "1_12": int(((expired_ages >= 1) & (expired_ages <= 12)).sum()),
            "13_24": int(((expired_ages >= 13) & (expired_ages <= 24)).sum()),
            "25_36": int(((expired_ages >= 25) & (expired_ages <= 36)).sum()),
            "37_48": int(((expired_ages >= 37) & (expired_ages <= 48)).sum()),
        },
        "mean_age_at_expiry": (
            float(expired_ages.mean()) if not expired_events.empty else np.nan
        ),
        "median_age_at_expiry": (
            float(expired_ages.median()) if not expired_events.empty else np.nan
        ),
        "p90_age_at_expiry": (
            float(expired_ages.quantile(0.9)) if not expired_events.empty else np.nan
        ),
        "max_age_at_expiry": (
            float(expired_ages.max()) if not expired_events.empty else np.nan
        ),
        "expiry_main_behavior": (
            "primarily_untouched_stale_zones_with_substantial_partial_fill_non_terminal_cleanup"
            if int((expired_mask & never_touched_mask).sum())
            >= int((expired_mask & touched_mask).sum())
            else "mostly_terminating_touched_or_partially_filled_zones"
        ),
    }

    mapped_mask = events["fvg_r_final_outcome"].isin(FINAL_OUTCOME_ORDER)
    excluded_mask = events["fvg_r_research_exclusion_reason"].notna()
    exact_one_mapping_mask = mapped_mask ^ excluded_mask

    crosstab = pd.crosstab(
        events["fvg_terminal_label"],
        events["fvg_r_final_outcome"],
        dropna=False,
    ).reindex(
        index=CORE_TERMINAL_LABEL_ORDER, columns=FINAL_OUTCOME_ORDER, fill_value=0
    )
    crosstab_dict = _crosstab_to_dict(crosstab)

    state_family_reconciliation = {}
    for label in CORE_TERMINAL_LABEL_ORDER:
        family_mask = events["fvg_terminal_label"] == label
        family_count = int(family_mask.sum())
        same_label = int((family_mask & (events["fvg_r_final_outcome"] == label)).sum())
        excluded_count = int((family_mask & excluded_mask).sum())
        missing_count = int((family_mask & ~mapped_mask & ~excluded_mask).sum())
        state_family_reconciliation[label] = {
            "core_count": family_count,
            "research_same_label_count": same_label,
            "research_reassigned_count": family_count
            - same_label
            - excluded_count
            - missing_count,
            "research_excluded_count": excluded_count,
            "research_missing_count": missing_count,
        }

    core_vs_research_audit = {
        "total_core_event_count": int(len(events)),
        "total_research_mapped_count": int(mapped_mask.sum()),
        "total_research_excluded_count": int(excluded_mask.sum()),
        "total_missing_count": int((~mapped_mask & ~excluded_mask).sum()),
        "exact_one_mapping_or_exclusion_check": bool(exact_one_mapping_mask.all()),
    }
    reconciliation_summary = {
        "each_core_event_has_exactly_one_research_mapping_or_exclusion": bool(
            exact_one_mapping_mask.all()
        ),
        "core_terminal_state_counts_sum_to_total_core_event_count": int(
            sum(core_terminal_counts.values())
        )
        == int(len(events)),
        "research_final_outcome_is_terminal_faithful": bool(
            (events["fvg_terminal_label"] == events["fvg_r_final_outcome"]).all()
        ),
    }
    breakdown_reconciliation_checks = {
        "continuation_side_counts_reconcile": _breakdown_count_total(cwt_side_breakdown)
        == eligible_never_touched_count,
        "continuation_width_counts_reconcile": _breakdown_count_total(
            cwt_width_breakdown
        )
        == eligible_never_touched_count,
        "continuation_cleanliness_counts_reconcile": _breakdown_count_total(
            cwt_cleanliness_breakdown
        )
        == eligible_never_touched_count,
        "continuation_trigger_delay_counts_reconcile": _breakdown_count_total(
            cwt_trigger_delay_breakdown
        )
        == eligible_never_touched_count,
        "continuation_trend_counts_reconcile": (
            _breakdown_count_total(cwt_trend_breakdown) == eligible_never_touched_count
            if cwt_trend_breakdown
            else True
        ),
        "cleanliness_bucket_counts_reconcile": int(
            sum(
                bucket["eligible_count"]
                for bucket in cleanliness_bucket_tradeable.values()
            )
        )
        == int(len(eligible_for_cleanliness)),
        "empty_cleanliness_buckets_not_reported_as_zero_rate": bool(
            all(
                not (
                    bucket["eligible_count"] == 0
                    and np.isfinite(bucket["rate"])
                    and float(bucket["rate"]) == 0.0
                )
                for bucket in cleanliness_bucket_tradeable.values()
            )
        ),
    }

    consistency_checks = {
        "final_outcomes_sum_to_event_count": int(sum(final_outcome_counts.values()))
        == int(len(events)),
        "each_event_has_exactly_one_final_outcome": bool(mapped_mask.all()),
        "each_core_event_has_exactly_one_research_mapping_or_exclusion": bool(
            exact_one_mapping_mask.all()
        ),
        "continuation_without_touch_implies_never_touched": bool(
            (~events.loc[current_cwt_mask, "fvg_r_ever_touched"].astype(bool)).all()
        ),
        "untouched_behavior_implies_never_touched": bool(
            (
                ~events.loc[
                    events["fvg_r_untouched_behavior_flag"].astype(bool),
                    "fvg_r_ever_touched",
                ].astype(bool)
            ).all()
        ),
        "unresolved_implies_insufficient_forward_horizon": bool(
            (
                ~events.loc[
                    unresolved_mask, "fvg_r_has_sufficient_forward_horizon"
                ].astype(bool)
            ).all()
        ),
        "expired_and_merged_counted_explicitly": bool(
            final_outcome_counts["expired"] >= 0 and final_outcome_counts["merged"] >= 0
        ),
        "core_terminal_state_counts_sum_to_total_core_event_count": reconciliation_summary[
            "core_terminal_state_counts_sum_to_total_core_event_count"
        ],
        "research_final_outcome_is_terminal_faithful": reconciliation_summary[
            "research_final_outcome_is_terminal_faithful"
        ],
        "touch_rate_reconciles_with_audits": int(never_touched_mask.sum())
        == int(len(events) - touched_audit["touched_count"]),
    }

    return {
        "event_count": int(len(events)),
        "touch_rate": float(events["fvg_first_touch_idx"].notna().mean()),
        "hold_after_touch_5_rate": _bool_rate("fvg_r_hold_after_touch_5"),
        "fail_after_touch_5_rate": _bool_rate("fvg_r_fail_after_touch_5"),
        "tradeable_touch_reaction_rate": _bool_rate("fvg_r_tradeable_touch_reaction"),
        "strong_continuation_rate": _bool_rate("fvg_r_strong_continuation"),
        "mean_touch_reaction_score": float(events["fvg_r_touch_reaction_score"].mean()),
        "mean_structural_respect_score": float(
            events["fvg_r_structural_respect_score"].mean()
        ),
        "mean_tradability_score": float(events["fvg_r_tradability_score"].mean()),
        "mean_failure_severity_score": float(
            events["fvg_r_failure_severity_score"].mean()
        ),
        "final_outcome_distribution": final_outcome_counts,
        "core_terminal_distribution": core_terminal_counts,
        "reconciliation_summary": reconciliation_summary,
        "core_vs_research_audit": core_vs_research_audit,
        "core_terminal_vs_research_final_crosstab": crosstab_dict,
        "state_family_reconciliation": state_family_reconciliation,
        "width_bucket_hold_after_touch_5": width_bucket_hold,
        "cleanliness_bucket_tradeable": cleanliness_bucket_tradeable,
        "continuation_without_touch_count_current": int(current_cwt_mask.sum()),
        "continuation_without_touch_count_recommended": int(recommended_cwt_mask.sum()),
        "never_touched_count": int(never_touched_mask.sum()),
        "proportion_of_never_touched_current": (
            float(current_cwt_mask.sum() / max(int(never_touched_mask.sum()), 1))
            if int(never_touched_mask.sum()) > 0
            else np.nan
        ),
        "proportion_of_never_touched_recommended": (
            float(recommended_cwt_mask.sum() / max(int(never_touched_mask.sum()), 1))
            if int(never_touched_mask.sum()) > 0
            else np.nan
        ),
        "continuation_without_touch_audit": continuation_without_touch_audit,
        "unresolved_count": final_outcome_counts["unresolved"],
        "unresolved_never_touched_count": int(
            (unresolved_mask & never_touched_mask).sum()
        ),
        "unresolved_touched_count": int((unresolved_mask & touched_mask).sum()),
        "unresolved_touched_only_count": int(
            (
                unresolved_mask
                & touched_mask
                & ~events["fvg_r_ever_partial_fill"].astype(bool)
            ).sum()
        ),
        "unresolved_partial_fill_count": int(
            (unresolved_mask & events["fvg_r_ever_partial_fill"].astype(bool)).sum()
        ),
        "unresolved_deep_partial_fill_count": int(
            (
                unresolved_mask
                & events["fvg_r_deep_partial_fill_before_terminal"].astype(bool)
            ).sum()
        ),
        "never_touched_audit": never_touched_audit,
        "touched_audit": touched_audit,
        "expiry_audit": expiry_audit,
        "breakdown_reconciliation_checks": breakdown_reconciliation_checks,
        "consistency_checks": consistency_checks,
        "mean_gap_cleanliness_capped": (
            float(events["fvg_gap_cleanliness_capped"].dropna().mean())
            if events["fvg_gap_cleanliness_capped"].notna().any()
            else np.nan
        ),
    }


def summarize_old_no_expiry_unresolved_audit(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {
            "reconstructable": 0,
            "unresolved_never_touched_count": 0,
            "unresolved_touched_only_count": 0,
            "unresolved_partial_fill_count": 0,
            "unresolved_deep_partial_fill_count": 0,
        }

    unresolved = events["fvg_r_final_outcome"] == "unresolved"
    touched = events["fvg_r_ever_touched"].astype(bool)
    partial = events["fvg_r_ever_partial_fill"].astype(bool)
    deep_partial = events["fvg_r_deep_partial_fill_before_terminal"].astype(bool)
    never_touched = ~touched

    return {
        "reconstructable": 1,
        "unresolved_never_touched_count": int((unresolved & never_touched).sum()),
        "unresolved_touched_only_count": int((unresolved & touched & ~partial).sum()),
        "unresolved_partial_fill_count": int((unresolved & partial).sum()),
        "unresolved_deep_partial_fill_count": int((unresolved & deep_partial).sum()),
    }
