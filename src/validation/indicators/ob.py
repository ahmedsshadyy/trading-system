"""Validation and research surface for canonical Order Blocks.

Repo stance
-----------
- BOS and CHoCH are superior to OB as primary structural signals.
- OB is validated here as an optional execution/research overlay only.
- Downstream strategies do not need OB when BOS or CHoCH already provides the
  structural decision in a simpler and stronger form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators.smc.ob import (
    OB_STATE_ACTIVE_FRESH,
    OB_STATE_ACTIVE_TOUCHED,
    OB_STATE_INVALIDATED,
    OB_STATE_MITIGATED_FULL,
    OB_STATE_MITIGATED_PARTIAL,
    OB_STATE_RETIRED,
)
from src.validation.common import write_csv_atomic, write_text_atomic

REFERENCE_START = pd.Timestamp("2026-02-01", tz="UTC")
REFERENCE_END = pd.Timestamp("2026-03-13 23:59:59", tz="UTC")
LEGACY_REFERENCE_HTML = Path("notebooks/old/detect_05_ob.html")
EXECUTION_HORIZON_BARS = 48
NEAR_BAND_ATR = 0.5
TRADABLE_NEAR_BAND_ATR = 1.0
CONTEXT_NEAR_BAND_ATR = 2.0

REQUIRED_OB_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "ob_id",
    "ob_family",
    "ob_side",
    "ob_parent_event_type",
    "ob_parent_bos_idx",
    "ob_parent_bos_ts",
    "ob_source_idx",
    "ob_source_ts",
    "ob_source_open",
    "ob_source_high",
    "ob_source_low",
    "ob_source_close",
    "ob_source_is_opposing_candle_bool",
    "ob_traceback_start_idx",
    "ob_traceback_end_idx",
    "ob_source_selection_reason",
    "ob_activate_idx",
    "ob_activate_ts",
    "ob_zone_low",
    "ob_zone_high",
    "ob_zone_mid",
    "ob_zone_height_abs",
    "ob_zone_height_atr",
    "ob_body_low",
    "ob_body_high",
    "ob_body_mid",
    "ob_parent_displacement_score",
    "ob_parent_move_away_atr",
    "ob_parent_bos_quality",
    "ob_strength",
}


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
            "p10": np.nan,
            "p90": np.nan,
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


def _fraction(mask: pd.Series | np.ndarray) -> float:
    series = pd.Series(mask)
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _event_frame(df: pd.DataFrame) -> pd.DataFrame:
    scoped = df[pd.to_numeric(df["ob_id"], errors="coerce").fillna(0) > 0].copy()
    if scoped.empty:
        return scoped
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], utc=True)
    scoped["ob_side_label"] = np.where(
        pd.to_numeric(scoped["ob_side"], errors="coerce") == 1, "bull", "bear"
    )
    return scoped.reset_index(drop=False).rename(columns={"index": "row_idx"})


def _nearest_row_idx(ts_index: pd.Series, ts_value: pd.Timestamp) -> int | None:
    if pd.isna(ts_value):
        return None
    deltas = (ts_index - ts_value).abs()
    if deltas.isna().all():
        return None
    return int(deltas.idxmin())


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _clip_unit(value: float | None, *, default: float = np.nan) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(np.clip(value, 0.0, 1.0))


def _distance_to_zone_atr(
    close_value: float,
    zone_low: float,
    zone_high: float,
    atr_value: float,
) -> float:
    if not np.isfinite(close_value) or not np.isfinite(atr_value) or atr_value <= 0:
        return np.nan
    if close_value < zone_low:
        return float((zone_low - close_value) / atr_value)
    if close_value > zone_high:
        return float((close_value - zone_high) / atr_value)
    return 0.0


def _zone_overlap_fraction(
    a_low: float, a_high: float, b_low: float, b_high: float
) -> float:
    lo = max(a_low, b_low)
    hi = min(a_high, b_high)
    if hi <= lo:
        return 0.0
    overlap = hi - lo
    denom = max(a_high - a_low, b_high - b_low, 1e-12)
    return float(overlap / denom)


def _row_offset(events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    deltas = (
        pd.to_numeric(events["ob_activate_idx"], errors="coerce")
        - pd.to_numeric(events["row_idx"], errors="coerce")
    ).dropna()
    if deltas.empty:
        return 0
    return int(deltas.median())


def _local_pos(value: float | int | None, *, offset: int) -> int | None:
    if value is None or not np.isfinite(value):
        return None
    return int(value) - offset


def summarize_ob(df: pd.DataFrame) -> dict[str, object]:
    missing = REQUIRED_OB_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing OB columns: {sorted(missing)}")

    events = _event_frame(df)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    confirmed_bos_mask = (
        pd.to_numeric(df.get("bos_bull", 0), errors="coerce").fillna(0).astype(int) == 1
    ) | (
        pd.to_numeric(df.get("bos_bear", 0), errors="coerce").fillna(0).astype(int) == 1
    )
    confirmed_bos_count = int(confirmed_bos_mask.sum())
    if events.empty:
        return {
            "window": {
                "start": str(ts.min()),
                "end": str(ts.max()),
                "rows": int(len(df)),
            },
            "event_counts": {
                "confirmed_bos_count": confirmed_bos_count,
                "ob_count": 0,
                "raw_canonical_ob_count": 0,
                "qualified_canonical_ob_count": 0,
            },
            "coverage": {
                "coverage_fraction": 0.0 if confirmed_bos_count > 0 else np.nan,
                "pathological_exclusion_count": confirmed_bos_count,
            },
            "sanity_checks": {
                "has_events": False,
                "every_ob_has_parent_bos": True,
                "source_selection_consistency_fraction": np.nan,
                "activation_equality_fraction": np.nan,
                "activation_equals_parent_confirmation": True,
                "geometry_positive": True,
                "geometry_full_range_consistency": True,
                "source_before_or_at_activation": True,
                "bull_source_candle_is_bearish": True,
                "bear_source_candle_is_bullish": True,
                "one_raw_ob_per_confirmed_bos_or_pathological": True,
            },
        }

    source_is_bearish = events["ob_source_close"] < events["ob_source_open"]
    source_is_bullish = events["ob_source_close"] > events["ob_source_open"]
    bull_mask = pd.to_numeric(events["ob_side"], errors="coerce") == 1
    bear_mask = pd.to_numeric(events["ob_side"], errors="coerce") == -1
    activation_latency = pd.to_numeric(
        events["ob_activate_idx"], errors="coerce"
    ) - pd.to_numeric(events["ob_source_idx"], errors="coerce")
    coverage_fraction = (
        float(len(events) / confirmed_bos_count) if confirmed_bos_count > 0 else np.nan
    )

    summary: dict[str, object] = {
        "window": {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "rows": int(len(df)),
        },
        "event_counts": {
            "confirmed_bos_count": confirmed_bos_count,
            "ob_count": int(len(events)),
            "raw_canonical_ob_count": int(len(events)),
            "qualified_canonical_ob_count": int(len(events)),
            "bull_count": int(bull_mask.sum()),
            "bear_count": int(bear_mask.sum()),
            "active_count": int(
                pd.to_numeric(events.get("ob_is_active", 0), errors="coerce")
                .fillna(0)
                .sum()
            ),
            "fresh_count": int(
                pd.to_numeric(events.get("ob_is_fresh", 0), errors="coerce")
                .fillna(0)
                .sum()
            ),
        },
        "coverage": {
            "coverage_fraction": coverage_fraction,
            "pathological_exclusion_count": max(
                confirmed_bos_count - int(len(events)), 0
            ),
        },
        "distributions": {
            "zone_height_atr": _continuous_stats(events["ob_zone_height_atr"]),
            "activation_latency_bars": _continuous_stats(activation_latency),
            "parent_displacement_score": _continuous_stats(
                events["ob_parent_displacement_score"]
            ),
            "parent_move_away_atr": _continuous_stats(
                events["ob_parent_move_away_atr"]
            ),
            "strength": _continuous_stats(events["ob_strength"]),
        },
        "sanity_checks": {
            "every_ob_has_parent_bos": bool(events["ob_parent_bos_idx"].notna().all()),
            "source_selection_consistency_fraction": float(
                (
                    pd.to_numeric(
                        events["ob_source_is_opposing_candle_bool"], errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                    == 1
                ).mean()
            ),
            "activation_equality_fraction": float(
                (
                    pd.to_numeric(events["ob_activate_idx"], errors="coerce")
                    == pd.to_numeric(events["ob_parent_bos_idx"], errors="coerce")
                ).mean()
            ),
            "activation_equals_parent_confirmation": bool(
                (
                    pd.to_numeric(events["ob_activate_idx"], errors="coerce")
                    == pd.to_numeric(events["ob_parent_bos_idx"], errors="coerce")
                ).all()
            ),
            "geometry_positive": bool((events["ob_zone_height_abs"] > 0).all()),
            "geometry_full_range_consistency": bool(
                (
                    np.isclose(
                        events["ob_zone_low"], events["ob_source_low"], equal_nan=False
                    )
                    & np.isclose(
                        events["ob_zone_high"],
                        events["ob_source_high"],
                        equal_nan=False,
                    )
                ).all()
            ),
            "source_before_or_at_activation": bool((activation_latency >= 0).all()),
            "bull_source_candle_is_bearish": bool(
                (~bull_mask | source_is_bearish).all()
            ),
            "bear_source_candle_is_bullish": bool(
                (~bear_mask | source_is_bullish).all()
            ),
            "one_raw_ob_per_confirmed_bos_or_pathological": bool(
                int(len(events)) <= confirmed_bos_count
            ),
        },
    }

    if "ob_first_touch_idx" in events.columns:
        summary["mitigation_checks"] = {
            "no_touch_before_activation": bool(
                (
                    events["ob_first_touch_idx"].isna()
                    | (
                        pd.to_numeric(events["ob_first_touch_idx"], errors="coerce")
                        >= pd.to_numeric(events["ob_activate_idx"], errors="coerce")
                    )
                ).all()
            ),
            "full_not_before_touch": bool(
                (
                    events["ob_first_full_mitigation_idx"].isna()
                    | events["ob_first_touch_idx"].isna()
                    | (
                        pd.to_numeric(
                            events["ob_first_full_mitigation_idx"], errors="coerce"
                        )
                        >= pd.to_numeric(events["ob_first_touch_idx"], errors="coerce")
                    )
                ).all()
            ),
        }
    return summary


def ob_event_windows(
    df: pd.DataFrame,
    *,
    side: str = "all",
    limit: int = 5,
    bars_before: int = 6,
    bars_after: int = 14,
) -> list[pd.DataFrame]:
    events = _event_frame(df)
    if events.empty:
        return []
    if side == "bull":
        events = events[events["ob_side"] == 1]
    elif side == "bear":
        events = events[events["ob_side"] == -1]
    events = events.sort_values(["ob_activate_idx", "ob_id"], kind="stable").head(limit)
    offset = _row_offset(events)
    windows: list[pd.DataFrame] = []
    for _, event in events.iterrows():
        center = _local_pos(event["ob_activate_idx"], offset=offset)
        if center is None:
            continue
        start = max(center - bars_before, 0)
        end = min(center + bars_after + 1, len(df))
        cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "bos_bull",
            "bos_bear",
            "ob_id",
            "ob_family",
            "ob_side",
            "ob_source_idx",
            "ob_parent_bos_idx",
            "ob_activate_idx",
            "ob_zone_low",
            "ob_zone_high",
            "ob_strength",
            "ob_state",
            "ob_first_touch_idx",
            "ob_first_full_mitigation_idx",
            "ob_invalidation_idx",
        ]
        avail = [col for col in cols if col in df.columns]
        window = df.iloc[start:end][avail].copy()
        window["event_row"] = 0
        local_row = center - start
        if 0 <= local_row < len(window):
            window.iloc[local_row, window.columns.get_loc("event_row")] = 1
        windows.append(window.reset_index(drop=True))
    return windows


def plot_ob_validation(df: pd.DataFrame, *, outpath: str | Path, title: str) -> Path:
    scoped = df.copy()
    scoped["timestamp"] = pd.to_datetime(scoped["timestamp"], utc=True)
    events = _event_frame(scoped)
    offset = _row_offset(events)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=scoped["timestamp"],
            open=scoped["open"],
            high=scoped["high"],
            low=scoped["low"],
            close=scoped["close"],
            name="OHLC",
        )
    )
    if not events.empty:
        top_events = events.sort_values(
            ["ob_activate_idx", "ob_id"], kind="stable"
        ).tail(25)
        for _, event in top_events.iterrows():
            activate_idx = _local_pos(event["ob_activate_idx"], offset=offset)
            source_idx = _local_pos(event["ob_source_idx"], offset=offset)
            if activate_idx is None or activate_idx >= len(scoped):
                continue
            x0 = scoped["timestamp"].iloc[max(activate_idx - 1, 0)]
            x1_idx = (
                int(
                    pd.to_numeric(
                        event.get("ob_invalidation_idx", np.nan), errors="coerce"
                    )
                )
                if pd.notna(event.get("ob_invalidation_idx", np.nan))
                else min(activate_idx + 12, len(scoped) - 1)
            )
            x1 = scoped["timestamp"].iloc[
                min(max(x1_idx - offset, activate_idx), len(scoped) - 1)
            ]
            fig.add_hrect(
                y0=float(event["ob_zone_low"]),
                y1=float(event["ob_zone_high"]),
                x0=x0,
                x1=x1,
                fillcolor=(
                    "rgba(21,128,61,0.10)"
                    if int(event["ob_side"]) == 1
                    else "rgba(185,28,28,0.10)"
                ),
                line_color=(
                    "rgba(21,128,61,0.55)"
                    if int(event["ob_side"]) == 1
                    else "rgba(185,28,28,0.55)"
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[scoped["timestamp"].iloc[activate_idx]],
                    y=[scoped["close"].iloc[activate_idx]],
                    mode="markers",
                    marker=dict(
                        color="#15803d" if int(event["ob_side"]) == 1 else "#b91c1c",
                        symbol=(
                            "triangle-up"
                            if int(event["ob_side"]) == 1
                            else "triangle-down"
                        ),
                        size=10,
                    ),
                    name=f"OB #{int(event['ob_id'])}",
                    text=[
                        f"OB #{int(event['ob_id'])} source={int(event['ob_source_idx'])}"
                    ],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            if source_idx is not None and 0 <= source_idx < len(scoped):
                fig.add_trace(
                    go.Scatter(
                        x=[scoped["timestamp"].iloc[source_idx]],
                        y=[float(event["ob_zone_mid"])],
                        mode="markers",
                        marker=dict(color="#1d4ed8", size=8, symbol="circle"),
                        name="Source",
                        showlegend=False,
                    )
                )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        xaxis_rangeslider_visible=False,
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def validate_ob(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str,
    n_windows: int = 5,
) -> dict[str, Any]:
    diagnostics = build_ob_diagnostic_package(df, instrument="XAU_USD", timeframe="H4")
    html_path = plot_ob_validation(df, outpath=outpath, title=title)
    return {
        "summary": summarize_ob(df),
        "bull_windows": ob_event_windows(df, side="bull", limit=n_windows),
        "bear_windows": ob_event_windows(df, side="bear", limit=n_windows),
        "diagnostics": diagnostics,
        "html_path": str(html_path),
    }


@dataclass
class _LifecycleEvent:
    ob_id: int
    ob_side: int
    activate_idx: int
    source_idx: int
    zone_low: float
    zone_high: float
    zone_mid: float
    zone_height_atr: float
    body_low: float
    body_high: float
    body_mid: float
    strength: float
    parent_quality: float
    displacement_score: float
    family: str
    state: str
    first_touch_idx: int | None
    first_partial_idx: int | None
    first_full_idx: int | None
    invalidation_idx: int | None
    retire_idx: int | None


def _state_label(value: int) -> str:
    return {
        OB_STATE_ACTIVE_FRESH: "fresh",
        OB_STATE_ACTIVE_TOUCHED: "touched",
        OB_STATE_MITIGATED_PARTIAL: "partial",
        OB_STATE_MITIGATED_FULL: "full",
        OB_STATE_INVALIDATED: "invalidated",
        OB_STATE_RETIRED: "retired",
    }.get(int(value), "inactive")


def _reconstruct_lifecycle(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[_LifecycleEvent]]:
    events = _event_frame(df)
    if events.empty:
        return pd.DataFrame(), []
    rows: list[dict[str, Any]] = []
    lifecycle: list[_LifecycleEvent] = []
    for _, row in events.iterrows():
        activate_idx = int(pd.to_numeric(row["ob_activate_idx"], errors="coerce"))
        source_idx = int(pd.to_numeric(row["ob_source_idx"], errors="coerce"))
        first_touch_idx = (
            int(pd.to_numeric(row.get("ob_first_touch_idx"), errors="coerce"))
            if pd.notna(row.get("ob_first_touch_idx"))
            else None
        )
        first_partial_idx = (
            int(
                pd.to_numeric(
                    row.get(
                        "ob_first_partial_idx",
                        row.get("ob_first_partial_mitigation_idx"),
                    ),
                    errors="coerce",
                )
            )
            if pd.notna(
                row.get(
                    "ob_first_partial_idx", row.get("ob_first_partial_mitigation_idx")
                )
            )
            else None
        )
        first_full_idx = (
            int(pd.to_numeric(row.get("ob_first_full_mitigation_idx"), errors="coerce"))
            if pd.notna(row.get("ob_first_full_mitigation_idx"))
            else None
        )
        invalidation_idx = (
            int(pd.to_numeric(row.get("ob_invalidation_idx"), errors="coerce"))
            if pd.notna(row.get("ob_invalidation_idx"))
            else None
        )
        retire_idx = (
            int(pd.to_numeric(row.get("ob_retire_idx"), errors="coerce"))
            if pd.notna(row.get("ob_retire_idx"))
            else None
        )
        terminal_candidates = [
            value
            for value in [first_full_idx, invalidation_idx, retire_idx]
            if value is not None
        ]
        terminal_idx = min(terminal_candidates) if terminal_candidates else len(df)
        active_end_exclusive = terminal_idx if terminal_idx is not None else len(df)
        fresh_duration = (
            (first_touch_idx - activate_idx)
            if first_touch_idx is not None
            else (active_end_exclusive - activate_idx)
        )
        touched_duration = 0
        partial_duration = 0
        if (
            first_touch_idx is not None
            and first_partial_idx is not None
            and first_partial_idx > first_touch_idx
        ):
            touched_duration = first_partial_idx - first_touch_idx
        if first_partial_idx is not None:
            partial_end = min(
                [
                    value
                    for value in [first_full_idx, invalidation_idx, retire_idx, len(df)]
                    if value is not None
                ]
            )
            partial_duration = max(partial_end - first_partial_idx, 0)
        total_live = max(active_end_exclusive - activate_idx, 0)
        rows.append(
            {
                "ob_id": int(row["ob_id"]),
                "family": row["ob_family"],
                "side_label": row["ob_side_label"],
                "activate_idx": activate_idx,
                "source_idx": source_idx,
                "first_touch_idx": (
                    float(first_touch_idx) if first_touch_idx is not None else np.nan
                ),
                "first_partial_idx": (
                    float(first_partial_idx)
                    if first_partial_idx is not None
                    else np.nan
                ),
                "first_full_mitigation_idx": (
                    float(first_full_idx) if first_full_idx is not None else np.nan
                ),
                "invalidation_idx": (
                    float(invalidation_idx) if invalidation_idx is not None else np.nan
                ),
                "retire_idx": float(retire_idx) if retire_idx is not None else np.nan,
                "fresh_state_duration": float(fresh_duration),
                "touched_state_duration": float(touched_duration),
                "partial_state_duration": float(partial_duration),
                "post_full_pre_terminal_duration": 0.0,
                "total_live_duration": float(total_live),
                "bars_to_first_touch": (
                    float(first_touch_idx - activate_idx)
                    if first_touch_idx is not None
                    else np.nan
                ),
                "bars_to_first_partial_mitigation": (
                    float(first_partial_idx - activate_idx)
                    if first_partial_idx is not None
                    else np.nan
                ),
                "bars_to_first_full_mitigation": (
                    float(first_full_idx - activate_idx)
                    if first_full_idx is not None
                    else np.nan
                ),
                "bars_to_invalidation": (
                    float(invalidation_idx - activate_idx)
                    if invalidation_idx is not None
                    else np.nan
                ),
                "bars_to_retirement": (
                    float(retire_idx - activate_idx)
                    if retire_idx is not None
                    else np.nan
                ),
                "fresh_life_bars": (
                    float(first_touch_idx - activate_idx)
                    if first_touch_idx is not None
                    else float(total_live)
                ),
                "terminal_state": _state_label(
                    int(pd.to_numeric(row["ob_state"], errors="coerce"))
                ),
            }
        )
        lifecycle.append(
            _LifecycleEvent(
                ob_id=int(row["ob_id"]),
                ob_side=int(pd.to_numeric(row["ob_side"], errors="coerce")),
                activate_idx=activate_idx,
                source_idx=source_idx,
                zone_low=float(row["ob_zone_low"]),
                zone_high=float(row["ob_zone_high"]),
                zone_mid=float(row["ob_zone_mid"]),
                zone_height_atr=float(row["ob_zone_height_atr"]),
                body_low=float(row["ob_body_low"]),
                body_high=float(row["ob_body_high"]),
                body_mid=float(row["ob_body_mid"]),
                strength=float(row["ob_strength"]),
                parent_quality=float(row.get("ob_parent_bos_quality", np.nan)),
                displacement_score=float(
                    row.get("ob_parent_displacement_score", np.nan)
                ),
                family=str(row["ob_family"]),
                state=_state_label(
                    int(pd.to_numeric(row["ob_state"], errors="coerce"))
                ),
                first_touch_idx=first_touch_idx,
                first_partial_idx=first_partial_idx,
                first_full_idx=first_full_idx,
                invalidation_idx=invalidation_idx,
                retire_idx=retire_idx,
            )
        )
    return pd.DataFrame(rows), lifecycle


def _monitorability_score(
    *,
    distance_atr: float,
    is_fresh: bool,
    age_since_activation: int,
    age_since_first_touch: int | None,
    parent_quality: float,
    displacement_score: float,
    zone_height_atr: float,
    trend_state: float,
    side: int,
    family: str,
    price: float,
    zone_low: float,
    zone_high: float,
) -> float:
    distance_component = _clip_unit(
        1.0 / (1.0 + max(distance_atr, 0.0)), default=np.nan
    )
    freshness_component = 1.0 if is_fresh else 0.55
    age_component = _clip_unit(
        np.exp(-max(age_since_activation, 0) / 18.0), default=np.nan
    )
    touch_age_component = (
        1.0
        if age_since_first_touch is None
        else _clip_unit(np.exp(-max(age_since_first_touch, 0) / 12.0), default=np.nan)
    )
    strength_component = _clip_unit(
        np.nanmean([parent_quality, displacement_score]), default=np.nan
    )
    size_penalty_component = _clip_unit(
        1.0 - _safe_fraction(zone_height_atr, 4.0), default=np.nan
    )
    trend_component = (
        1.0 if np.isfinite(trend_state) and int(trend_state) == side else 0.65
    )
    family_component = 1.0 if family == "bos" else 0.9
    side_relevance_component = 1.0
    if side == 1 and price < zone_low:
        side_relevance_component = 0.55
    if side == -1 and price > zone_high:
        side_relevance_component = 0.55
    return _clip_unit(
        0.30 * distance_component
        + 0.18 * freshness_component
        + 0.14 * age_component
        + 0.08 * touch_age_component
        + 0.12 * strength_component
        + 0.08 * size_penalty_component
        + 0.06 * trend_component
        + 0.02 * family_component
        + 0.02 * side_relevance_component,
        default=np.nan,
    )


def _build_inventory_surfaces(
    df: pd.DataFrame, lifecycle: list[_LifecycleEvent]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close_arr = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr_arr = pd.to_numeric(df.get("atr_14", np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    trend_arr = pd.to_numeric(df.get("trend_state", np.nan), errors="coerce").to_numpy(
        dtype=float
    )

    detail_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for row_idx in range(len(df)):
        active_entries: list[dict[str, Any]] = []
        for event in lifecycle:
            terminal_candidates = [
                value
                for value in [
                    event.first_full_idx,
                    event.invalidation_idx,
                    event.retire_idx,
                ]
                if value is not None
            ]
            active_end_exclusive = (
                min(terminal_candidates) if terminal_candidates else len(df)
            )
            if row_idx < event.activate_idx or row_idx >= active_end_exclusive:
                continue
            close_value = close_arr[row_idx]
            atr_value = atr_arr[row_idx]
            distance_atr = _distance_to_zone_atr(
                close_value, event.zone_low, event.zone_high, atr_value
            )
            strict_fresh = int(
                event.first_touch_idx is None or row_idx < event.first_touch_idx
            )
            display_fresh = int(
                strict_fresh == 1
                and np.isfinite(distance_atr)
                and distance_atr <= CONTEXT_NEAR_BAND_ATR
            )
            age_since_first_touch = None
            if event.first_touch_idx is not None and row_idx >= event.first_touch_idx:
                age_since_first_touch = row_idx - event.first_touch_idx
            score = _monitorability_score(
                distance_atr=distance_atr,
                is_fresh=bool(strict_fresh),
                age_since_activation=row_idx - event.activate_idx,
                age_since_first_touch=age_since_first_touch,
                parent_quality=event.parent_quality,
                displacement_score=event.displacement_score,
                zone_height_atr=event.zone_height_atr,
                trend_state=trend_arr[row_idx],
                side=event.ob_side,
                family=event.family,
                price=close_value,
                zone_low=event.zone_low,
                zone_high=event.zone_high,
            )
            entry = {
                "row_idx": row_idx,
                "timestamp": ts.iloc[row_idx],
                "ob_id": event.ob_id,
                "ob_family": event.family,
                "ob_side": event.ob_side,
                "ob_distance_to_price_atr": distance_atr,
                "ob_monitorability_score": score,
                "ob_rank_among_active_same_side": np.nan,
                "ob_rank_among_active_all": np.nan,
                "ob_is_fresh_strict": strict_fresh,
                "ob_is_fresh_display": display_fresh,
                "ob_state_primary": "fresh" if strict_fresh == 1 else "active",
                "ob_state_detail": event.state if strict_fresh == 0 else "fresh",
                "ob_first_touch_idx": (
                    float(event.first_touch_idx)
                    if event.first_touch_idx is not None
                    else np.nan
                ),
                "ob_first_partial_idx": (
                    float(event.first_partial_idx)
                    if event.first_partial_idx is not None
                    else np.nan
                ),
                "ob_first_full_mitigation_idx": (
                    float(event.first_full_idx)
                    if event.first_full_idx is not None
                    else np.nan
                ),
                "ob_invalidation_idx": (
                    float(event.invalidation_idx)
                    if event.invalidation_idx is not None
                    else np.nan
                ),
                "ob_retire_idx": (
                    float(event.retire_idx) if event.retire_idx is not None else np.nan
                ),
            }
            active_entries.append(entry)

        active_entries.sort(
            key=lambda item: (
                (
                    float(item["ob_monitorability_score"])
                    if np.isfinite(item["ob_monitorability_score"])
                    else -1.0
                ),
                -(
                    float(item["ob_distance_to_price_atr"])
                    if np.isfinite(item["ob_distance_to_price_atr"])
                    else 999.0
                ),
            ),
            reverse=True,
        )
        bulls = [entry for entry in active_entries if int(entry["ob_side"]) == 1]
        bears = [entry for entry in active_entries if int(entry["ob_side"]) == -1]
        strict_fresh_entries = [
            entry for entry in active_entries if int(entry["ob_is_fresh_strict"]) == 1
        ]
        display_fresh_entries = [
            entry for entry in active_entries if int(entry["ob_is_fresh_display"]) == 1
        ]

        for side_entries in (bulls, bears):
            for rank, entry in enumerate(side_entries, start=1):
                entry["ob_rank_among_active_same_side"] = rank
        for rank, entry in enumerate(active_entries, start=1):
            entry["ob_rank_among_active_all"] = rank
            detail_rows.append(entry)

        def _top_id(entries: list[dict[str, Any]]) -> float:
            return float(entries[0]["ob_id"]) if entries else np.nan

        def _top_score(entries: list[dict[str, Any]]) -> float:
            return float(entries[0]["ob_monitorability_score"]) if entries else np.nan

        def _top_distance(entries: list[dict[str, Any]]) -> float:
            return float(entries[0]["ob_distance_to_price_atr"]) if entries else np.nan

        top_bull_id = _top_id(bulls)
        top_bear_id = _top_id(bears)
        top_all_id = _top_id(active_entries)
        raw_active_count = len(active_entries)
        raw_fresh_count = len(strict_fresh_entries)
        display_fresh_count = len(display_fresh_entries)
        top_active_distance = _top_distance(active_entries)
        top_fresh_distance = _top_distance(display_fresh_entries)
        best_bull_score = _top_score(bulls)
        best_bear_score = _top_score(bears)
        endpoint_warning_flags: list[str] = []
        if raw_active_count == 0:
            endpoint_warning_flags.append("no_active_inventory")
        if raw_fresh_count == 0:
            endpoint_warning_flags.append("no_fresh_inventory")
        if (
            np.isfinite(top_active_distance)
            and top_active_distance > CONTEXT_NEAR_BAND_ATR
        ):
            endpoint_warning_flags.append("top_inventory_too_far")
        if raw_active_count > 0 and (
            not np.isfinite(_top_score(active_entries))
            or _top_score(active_entries) < 0.40
        ):
            endpoint_warning_flags.append(
                "raw_active_inventory_exists_but_low_monitorability"
            )
        if raw_active_count >= 3:
            endpoint_warning_flags.append("active_inventory_cluttered")
        if bulls and bears:
            endpoint_warning_flags.append("side_mix_conflicted")
        if raw_fresh_count != display_fresh_count:
            endpoint_warning_flags.append("inventory_semantics_ambiguous")

        inventory_rows.append(
            {
                "row_idx": row_idx,
                "timestamp": ts.iloc[row_idx],
                "raw_active_count": raw_active_count,
                "raw_fresh_count": raw_fresh_count,
                "display_fresh_count": display_fresh_count,
                "top_bull_ob_id": top_bull_id,
                "top_bear_ob_id": top_bear_id,
                "top_all_ob_id": top_all_id,
                "best_bull_ob_score": best_bull_score,
                "best_bear_ob_score": best_bear_score,
                "top_active_distance_atr": top_active_distance,
                "top_fresh_distance_atr": top_fresh_distance,
                "distance_to_nearest_active_ob_atr": _top_distance(
                    sorted(
                        active_entries,
                        key=lambda item: (
                            float(item["ob_distance_to_price_atr"])
                            if np.isfinite(item["ob_distance_to_price_atr"])
                            else 999.0
                        ),
                    )
                ),
                "distance_to_nearest_fresh_ob_atr": _top_distance(
                    sorted(
                        strict_fresh_entries,
                        key=lambda item: (
                            float(item["ob_distance_to_price_atr"])
                            if np.isfinite(item["ob_distance_to_price_atr"])
                            else 999.0
                        ),
                    )
                ),
                "fractional_distance_band_active": (
                    "near"
                    if np.isfinite(top_active_distance)
                    and top_active_distance <= NEAR_BAND_ATR
                    else (
                        "tradable_near"
                        if np.isfinite(top_active_distance)
                        and top_active_distance <= TRADABLE_NEAR_BAND_ATR
                        else (
                            "context_near"
                            if np.isfinite(top_active_distance)
                            and top_active_distance <= CONTEXT_NEAR_BAND_ATR
                            else "distant"
                        )
                    )
                ),
                "endpoint_inventory_is_good_bool": int(
                    raw_active_count > 0
                    and np.isfinite(top_active_distance)
                    and top_active_distance <= TRADABLE_NEAR_BAND_ATR
                    and np.isfinite(_top_score(active_entries))
                    and _top_score(active_entries) >= 0.45
                ),
                "endpoint_inventory_warning_flags": ",".join(endpoint_warning_flags),
                "fraction_with_active_within_0_5atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"]) <= NEAR_BAND_ATR
                        for entry in active_entries
                    )
                ),
                "fraction_with_active_within_1_0atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"])
                        <= TRADABLE_NEAR_BAND_ATR
                        for entry in active_entries
                    )
                ),
                "fraction_with_active_within_2_0atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"])
                        <= CONTEXT_NEAR_BAND_ATR
                        for entry in active_entries
                    )
                ),
                "fraction_with_fresh_within_0_5atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"]) <= NEAR_BAND_ATR
                        for entry in strict_fresh_entries
                    )
                ),
                "fraction_with_fresh_within_1_0atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"])
                        <= TRADABLE_NEAR_BAND_ATR
                        for entry in strict_fresh_entries
                    )
                ),
                "fraction_with_fresh_within_2_0atr": int(
                    any(
                        np.isfinite(entry["ob_distance_to_price_atr"])
                        and float(entry["ob_distance_to_price_atr"])
                        <= CONTEXT_NEAR_BAND_ATR
                        for entry in strict_fresh_entries
                    )
                ),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    distance_summary = {
        "strict_fresh_definition": "untouched_since_activation",
        "display_fresh_definition": "strict_fresh_and_within_2atr",
        "fraction_with_top_inventory_within_1atr": float(
            (
                pd.to_numeric(inventory_df["top_active_distance_atr"], errors="coerce")
                <= TRADABLE_NEAR_BAND_ATR
            )
            .fillna(False)
            .mean()
        ),
        "fraction_with_top_inventory_within_2atr": float(
            (
                pd.to_numeric(inventory_df["top_active_distance_atr"], errors="coerce")
                <= CONTEXT_NEAR_BAND_ATR
            )
            .fillna(False)
            .mean()
        ),
        "fraction_bars_with_active_within_0_5_atr": float(
            (inventory_df["fraction_with_active_within_0_5atr"] == 1).mean()
        ),
        "fraction_bars_with_active_within_1_0_atr": float(
            (inventory_df["fraction_with_active_within_1_0atr"] == 1).mean()
        ),
        "fraction_bars_with_active_within_2_0_atr": float(
            (inventory_df["fraction_with_active_within_2_0atr"] == 1).mean()
        ),
        "fraction_bars_with_fresh_within_0_5_atr": float(
            (inventory_df["fraction_with_fresh_within_0_5atr"] == 1).mean()
        ),
        "fraction_bars_with_fresh_within_1_0_atr": float(
            (inventory_df["fraction_with_fresh_within_1_0atr"] == 1).mean()
        ),
        "fraction_bars_with_fresh_within_2_0_atr": float(
            (inventory_df["fraction_with_fresh_within_2_0atr"] == 1).mean()
        ),
        "median_distance_to_top_inventory_atr": float(
            pd.to_numeric(
                inventory_df["top_active_distance_atr"], errors="coerce"
            ).median()
        ),
        "median_distance_to_top_ranked_fresh_ob_atr": float(
            pd.to_numeric(
                inventory_df["top_fresh_distance_atr"], errors="coerce"
            ).median()
        ),
        "endpoint_raw_fresh_count_strict": int(
            pd.to_numeric(inventory_df["raw_fresh_count"], errors="coerce")
            .fillna(0)
            .iloc[-1]
        ),
        "endpoint_display_fresh_count": int(
            pd.to_numeric(inventory_df["display_fresh_count"], errors="coerce")
            .fillna(0)
            .iloc[-1]
        ),
    }
    return detail_df, inventory_df, distance_summary


def _build_bos_coverage_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    events = _event_frame(df)
    bos_rows = []
    bos_bull = (
        pd.to_numeric(df.get("bos_bull", 0), errors="coerce").fillna(0).astype(int)
    )
    bos_bear = (
        pd.to_numeric(df.get("bos_bear", 0), errors="coerce").fillna(0).astype(int)
    )
    event_parent_idx = (
        set(
            pd.to_numeric(events["ob_parent_bos_idx"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )
        if not events.empty
        else set()
    )
    for idx in range(len(df)):
        side = (
            1
            if int(bos_bull.iloc[idx]) == 1
            else -1 if int(bos_bear.iloc[idx]) == 1 else 0
        )
        if side == 0:
            continue
        covered = int(idx in event_parent_idx)
        bos_rows.append(
            {
                "bos_idx": idx,
                "bos_ts": ts.iloc[idx],
                "side_label": "bull" if side == 1 else "bear",
                "covered_by_raw_canonical_ob": covered,
                "reject_reason": (
                    "" if covered == 1 else "pathological_no_valid_source_or_geometry"
                ),
                "calendar_year": str(ts.iloc[idx].year),
            }
        )
    detail = pd.DataFrame(bos_rows)
    if detail.empty:
        agg = pd.DataFrame(
            columns=[
                "calendar_year",
                "side_label",
                "confirmed_bos_count",
                "raw_canonical_bos_ob_count",
                "qualified_canonical_ob_count",
                "coverage_fraction",
                "pathological_exclusion_count",
            ]
        )
    else:
        parts = []
        for year in ["ALL", *sorted(detail["calendar_year"].unique())]:
            scoped_year = (
                detail if year == "ALL" else detail[detail["calendar_year"] == year]
            )
            for side in ["all", "bull", "bear"]:
                scoped = (
                    scoped_year
                    if side == "all"
                    else scoped_year[scoped_year["side_label"] == side]
                )
                if scoped.empty:
                    continue
                confirmed = int(len(scoped))
                covered = int(scoped["covered_by_raw_canonical_ob"].sum())
                parts.append(
                    {
                        "calendar_year": year,
                        "side_label": side,
                        "confirmed_bos_count": confirmed,
                        "raw_canonical_bos_ob_count": covered,
                        "qualified_canonical_ob_count": covered,
                        "coverage_fraction": (
                            float(covered / confirmed) if confirmed > 0 else np.nan
                        ),
                        "pathological_exclusion_count": max(confirmed - covered, 0),
                    }
                )
        agg = pd.DataFrame(parts)
    summary_row = agg[(agg["calendar_year"] == "ALL") & (agg["side_label"] == "all")]
    summary = (
        summary_row.iloc[0].to_dict()
        if not summary_row.empty
        else {
            "confirmed_bos_count": 0,
            "raw_canonical_bos_ob_count": 0,
            "qualified_canonical_ob_count": 0,
            "coverage_fraction": np.nan,
            "pathological_exclusion_count": 0,
        }
    )
    return agg, summary


def _parse_legacy_nonlive_reference(
    df: pd.DataFrame,
    *,
    html_path: Path = LEGACY_REFERENCE_HTML,
    start: pd.Timestamp = REFERENCE_START,
    end: pd.Timestamp = REFERENCE_END,
) -> pd.DataFrame:
    path = Path(html_path)
    if not path.exists():
        return pd.DataFrame()
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'\{"fillcolor":"(?P<fill>rgba\([^"]+\))","line":\{.*?\},"type":"rect",'
        r'"x0":"(?P<x0>[^"]+)","x1":"(?P<x1>[^"]+)","y0":(?P<y0>-?\d+(?:\.\d+)?),'
        r'"y1":(?P<y1>-?\d+(?:\.\d+)?)\}'
    )
    ts_index = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    next_id = 1
    for match in pattern.finditer(text):
        fill = match.group("fill")
        side = 1 if "0,100,255" in fill else -1 if "255,100,0" in fill else 0
        if side == 0:
            continue
        x0 = pd.to_datetime(match.group("x0"), utc=True, errors="coerce")
        if pd.isna(x0) or x0 < start or x0 > end:
            continue
        row_idx = _nearest_row_idx(ts_index, x0)
        rows.append(
            {
                "reference_case_id": next_id,
                "reference_side": "bull" if side == 1 else "bear",
                "reference_side_int": side,
                "reference_source_ts": x0,
                "reference_source_idx": (
                    float(row_idx) if row_idx is not None else np.nan
                ),
                "reference_zone_low": float(match.group("y0")),
                "reference_zone_high": float(match.group("y1")),
            }
        )
        next_id += 1
    return pd.DataFrame(rows)


def _build_equivalence_casebook(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = _parse_legacy_nonlive_reference(df)
    events = _event_frame(df)
    if reference.empty:
        return pd.DataFrame(), {
            "reference_nonlive_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "matched_fraction": np.nan,
            "median_source_lag_bars": np.nan,
            "median_activation_lag_bars": np.nan,
            "median_geometry_drift_atr": np.nan,
        }
    rows: list[dict[str, Any]] = []
    for ref in reference.to_dict("records"):
        candidates = (
            events[events["ob_side"] == ref["reference_side_int"]].copy()
            if not events.empty
            else pd.DataFrame()
        )
        if not candidates.empty and np.isfinite(ref["reference_source_idx"]):
            candidates["source_lag_abs"] = (
                pd.to_numeric(candidates["ob_source_idx"], errors="coerce")
                - float(ref["reference_source_idx"])
            ).abs()
            candidates["activation_lag_abs"] = (
                pd.to_numeric(candidates["ob_activate_idx"], errors="coerce")
                - float(ref["reference_source_idx"])
            ).abs()
            candidates["overlap_fraction"] = candidates.apply(
                lambda row: _zone_overlap_fraction(
                    float(ref["reference_zone_low"]),
                    float(ref["reference_zone_high"]),
                    float(row["ob_zone_low"]),
                    float(row["ob_zone_high"]),
                ),
                axis=1,
            )
            candidates["geometry_drift_atr"] = (
                (
                    pd.to_numeric(candidates["ob_zone_low"], errors="coerce")
                    - float(ref["reference_zone_low"])
                ).abs()
                + (
                    pd.to_numeric(candidates["ob_zone_high"], errors="coerce")
                    - float(ref["reference_zone_high"])
                ).abs()
            ) / (
                2.0 * pd.to_numeric(df.get("atr_14", np.nan), errors="coerce").median()
            )
            candidates["match_score"] = (
                candidates["overlap_fraction"].fillna(0.0) * 3.0
                - candidates["source_lag_abs"].fillna(999.0) * 0.08
                - candidates["activation_lag_abs"].fillna(999.0) * 0.04
            )
            candidates = candidates.sort_values(
                ["match_score", "overlap_fraction"],
                ascending=[False, False],
                kind="stable",
            )
        best = (
            candidates.iloc[0]
            if not candidates.empty
            and (
                candidates.iloc[0]["overlap_fraction"] >= 0.20
                or candidates.iloc[0]["source_lag_abs"] <= 8
                or candidates.iloc[0]["activation_lag_abs"] <= 10
            )
            else None
        )
        if best is None:
            rows.append(
                {
                    **ref,
                    "best_live_match_variant": "canonical_bos",
                    "best_live_match_ob_id": np.nan,
                    "match_class": "unmatched",
                    "source_lag_bars": np.nan,
                    "activation_lag_bars": np.nan,
                    "geometry_drift_atr": np.nan,
                    "failure_primary_reason": "no_nearby_candidate_exists",
                    "failure_secondary_reason": "legacy_nonlive_may_be_noncausal",
                }
            )
            continue
        source_lag = float(best["ob_source_idx"] - ref["reference_source_idx"])
        activation_lag = float(best["ob_activate_idx"] - ref["reference_source_idx"])
        overlap_fraction = float(best["overlap_fraction"])
        match_class = "exact_source_exact_geometry_match"
        if abs(source_lag) > 0 or overlap_fraction < 0.98:
            match_class = (
                "exact_source_relaxed_geometry_match"
                if abs(source_lag) == 0
                else "nearby_source_relaxed_geometry_match"
            )
        rows.append(
            {
                **ref,
                "best_live_match_variant": "canonical_bos",
                "best_live_match_ob_id": int(best["ob_id"]),
                "match_class": match_class,
                "source_lag_bars": source_lag,
                "activation_lag_bars": activation_lag,
                "geometry_drift_atr": float(best["geometry_drift_atr"]),
                "failure_primary_reason": "",
                "failure_secondary_reason": "",
            }
        )
    casebook = pd.DataFrame(rows)
    matched_mask = casebook["best_live_match_ob_id"].notna()
    summary = {
        "reference_nonlive_count": int(len(casebook)),
        "matched_count": int(matched_mask.sum()),
        "unmatched_count": int((~matched_mask).sum()),
        "matched_fraction": float(matched_mask.mean()) if len(casebook) > 0 else np.nan,
        "median_source_lag_bars": (
            float(
                pd.to_numeric(
                    casebook.loc[matched_mask, "source_lag_bars"], errors="coerce"
                ).median()
            )
            if matched_mask.any()
            else np.nan
        ),
        "median_activation_lag_bars": (
            float(
                pd.to_numeric(
                    casebook.loc[matched_mask, "activation_lag_bars"], errors="coerce"
                ).median()
            )
            if matched_mask.any()
            else np.nan
        ),
        "median_geometry_drift_atr": (
            float(
                pd.to_numeric(
                    casebook.loc[matched_mask, "geometry_drift_atr"], errors="coerce"
                ).median()
            )
            if matched_mask.any()
            else np.nan
        ),
    }
    return casebook, summary


def _first_touch_entry_idx(
    df: pd.DataFrame, event: _LifecycleEvent, level_low: float, level_high: float
) -> int | None:
    last_idx = min(
        [
            value
            for value in [
                event.invalidation_idx,
                event.first_full_idx,
                event.retire_idx,
                len(df) - 1,
            ]
            if value is not None
        ]
    )
    for idx in range(event.activate_idx + 1, last_idx + 1):
        if (
            float(df["high"].iloc[idx]) >= level_low
            and float(df["low"].iloc[idx]) <= level_high
        ):
            return idx
    return None


def _simulate_trade(
    df: pd.DataFrame,
    *,
    side: int,
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    horizon_bars: int = EXECUTION_HORIZON_BARS,
) -> dict[str, Any]:
    high_arr = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    risk = (entry_price - stop_price) if side == 1 else (stop_price - entry_price)
    if not np.isfinite(risk) or risk <= 0:
        return {
            "filled": 0,
            "expectancy_r": np.nan,
            "hit_1r": 0,
            "hit_2r": 0,
            "hit_3r": 0,
            "mae_r": np.nan,
            "mfe_r": np.nan,
            "time_to_stop": np.nan,
            "time_to_1r": np.nan,
            "time_to_2r": np.nan,
            "time_to_3r": np.nan,
        }
    stop_hit_idx: int | None = None
    target_hit_idx = {1: None, 2: None, 3: None}
    mae = 0.0
    mfe = 0.0
    end_idx = min(entry_idx + horizon_bars, len(df) - 1)
    for idx in range(entry_idx + 1, end_idx + 1):
        high = high_arr[idx]
        low = low_arr[idx]
        adverse = (entry_price - low) if side == 1 else (high - entry_price)
        favorable = (high - entry_price) if side == 1 else (entry_price - low)
        mae = max(mae, adverse)
        mfe = max(mfe, favorable)
        stop_hit = low <= stop_price if side == 1 else high >= stop_price
        for mult in (1, 2, 3):
            target = entry_price + side * risk * mult
            target_hit = high >= target if side == 1 else low <= target
            if target_hit and target_hit_idx[mult] is None:
                target_hit_idx[mult] = idx
        if stop_hit:
            stop_hit_idx = idx
            break
        if target_hit_idx[3] is not None:
            break
    realized_r = _safe_fraction((close_arr[end_idx] - entry_price) * side, risk)
    if stop_hit_idx is not None:
        realized_r = -1.0
    elif target_hit_idx[3] is not None:
        realized_r = 3.0
    return {
        "filled": 1,
        "expectancy_r": realized_r,
        "hit_1r": int(
            target_hit_idx[1] is not None
            and (stop_hit_idx is None or target_hit_idx[1] <= stop_hit_idx)
        ),
        "hit_2r": int(
            target_hit_idx[2] is not None
            and (stop_hit_idx is None or target_hit_idx[2] <= stop_hit_idx)
        ),
        "hit_3r": int(
            target_hit_idx[3] is not None
            and (stop_hit_idx is None or target_hit_idx[3] <= stop_hit_idx)
        ),
        "mae_r": _safe_fraction(mae, risk),
        "mfe_r": _safe_fraction(mfe, risk),
        "time_to_stop": (
            float(stop_hit_idx - entry_idx) if stop_hit_idx is not None else np.nan
        ),
        "time_to_1r": (
            float(target_hit_idx[1] - entry_idx)
            if target_hit_idx[1] is not None
            else np.nan
        ),
        "time_to_2r": (
            float(target_hit_idx[2] - entry_idx)
            if target_hit_idx[2] is not None
            else np.nan
        ),
        "time_to_3r": (
            float(target_hit_idx[3] - entry_idx)
            if target_hit_idx[3] is not None
            else np.nan
        ),
    }


def _execution_rows(df: pd.DataFrame, lifecycle: list[_LifecycleEvent]) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    close_arr = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    atr_arr = pd.to_numeric(df.get("atr_14", np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    rows: list[dict[str, Any]] = []
    for event in lifecycle:
        year = int(ts.iloc[event.activate_idx].year)
        stop_price = event.zone_low if event.ob_side == 1 else event.zone_high
        entry_specs = [
            ("bos_close", event.activate_idx, float(close_arr[event.activate_idx])),
            (
                "ob_first_touch",
                _first_touch_entry_idx(df, event, event.zone_low, event.zone_high),
                float(event.zone_high if event.ob_side == 1 else event.zone_low),
            ),
            (
                "ob_mean_threshold",
                _first_touch_entry_idx(df, event, event.zone_mid, event.zone_mid),
                float(event.zone_mid),
            ),
            (
                "ob_body_entry",
                _first_touch_entry_idx(df, event, event.body_low, event.body_high),
                float(event.body_mid),
            ),
        ]
        for policy, entry_idx, entry_price in entry_specs:
            filled = entry_idx is not None
            sim = (
                _simulate_trade(
                    df,
                    side=event.ob_side,
                    entry_idx=(
                        int(entry_idx) if entry_idx is not None else event.activate_idx
                    ),
                    entry_price=entry_price,
                    stop_price=stop_price,
                )
                if filled or policy == "bos_close"
                else {
                    "filled": 0,
                    "expectancy_r": np.nan,
                    "hit_1r": 0,
                    "hit_2r": 0,
                    "hit_3r": 0,
                    "mae_r": np.nan,
                    "mfe_r": np.nan,
                    "time_to_stop": np.nan,
                    "time_to_1r": np.nan,
                    "time_to_2r": np.nan,
                    "time_to_3r": np.nan,
                }
            )
            atr_value = atr_arr[event.activate_idx]
            bos_entry_price = float(close_arr[event.activate_idx])
            bos_risk = abs(bos_entry_price - stop_price)
            policy_risk = (
                abs(entry_price - stop_price)
                if filled or policy == "bos_close"
                else np.nan
            )
            rows.append(
                {
                    "ob_id": event.ob_id,
                    "policy": policy,
                    "side_label": "bull" if event.ob_side == 1 else "bear",
                    "calendar_year": year,
                    "filled": int(sim["filled"]) if policy != "bos_close" else 1,
                    "missed_trade": int(policy != "bos_close" and not filled),
                    "expectancy_r": sim["expectancy_r"],
                    "hit_1r": sim["hit_1r"],
                    "hit_2r": sim["hit_2r"],
                    "hit_3r": sim["hit_3r"],
                    "mae_r": sim["mae_r"],
                    "mfe_r": sim["mfe_r"],
                    "time_to_stop": sim["time_to_stop"],
                    "time_to_1r": sim["time_to_1r"],
                    "time_to_2r": sim["time_to_2r"],
                    "time_to_3r": sim["time_to_3r"],
                    "entry_improvement_atr": (
                        _safe_fraction(
                            (bos_entry_price - entry_price) * event.ob_side, atr_value
                        )
                        if policy != "bos_close" and filled
                        else 0.0
                    ),
                    "stop_compression_improvement_atr": (
                        _safe_fraction(bos_risk - policy_risk, atr_value)
                        if policy != "bos_close" and filled and np.isfinite(policy_risk)
                        else 0.0
                    ),
                    "bos_entry_price": bos_entry_price,
                    "policy_entry_price": entry_price,
                    "stop_price": stop_price,
                }
            )
    return pd.DataFrame(rows)


def _summarize_execution(
    execution_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if execution_rows.empty:
        empty = pd.DataFrame()
        return (
            empty,
            {
                "bos_expectancy": np.nan,
                "ob_first_touch_expectancy": np.nan,
                "ob_mean_threshold_expectancy": np.nan,
                "stop_compression_improvement": np.nan,
                "entry_improvement": np.nan,
                "verdict": "OB mostly redundant to BOS",
            },
            {
                "percent_bos_events_that_retrace_into_canonical_ob_before_invalidation": np.nan,
                "percent_of_retraces_that_materially_improve_entry_price_vs_bos_close": np.nan,
                "average_entry_improvement_in_atr": np.nan,
                "average_stop_compression_improvement_in_atr": np.nan,
                "r_multiple_uplift_vs_bos_close_entry": np.nan,
                "ob_adds_execution_value_bool": False,
                "ob_is_mostly_redundant_to_bos_bool": False,
                "ob_value_casebreakdown": {},
            },
        )
    summary_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", execution_rows)]
    for side in sorted(execution_rows["side_label"].unique()):
        scoped = execution_rows[execution_rows["side_label"] == side]
        scopes.append(("side", side, scoped))
    for year in sorted(execution_rows["calendar_year"].unique()):
        scoped = execution_rows[execution_rows["calendar_year"] == year]
        scopes.append(("year", str(year), scoped))
    for scope_type, scope_value, scoped in scopes:
        for policy in sorted(scoped["policy"].unique()):
            policy_df = scoped[scoped["policy"] == policy]
            filled_mask = policy_df["filled"] == 1
            summary_rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "policy": policy,
                    "sample_count": int(len(policy_df)),
                    "fill_rate": float(policy_df["filled"].mean()),
                    "missed_trade_rate": float(policy_df["missed_trade"].mean()),
                    "win_rate_1r": float(policy_df["hit_1r"].mean()),
                    "win_rate_2r": float(policy_df["hit_2r"].mean()),
                    "win_rate_3r": float(policy_df["hit_3r"].mean()),
                    "expectancy_r": float(
                        pd.to_numeric(policy_df["expectancy_r"], errors="coerce").mean()
                    ),
                    "average_mae_r": float(
                        pd.to_numeric(
                            policy_df.loc[filled_mask, "mae_r"], errors="coerce"
                        ).mean()
                    ),
                    "average_mfe_r": float(
                        pd.to_numeric(
                            policy_df.loc[filled_mask, "mfe_r"], errors="coerce"
                        ).mean()
                    ),
                    "median_mae_r": float(
                        pd.to_numeric(
                            policy_df.loc[filled_mask, "mae_r"], errors="coerce"
                        ).median()
                    ),
                    "median_mfe_r": float(
                        pd.to_numeric(
                            policy_df.loc[filled_mask, "mfe_r"], errors="coerce"
                        ).median()
                    ),
                    "time_to_target_1r": float(
                        pd.to_numeric(policy_df["time_to_1r"], errors="coerce").median()
                    ),
                    "time_to_stop": float(
                        pd.to_numeric(
                            policy_df["time_to_stop"], errors="coerce"
                        ).median()
                    ),
                    "entry_improvement_atr": float(
                        pd.to_numeric(
                            policy_df["entry_improvement_atr"], errors="coerce"
                        ).mean()
                    ),
                    "stop_compression_improvement_atr": float(
                        pd.to_numeric(
                            policy_df["stop_compression_improvement_atr"],
                            errors="coerce",
                        ).mean()
                    ),
                }
            )
    comparison = pd.DataFrame(summary_rows)
    overall = comparison[comparison["scope_type"] == "overall"].set_index("policy")
    bos_expectancy = (
        float(overall.loc["bos_close", "expectancy_r"])
        if "bos_close" in overall.index
        else np.nan
    )
    first_touch_expectancy = (
        float(overall.loc["ob_first_touch", "expectancy_r"])
        if "ob_first_touch" in overall.index
        else np.nan
    )
    mean_expectancy = (
        float(overall.loc["ob_mean_threshold", "expectancy_r"])
        if "ob_mean_threshold" in overall.index
        else np.nan
    )
    redundancy = execution_rows[execution_rows["policy"] == "ob_first_touch"].copy()
    bos_only = execution_rows[execution_rows["policy"] == "bos_close"].set_index(
        "ob_id"
    )
    redundancy["bos_expectancy_r"] = redundancy["ob_id"].map(bos_only["expectancy_r"])
    redundancy["case_label"] = "redundant"
    redundancy.loc[
        (redundancy["filled"] == 1)
        & (redundancy["stop_compression_improvement_atr"] > 0.0)
        & (redundancy["expectancy_r"] >= redundancy["bos_expectancy_r"]),
        "case_label",
    ] = "improves_rr"
    redundancy.loc[
        (redundancy["filled"] == 0) & (redundancy["bos_expectancy_r"] > 0.0),
        "case_label",
    ] = "misses_good_breakout"
    redundancy.loc[
        (redundancy["filled"] == 0) & (redundancy["bos_expectancy_r"] <= 0.0),
        "case_label",
    ] = "filters_bad_breakout"
    redundancy.loc[
        (redundancy["filled"] == 1)
        & (redundancy["expectancy_r"] < redundancy["bos_expectancy_r"]),
        "case_label",
    ] = "delays_entry_but_same_trade"
    redundancy.loc[
        (redundancy["filled"] == 1) & (redundancy["entry_improvement_atr"] < 0.0),
        "case_label",
    ] = "signals_failure"
    redundancy_summary = {
        "percent_bos_events_that_retrace_into_canonical_ob_before_invalidation": float(
            redundancy["filled"].mean()
        ),
        "percent_of_retraces_that_materially_improve_entry_price_vs_bos_close": float(
            (
                pd.to_numeric(redundancy["entry_improvement_atr"], errors="coerce")
                > 0.10
            ).mean()
        ),
        "average_entry_improvement_in_atr": float(
            pd.to_numeric(redundancy["entry_improvement_atr"], errors="coerce").mean()
        ),
        "average_stop_compression_improvement_in_atr": float(
            pd.to_numeric(
                redundancy["stop_compression_improvement_atr"], errors="coerce"
            ).mean()
        ),
        "r_multiple_uplift_vs_bos_close_entry": (
            float(first_touch_expectancy - bos_expectancy)
            if np.isfinite(first_touch_expectancy) and np.isfinite(bos_expectancy)
            else np.nan
        ),
        "ob_adds_execution_value_bool": bool(
            np.isfinite(first_touch_expectancy)
            and np.isfinite(bos_expectancy)
            and first_touch_expectancy > bos_expectancy
        ),
        "ob_is_mostly_redundant_to_bos_bool": bool(
            np.isfinite(first_touch_expectancy)
            and np.isfinite(bos_expectancy)
            and abs(first_touch_expectancy - bos_expectancy) < 0.05
        ),
        "ob_value_casebreakdown": redundancy["case_label"].value_counts().to_dict(),
    }
    execution_summary = {
        "bos_expectancy": bos_expectancy,
        "ob_first_touch_expectancy": first_touch_expectancy,
        "ob_mean_threshold_expectancy": mean_expectancy,
        "stop_compression_improvement": redundancy_summary[
            "average_stop_compression_improvement_in_atr"
        ],
        "entry_improvement": redundancy_summary["average_entry_improvement_in_atr"],
        "verdict": (
            "OB adds value"
            if redundancy_summary["ob_adds_execution_value_bool"]
            else "OB mostly redundant to BOS"
        ),
    }
    return comparison, execution_summary, redundancy_summary


def _frame_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return "```text\n" + frame.to_string(index=False) + "\n```"


def _canonical_contract_memo(
    *,
    coverage_summary: dict[str, Any],
    inventory_summary: dict[str, Any],
    execution_summary: dict[str, Any],
    redundancy_summary: dict[str, Any],
    equivalence_summary: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# OB Canonical Contract Memo",
            "",
            "- Canonical source doctrine: `last_opposing_before_displacement_leg`.",
            "- Parent event type: `bos` for production canonical family.",
            "- Activation doctrine: `ob_activate_idx == ob_parent_bos_idx`.",
            "- Geometry doctrine: full source candle wick-to-wick range.",
            "- Displacement is metadata, not a hard existence gate.",
            "",
            "## Coverage",
            f"- confirmed_bos_count: `{coverage_summary['confirmed_bos_count']}`",
            f"- raw_canonical_ob_count: `{coverage_summary['raw_canonical_bos_ob_count']}`",
            f"- qualified_canonical_ob_count: `{coverage_summary['qualified_canonical_ob_count']}`",
            f"- coverage_fraction: `{coverage_summary['coverage_fraction']}`",
            "",
            "## Inventory",
            f"- fraction_with_top_inventory_within_1atr: `{inventory_summary['fraction_with_top_inventory_within_1atr']}`",
            f"- fraction_with_top_inventory_within_2atr: `{inventory_summary['fraction_with_top_inventory_within_2atr']}`",
            f"- median_distance_to_top_inventory_atr: `{inventory_summary['median_distance_to_top_inventory_atr']}`",
            f"- endpoint_raw_fresh_count_strict: `{inventory_summary['endpoint_raw_fresh_count_strict']}`",
            f"- endpoint_display_fresh_count: `{inventory_summary['endpoint_display_fresh_count']}`",
            "- Freshness note: strict fresh means untouched since activation; display fresh means strict fresh and within the 2 ATR context-near band.",
            "",
            "## Execution",
            f"- BOS expectancy: `{execution_summary['bos_expectancy']}`",
            f"- OB first-touch expectancy: `{execution_summary['ob_first_touch_expectancy']}`",
            f"- OB mean-threshold expectancy: `{execution_summary['ob_mean_threshold_expectancy']}`",
            f"- Redundancy verdict: `{execution_summary['verdict']}`",
            "",
            "## Non-live equivalence",
            f"- matched_count: `{equivalence_summary['matched_count']}` / `{equivalence_summary['reference_nonlive_count']}`",
        ]
    )


def _redundancy_memo(redundancy_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OB Redundancy Diagnostic",
            "",
            f"- percent_bos_events_that_retrace_into_canonical_ob_before_invalidation: `{redundancy_summary['percent_bos_events_that_retrace_into_canonical_ob_before_invalidation']}`",
            f"- percent_of_retraces_that_materially_improve_entry_price_vs_bos_close: `{redundancy_summary['percent_of_retraces_that_materially_improve_entry_price_vs_bos_close']}`",
            f"- average_entry_improvement_in_atr: `{redundancy_summary['average_entry_improvement_in_atr']}`",
            f"- average_stop_compression_improvement_in_atr: `{redundancy_summary['average_stop_compression_improvement_in_atr']}`",
            f"- r_multiple_uplift_vs_bos_close_entry: `{redundancy_summary['r_multiple_uplift_vs_bos_close_entry']}`",
            f"- ob_adds_execution_value_bool: `{redundancy_summary['ob_adds_execution_value_bool']}`",
            f"- ob_is_mostly_redundant_to_bos_bool: `{redundancy_summary['ob_is_mostly_redundant_to_bos_bool']}`",
            f"- ob_value_casebreakdown: `{redundancy_summary['ob_value_casebreakdown']}`",
        ]
    )


def _decision_memo(
    *,
    summary: dict[str, Any],
    inventory_summary: dict[str, Any],
    execution_summary: dict[str, Any],
    redundancy_summary: dict[str, Any],
) -> str:
    freeze = "FREEZE"
    blockers: list[str] = []
    if not summary["sanity_checks"]["activation_equals_parent_confirmation"]:
        blockers.append("activation does not equal parent confirmation for all OBs")
    if not summary["sanity_checks"]["geometry_full_range_consistency"]:
        blockers.append("geometry is not consistently full source candle range")
    if not summary["sanity_checks"]["one_raw_ob_per_confirmed_bos_or_pathological"]:
        blockers.append("raw OB coverage exceeds BOS count or is not auditable")
    if (
        np.isfinite(inventory_summary["fraction_with_top_inventory_within_1atr"])
        and inventory_summary["fraction_with_top_inventory_within_1atr"] < 0.10
    ):
        blockers.append("top inventory remains too far from price too often")
    if not np.isfinite(execution_summary["bos_expectancy"]) or not np.isfinite(
        execution_summary["ob_first_touch_expectancy"]
    ):
        blockers.append("execution harness did not produce stable expectancy metrics")
    if blockers:
        freeze = "DO NOT FREEZE"
    lines = [
        "# OB Canonical Decision Memo",
        "",
        f"- Freeze recommendation: `{freeze}`.",
        f"- OB execution verdict: `{execution_summary['verdict']}`.",
        f"- OB adds execution value: `{redundancy_summary['ob_adds_execution_value_bool']}`.",
        f"- OB mostly redundant to BOS: `{redundancy_summary['ob_is_mostly_redundant_to_bos_bool']}`.",
        "- CHoCH-derived OB should remain separate and unfrozen until a separate sparse-event contract is added.",
        "- Strict fresh and display fresh remain separate by contract; endpoint inventory uses strict fresh for canonical counts and display fresh only as a usability view.",
    ]
    if blockers:
        lines.extend(["", "## Remaining blockers", *[f"- {item}" for item in blockers]])
    return "\n".join(lines)


def _plot_monitorability_series_html(
    inventory_df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str,
    y_columns: list[str],
) -> Path:
    fig = go.Figure()
    for col in y_columns:
        if col not in inventory_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(inventory_df["timestamp"], utc=True, errors="coerce"),
                y=pd.to_numeric(inventory_df[col], errors="coerce"),
                mode="lines",
                name=col,
            )
        )
    fig.update_layout(title=title, template="plotly_white", height=700)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def _plot_equivalence_casebook_overlay_html(
    df: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    accepted: pd.DataFrame,
    outpath: str | Path,
    title: str,
) -> Path:
    scoped = df[
        (
            pd.to_datetime(df["timestamp"], utc=True)
            >= REFERENCE_START - pd.Timedelta(days=3)
        )
        & (
            pd.to_datetime(df["timestamp"], utc=True)
            <= REFERENCE_END + pd.Timedelta(days=3)
        )
    ].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=pd.to_datetime(scoped["timestamp"], utc=True, errors="coerce"),
            open=scoped["open"],
            high=scoped["high"],
            low=scoped["low"],
            close=scoped["close"],
            name="OHLC",
        )
    )
    for _, row in reference.iterrows():
        fig.add_hrect(
            y0=float(row["reference_zone_low"]),
            y1=float(row["reference_zone_high"]),
            x0=pd.to_datetime(row["reference_source_ts"], utc=True),
            x1=pd.to_datetime(row["reference_source_ts"], utc=True)
            + pd.Timedelta(days=2),
            fillcolor=(
                "rgba(37,99,235,0.10)"
                if int(row["reference_side_int"]) == 1
                else "rgba(234,88,12,0.10)"
            ),
            line_color=(
                "rgba(37,99,235,0.60)"
                if int(row["reference_side_int"]) == 1
                else "rgba(234,88,12,0.60)"
            ),
        )
    for _, row in accepted.iterrows():
        fig.add_hrect(
            y0=float(row["ob_zone_low"]),
            y1=float(row["ob_zone_high"]),
            x0=pd.to_datetime(row["timestamp"], utc=True),
            x1=pd.to_datetime(row["timestamp"], utc=True) + pd.Timedelta(days=2),
            fillcolor=(
                "rgba(16,185,129,0.08)"
                if int(row["ob_side"]) == 1
                else "rgba(220,38,38,0.08)"
            ),
            line_color=(
                "rgba(16,185,129,0.55)"
                if int(row["ob_side"]) == 1
                else "rgba(220,38,38,0.55)"
            ),
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800,
        xaxis_rangeslider_visible=False,
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def _plot_endpoint_inventory_validation_html(
    df: pd.DataFrame,
    *,
    inventory_df: pd.DataFrame,
    accepted: pd.DataFrame,
    outpath: str | Path,
    title: str,
) -> Path:
    scoped = df[
        pd.to_datetime(df["timestamp"], utc=True)
        >= (pd.to_datetime(df["timestamp"], utc=True).max() - pd.Timedelta(days=90))
    ].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=pd.to_datetime(scoped["timestamp"], utc=True, errors="coerce"),
            open=scoped["open"],
            high=scoped["high"],
            low=scoped["low"],
            close=scoped["close"],
            name="OHLC",
        )
    )
    endpoint = inventory_df.iloc[-1]
    top_ids = [
        int(value)
        for value in [
            endpoint.get("top_all_ob_id"),
            endpoint.get("top_bull_ob_id"),
            endpoint.get("top_bear_ob_id"),
        ]
        if pd.notna(value)
    ]
    active_rows = (
        accepted[accepted["ob_id"].isin(top_ids)].copy()
        if not accepted.empty
        else pd.DataFrame()
    )
    for _, row in active_rows.iterrows():
        fig.add_hrect(
            y0=float(row["ob_zone_low"]),
            y1=float(row["ob_zone_high"]),
            x0=pd.to_datetime(row["timestamp"], utc=True),
            x1=pd.to_datetime(df["timestamp"], utc=True).max(),
            fillcolor="rgba(124,58,237,0.08)",
            line_color="rgba(124,58,237,0.55)",
            annotation_text=f"Top OB #{int(row['ob_id'])}",
            annotation_position="top left",
        )
    fig.add_annotation(
        x=pd.to_datetime(df["timestamp"], utc=True).max(),
        y=float(pd.to_numeric(df["close"], errors="coerce").iloc[-1]),
        text=f"Top dist={float(endpoint['top_active_distance_atr']) if np.isfinite(endpoint['top_active_distance_atr']) else np.nan:.2f} ATR<br>Good={bool(int(endpoint['endpoint_inventory_is_good_bool']))}",
        showarrow=True,
        arrowhead=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=850,
        xaxis_rangeslider_visible=False,
    )
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath


def build_ob_diagnostic_package(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> dict[str, Any]:
    summary = summarize_ob(df)
    events = _event_frame(df)
    event_audit, lifecycle = _reconstruct_lifecycle(df)
    coverage_audit, coverage_summary = _build_bos_coverage_audit(df)
    monitorability_timeseries, inventory_timeseries, inventory_summary = (
        _build_inventory_surfaces(df, lifecycle)
    )
    equivalence_casebook, equivalence_summary = _build_equivalence_casebook(df)
    execution_rows = _execution_rows(df, lifecycle)
    execution_comparison, execution_summary, redundancy_summary = _summarize_execution(
        execution_rows
    )

    distance_band_audit = (
        inventory_timeseries[
            [
                "row_idx",
                "timestamp",
                "distance_to_nearest_active_ob_atr",
                "distance_to_nearest_fresh_ob_atr",
                "top_active_distance_atr",
                "top_fresh_distance_atr",
                "fraction_with_active_within_0_5atr",
                "fraction_with_active_within_1_0atr",
                "fraction_with_active_within_2_0atr",
                "fraction_with_fresh_within_0_5atr",
                "fraction_with_fresh_within_1_0atr",
                "fraction_with_fresh_within_2_0atr",
            ]
        ].copy()
        if not inventory_timeseries.empty
        else pd.DataFrame()
    )

    contract_memo = _canonical_contract_memo(
        coverage_summary=coverage_summary,
        inventory_summary=inventory_summary,
        execution_summary=execution_summary,
        redundancy_summary=redundancy_summary,
        equivalence_summary=equivalence_summary,
    )
    redundancy_memo = _redundancy_memo(redundancy_summary)
    decision_memo = _decision_memo(
        summary=summary,
        inventory_summary=inventory_summary,
        execution_summary=execution_summary,
        redundancy_summary=redundancy_summary,
    )

    accepted = events.copy()
    return {
        "summary": summary,
        "coverage_summary": coverage_summary,
        "inventory_summary": inventory_summary,
        "equivalence_summary": equivalence_summary,
        "execution_summary": execution_summary,
        "redundancy_summary": redundancy_summary,
        "bos_coverage_audit": coverage_audit,
        "event_audit": event_audit,
        "accepted": accepted,
        "monitorability_timeseries": monitorability_timeseries,
        "inventory_timeseries": inventory_timeseries,
        "distance_band_audit": distance_band_audit,
        "live_vs_nonlive_casebook": equivalence_casebook,
        "execution_comparison": execution_comparison,
        "execution_trade_rows": execution_rows,
        "canonical_contract_memo": contract_memo,
        "redundancy_diagnostic_memo": redundancy_memo,
        "decision_memo": decision_memo,
    }


def write_ob_diagnostic_artifacts(
    diagnostics: dict[str, Any],
    *,
    out_dir: str | Path,
    instrument: str = "XAU_USD",
    timeframe: str = "H4",
    df: pd.DataFrame | None = None,
    write_html: bool = False,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{instrument}_{timeframe}"
    written: dict[str, Path] = {}
    frame_map = {
        f"ob_bos_coverage_audit_{suffix}.csv": diagnostics["bos_coverage_audit"],
        f"ob_distance_band_audit_{suffix}.csv": diagnostics["distance_band_audit"],
        f"ob_inventory_timeseries_{suffix}.csv": diagnostics["inventory_timeseries"],
        f"ob_monitorability_timeseries_{suffix}.csv": diagnostics[
            "monitorability_timeseries"
        ],
        f"ob_live_vs_nonlive_casebook_{suffix}.csv": diagnostics[
            "live_vs_nonlive_casebook"
        ],
        f"ob_bos_vs_ob_execution_comparison_{suffix}.csv": diagnostics[
            "execution_comparison"
        ],
    }
    for filename, frame in frame_map.items():
        written[filename] = write_csv_atomic(frame, out_dir / filename)
    written[f"ob_canonical_contract_memo_{suffix}.md"] = write_text_atomic(
        diagnostics["canonical_contract_memo"],
        out_dir / f"ob_canonical_contract_memo_{suffix}.md",
    )
    written[f"ob_redundancy_diagnostic_{suffix}.md"] = write_text_atomic(
        diagnostics["redundancy_diagnostic_memo"],
        out_dir / f"ob_redundancy_diagnostic_{suffix}.md",
    )
    written[f"ob_canonical_decision_memo_{suffix}.md"] = write_text_atomic(
        diagnostics["decision_memo"],
        out_dir / f"ob_canonical_decision_memo_{suffix}.md",
    )
    if write_html and df is not None and not df.empty:
        inventory_html = _plot_endpoint_inventory_validation_html(
            df,
            inventory_df=diagnostics["inventory_timeseries"],
            accepted=diagnostics["accepted"],
            outpath=out_dir / f"ob_validation_inventory_canonical_{suffix}.html",
            title=f"OB Validation Inventory Canonical — {instrument} {timeframe}",
        )
        written[inventory_html.name] = inventory_html
        monitor_html = _plot_monitorability_series_html(
            diagnostics["inventory_timeseries"],
            outpath=out_dir / f"ob_distance_to_top_inventory_{suffix}.html",
            title=f"OB Distance To Top Inventory — {instrument} {timeframe}",
            y_columns=[
                "top_active_distance_atr",
                "top_fresh_distance_atr",
                "raw_active_count",
                "raw_fresh_count",
            ],
        )
        written[monitor_html.name] = monitor_html
        casebook_html = _plot_equivalence_casebook_overlay_html(
            df,
            reference=diagnostics["live_vs_nonlive_casebook"],
            accepted=diagnostics["accepted"],
            outpath=out_dir / f"ob_live_vs_nonlive_casebook_{suffix}.html",
            title=f"OB Live vs Nonlive Casebook — {instrument} {timeframe}",
        )
        written[casebook_html.name] = casebook_html
    return written
