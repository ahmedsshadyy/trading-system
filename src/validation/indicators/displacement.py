from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.features.bos_context import add_bos_context
from src.indicators.features.choch_context import add_choch_context
from src.indicators.pipelines.build_research import build_research_indicators
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch
from src.indicators.structure.wedges import add_wedges
from src.indicators.research.displacement_research import (
    build_displacement_research_table,
    summarize_displacement_research,
)
from src.indicators.smc.displacement import add_displacement_candle

ANALYSIS_SWING_WINDOW = 4
ANALYSIS_MIN_RETRACE_ATR = 0.7
ANALYSIS_MIN_CONFIRM_BARS = 2
ANALYSIS_MIN_SOURCE_AGE_BARS = 1
ANALYSIS_MIN_BREAK_DISTANCE_ATR = 0.05
ANALYSIS_MIN_BODY_ATR = 0.10

OVERLAP_RADII: tuple[tuple[int, str], ...] = (
    (0, "same_bar"),
    (1, "pm1"),
    (2, "pm2"),
)

BODY_ATR_BUCKET_ORDER = ("1.5_2.0", "2.0_3.0", "3.0+")
BODY_FRAC_BUCKET_ORDER = ("0.60_0.70", "0.70_0.80", "0.80+")

DISPLACEMENT_DEFAULT_THRESHOLDS = {
    "atr_length": 14,
    "body_atr_mult": 1.50,
    "close_extreme_frac": 0.20,
    "min_body_frac": 0.60,
    "max_opposite_wick_frac": 0.20,
}

DISPLACEMENT_TUNING_BASELINE_THRESHOLDS = {
    "atr_length": 14,
    "body_atr_mult": 1.50,
    "close_extreme_frac": 0.20,
    "min_body_frac": 0.60,
    "max_opposite_wick_frac": 0.20,
}

DISPLACEMENT_TUNING_GRID = {
    "body_atr_mult": (1.50, 1.45, 1.40, 1.35),
    "min_body_frac": (0.60, 0.575, 0.55),
    "close_extreme_frac": (0.20, 0.22, 0.25),
    "max_opposite_wick_frac": (0.20, 0.22, 0.25),
}

DISPLACEMENT_TARGET_EVENT_COUNT_MIN = 574
DISPLACEMENT_TARGET_EVENT_COUNT_MAX = 645
DISPLACEMENT_TARGET_EVENT_COUNT_MIDPOINT = (
    DISPLACEMENT_TARGET_EVENT_COUNT_MIN + DISPLACEMENT_TARGET_EVENT_COUNT_MAX
) / 2.0
DISPLACEMENT_OVERLAP_MAX_DROP = 0.05
TOP_QUARTILE_OVERLAP_GUARDRAIL_KEYS = (
    "bos_quality_pm1_rate",
    "bos_tradeable_pm1_rate",
    "choch_quality_pm1_rate",
    "choch_tradeable_pm1_rate",
)


def build_displacement_analysis_base_frame(
    df: pd.DataFrame,
    *,
    instrument: str = "XAU_USD",
) -> pd.DataFrame:
    """Build the shared upstream frame used by displacement analysis scripts."""
    out = normalize_candle_schema(df, require_volume=True)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = build_research_indicators(
        out,
        instrument=instrument,
        swing_window=ANALYSIS_SWING_WINDOW,
        include_vp=False,
        include_avwap=False,
    )
    out = add_bos(
        out,
        min_source_age_bars=ANALYSIS_MIN_SOURCE_AGE_BARS,
        min_break_distance_atr=ANALYSIS_MIN_BREAK_DISTANCE_ATR,
        min_body_atr=ANALYSIS_MIN_BODY_ATR,
    )
    out = add_choch(
        out,
        min_source_age_bars=ANALYSIS_MIN_SOURCE_AGE_BARS,
        min_break_distance_atr=ANALYSIS_MIN_BREAK_DISTANCE_ATR,
        min_body_atr=ANALYSIS_MIN_BODY_ATR,
    )
    out = add_wedges(out)
    return out


def build_displacement_analysis_frame(
    base_df: pd.DataFrame,
    *,
    atr_length: int = 14,
    body_atr_mult: float = 1.35,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
) -> pd.DataFrame:
    """Apply displacement and dependent context layers to a prepared base frame."""
    out = add_displacement_candle(
        base_df,
        atr_length=atr_length,
        body_atr_mult=body_atr_mult,
        close_extreme_frac=close_extreme_frac,
        min_body_frac=min_body_frac,
        max_opposite_wick_frac=max_opposite_wick_frac,
    )
    out = add_bos_context(out, include_forward_diagnostics=True)
    out = add_choch_context(out, include_forward_diagnostics=True)
    return out


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


def _bool_rate(mask: pd.Series) -> float:
    clean = pd.to_numeric(mask, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.mean())


def _event_frame(
    df: pd.DataFrame,
    *,
    event_prefix: str,
) -> pd.DataFrame:
    """Build the canonical raw event-row overlap frame.

    Distances are measured in positional row space via ``Index.get_indexer``
    so they are correct regardless of whether ``df`` has a RangeIndex,
    DatetimeIndex, or any other index type.
    """
    bull_col = f"{event_prefix}_bull"
    bear_col = f"{event_prefix}_bear"
    quality_col = f"{event_prefix}_quality_score"
    tradeable_col = f"{event_prefix}_tradeable_score"
    direction_col = f"{event_prefix}_direction"

    if (
        bull_col not in df.columns
        or bear_col not in df.columns
        or "displacement_flag" not in df.columns
    ):
        return pd.DataFrame()

    mask = (pd.to_numeric(df[bull_col], errors="coerce").fillna(0) == 1) | (
        pd.to_numeric(df[bear_col], errors="coerce").fillna(0) == 1
    )
    scoped = df.loc[mask].copy()
    if scoped.empty:
        return pd.DataFrame()

    # Positional row numbers (0-based) of displacement events within df.
    all_positions = np.arange(len(df))
    disp_flag = (
        pd.to_numeric(df["displacement_flag"], errors="coerce").fillna(0).to_numpy()
        == 1
    )
    displacement_positions = all_positions[disp_flag]

    if len(displacement_positions) == 0:
        nearest_distance = np.full(len(scoped), np.nan)
    else:
        # get_indexer returns positional locations of scoped labels within df.index
        event_positions = df.index.get_indexer(scoped.index)
        nearest_distance = np.array(
            [
                float(np.min(np.abs(displacement_positions - pos)))
                for pos in event_positions
            ],
            dtype=float,
        )

    direction = (
        pd.to_numeric(scoped[direction_col], errors="coerce")
        if direction_col in scoped.columns
        else pd.Series(
            np.where(
                pd.to_numeric(scoped[bull_col], errors="coerce").fillna(0) == 1,
                1,
                -1,
            ),
            index=scoped.index,
            dtype=float,
        )
    )
    event_positions_for_output = df.index.get_indexer(scoped.index)
    event_frame = pd.DataFrame(
        {
            "event_idx": event_positions_for_output,
            "timestamp": (
                pd.to_datetime(scoped["timestamp"], utc=True)
                if "timestamp" in scoped.columns
                else pd.Series(pd.NaT, index=scoped.index)
            ),
            "event_type": event_prefix,
            "event_side": np.where(
                pd.to_numeric(scoped[bull_col], errors="coerce").fillna(0) == 1,
                "bull",
                "bear",
            ),
            "event_direction": direction.to_numpy(dtype=float),
            "quality_score": pd.to_numeric(
                scoped.get(quality_col, np.nan), errors="coerce"
            ).to_numpy(dtype=float),
            "tradeable_score": pd.to_numeric(
                scoped.get(tradeable_col, np.nan), errors="coerce"
            ).to_numpy(dtype=float),
            "nearest_displacement_distance": nearest_distance,
        }
    )
    for radius, label in OVERLAP_RADII:
        event_frame[f"displacement_{label}"] = (
            event_frame["nearest_displacement_distance"] <= radius
        ).astype(np.int8)
    return event_frame


def _nearby_summary(event_frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {"event_count": int(len(event_frame))}
    for _, label in OVERLAP_RADII:
        flag_col = f"displacement_{label}"
        count = int(
            pd.to_numeric(event_frame.get(flag_col), errors="coerce").fillna(0).sum()
        )
        rate = float(count / len(event_frame)) if len(event_frame) else np.nan
        summary[label] = {"count": count, "rate": rate}
    return summary


def _top_quartile_summary(
    event_frame: pd.DataFrame,
    *,
    score_col: str,
) -> dict[str, object]:
    if event_frame.empty or score_col not in event_frame.columns:
        return {"event_count": 0, "threshold": np.nan}
    clean = pd.to_numeric(event_frame[score_col], errors="coerce").dropna()
    if clean.empty:
        return {"event_count": 0, "threshold": np.nan}
    threshold = float(clean.quantile(0.75))
    subset = event_frame.loc[
        pd.to_numeric(event_frame[score_col], errors="coerce") >= threshold
    ].copy()
    summary = _nearby_summary(subset)
    summary["threshold"] = threshold
    return summary


def summarize_displacement_overlap(
    df: pd.DataFrame,
) -> dict[str, object]:
    bos_events = _event_frame(df, event_prefix="bos")
    choch_events = _event_frame(df, event_prefix="choch")
    if bos_events.empty and choch_events.empty:
        return {}

    return {
        "bos": {
            **_nearby_summary(bos_events),
            "top_quartile": {
                "quality": _top_quartile_summary(bos_events, score_col="quality_score"),
                "tradeable": _top_quartile_summary(
                    bos_events, score_col="tradeable_score"
                ),
            },
        },
        "choch": {
            **_nearby_summary(choch_events),
            "top_quartile": {
                "quality": _top_quartile_summary(
                    choch_events, score_col="quality_score"
                ),
                "tradeable": _top_quartile_summary(
                    choch_events, score_col="tradeable_score"
                ),
            },
        },
    }


def extract_displacement_overlap_tables(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    bos_events = _event_frame(df, event_prefix="bos").reset_index(drop=True)
    choch_events = _event_frame(df, event_prefix="choch").reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for event_type, event_frame in (("bos", bos_events), ("choch", choch_events)):
        for score_col, label in (
            ("quality_score", "quality"),
            ("tradeable_score", "tradeable"),
        ):
            top_summary = _top_quartile_summary(event_frame, score_col=score_col)
            row = {
                "event_type": event_type,
                "score_variant": label,
                "threshold": top_summary.get("threshold", np.nan),
                "event_count": int(top_summary.get("event_count", 0)),
            }
            for _, overlap_label in OVERLAP_RADII:
                overlap_summary = top_summary.get(overlap_label, {})
                row[f"{overlap_label}_count"] = int(overlap_summary.get("count", 0))
                row[f"{overlap_label}_rate"] = overlap_summary.get("rate", np.nan)
            summary_rows.append(row)

    return {
        "bos_events": bos_events,
        "choch_events": choch_events,
        "top_quartile_summary": pd.DataFrame(summary_rows),
    }


def _body_atr_monotonicity_ok(research_summary: dict[str, object]) -> bool:
    breakdown = research_summary.get("breakdowns", {}).get("body_atr_bucket", {})
    hold_rates = [
        breakdown[bucket]["hold_5_rate"]
        for bucket in BODY_ATR_BUCKET_ORDER
        if bucket in breakdown and np.isfinite(breakdown[bucket]["hold_5_rate"])
    ]
    if len(hold_rates) < 2:
        return True
    return bool(
        all(
            later >= earlier
            for earlier, later in zip(hold_rates, hold_rates[1:], strict=False)
        )
    )


def _tradeable_bucket_order_ok(research_summary: dict[str, object]) -> bool:
    body_atr_breakdown = research_summary.get("breakdowns", {}).get(
        "body_atr_bucket", {}
    )
    body_frac_breakdown = research_summary.get("breakdowns", {}).get(
        "body_frac_bucket", {}
    )

    checks: list[bool] = []
    if "1.5_2.0" in body_atr_breakdown and "3.0+" in body_atr_breakdown:
        checks.append(
            body_atr_breakdown["3.0+"]["tradeable_score_mean"]
            >= body_atr_breakdown["1.5_2.0"]["tradeable_score_mean"]
        )
    if "0.60_0.70" in body_frac_breakdown and "0.80+" in body_frac_breakdown:
        checks.append(
            body_frac_breakdown["0.80+"]["tradeable_score_mean"]
            >= body_frac_breakdown["0.60_0.70"]["tradeable_score_mean"]
        )
    return bool(all(checks)) if checks else True


def build_displacement_candidate_comparison(
    *,
    detector_summary: dict[str, object],
    research_summary: dict[str, object] | None,
    overlap_summary: dict[str, object] | None,
) -> dict[str, object]:
    research_summary = research_summary or {}
    overlap_summary = overlap_summary or {}
    candidate = {
        "event_count": detector_summary.get("displacement_rows", 0),
        "bull_count": detector_summary.get("bullish_rows", 0),
        "bear_count": detector_summary.get("bearish_rows", 0),
        "hold_5_rate": research_summary.get("hold_rates", {}).get("5", np.nan),
        "failed_5_rate": research_summary.get("failure_rates", {}).get("5", np.nan),
        "tradeable_score_mean": research_summary.get("score_stats", {})
        .get("tradeable", {})
        .get("mean", np.nan),
        "excursion_ratio_5_capped": research_summary.get("excursion_stats", {}).get(
            "excursion_ratio_5_capped", {}
        ),
        "recommended_excursion_ratio_variant": research_summary.get(
            "recommended_excursion_ratio_variant", "capped"
        ),
        "body_atr_hold_monotonicity_ok": _body_atr_monotonicity_ok(research_summary),
        "tradeable_bucket_order_ok": _tradeable_bucket_order_ok(research_summary),
    }

    for event_type in ("bos", "choch"):
        event_summary = overlap_summary.get(event_type, {})
        for _, label in OVERLAP_RADII:
            candidate[f"{event_type}_{label}_rate"] = event_summary.get(label, {}).get(
                "rate", np.nan
            )
        top_quartile = event_summary.get("top_quartile", {})
        for score_variant in ("quality", "tradeable"):
            for _, label in OVERLAP_RADII:
                candidate[f"{event_type}_{score_variant}_{label}_rate"] = (
                    top_quartile.get(score_variant, {})
                    .get(label, {})
                    .get("rate", np.nan)
                )
    return candidate


def displacement_overlap_guardrails_ok(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    max_drop: float = DISPLACEMENT_OVERLAP_MAX_DROP,
) -> bool:
    for key in TOP_QUARTILE_OVERLAP_GUARDRAIL_KEYS:
        baseline_rate = baseline.get(key, np.nan)
        candidate_rate = candidate.get(key, np.nan)
        if not np.isfinite(baseline_rate) or not np.isfinite(candidate_rate):
            return False
        if (baseline_rate - candidate_rate) > max_drop:
            return False
    return True


def displacement_candidate_in_target_range(
    candidate: dict[str, object],
    *,
    min_events: int = DISPLACEMENT_TARGET_EVENT_COUNT_MIN,
    max_events: int = DISPLACEMENT_TARGET_EVENT_COUNT_MAX,
) -> bool:
    event_count = int(candidate.get("event_count", 0))
    return min_events <= event_count <= max_events


def displacement_candidate_passes_acceptance(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    min_events: int = DISPLACEMENT_TARGET_EVENT_COUNT_MIN,
    max_events: int = DISPLACEMENT_TARGET_EVENT_COUNT_MAX,
    max_overlap_drop: float = DISPLACEMENT_OVERLAP_MAX_DROP,
) -> bool:
    return bool(
        displacement_candidate_in_target_range(
            candidate,
            min_events=min_events,
            max_events=max_events,
        )
        and displacement_overlap_guardrails_ok(
            candidate,
            baseline,
            max_drop=max_overlap_drop,
        )
        and bool(candidate.get("body_atr_hold_monotonicity_ok", True))
        and bool(candidate.get("tradeable_bucket_order_ok", True))
    )


def _audit_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "displacement_flag",
        "displacement_bull",
        "displacement_bear",
        "displacement_direction",
        "displacement_body_atr",
        "displacement_range_atr",
        "displacement_body_frac",
        "displacement_close_to_extreme_frac",
        "displacement_opposite_wick_frac",
        "displacement_score",
        "displacement_pass_valid_inputs",
        "displacement_pass_body_atr",
        "displacement_pass_body_frac",
        "displacement_pass_close_to_extreme",
        "displacement_pass_opposite_wick",
    ]
    return [col for col in preferred if col in df.columns]


def _research_audit_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "displacement_event_id",
        "displacement_detect_ts",
        "displacement_side",
        "displacement_direction",
        "displacement_body_atr",
        "displacement_body_frac",
        "displacement_close_to_extreme_frac",
        "displacement_opposite_wick_frac",
        "displacement_quality_score",
        "displacement_follow_through_score",
        "displacement_tradeable_score",
        "displacement_failure_severity_score",
        "displacement_mfe_5_atr",
        "displacement_mae_5_atr",
        "displacement_final_outcome_5",
        "displacement_first_retest_delay",
    ]
    return [col for col in preferred if col in df.columns]


def extract_displacement_audit_tables(
    df: pd.DataFrame,
    *,
    top_n: int = 20,
    body_atr_mult: float = 1.35,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
    atr_length: int = 14,
) -> dict[str, pd.DataFrame]:
    cols = _audit_columns(df)
    flagged = (
        df[df["displacement_flag"] == 1]
        .sort_values(["displacement_score", "displacement_body_atr"], ascending=False)
        .loc[:, cols]
        .head(top_n)
        .copy()
    )

    valid_mask = df["displacement_pass_valid_inputs"] == 1
    miss_body_atr = np.maximum(
        body_atr_mult - df["displacement_body_atr"].to_numpy(dtype=float), 0.0
    ) / max(body_atr_mult, 1e-9)
    miss_body_frac = np.maximum(
        min_body_frac - df["displacement_body_frac"].to_numpy(dtype=float), 0.0
    ) / max(1.0 - min_body_frac, 1e-9)
    miss_close = np.maximum(
        df["displacement_close_to_extreme_frac"].to_numpy(dtype=float)
        - close_extreme_frac,
        0.0,
    ) / max(close_extreme_frac, 1e-9)
    miss_opp = np.maximum(
        df["displacement_opposite_wick_frac"].to_numpy(dtype=float)
        - max_opposite_wick_frac,
        0.0,
    ) / max(max_opposite_wick_frac, 1e-9)
    near_miss_distance = np.nan_to_num(
        miss_body_atr + miss_body_frac + miss_close + miss_opp,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    near_miss = df.loc[valid_mask & (df["displacement_flag"] == 0), cols].copy()
    near_miss["near_miss_distance"] = near_miss_distance[
        valid_mask & (df["displacement_flag"] == 0)
    ]
    near_miss = near_miss.sort_values(
        ["near_miss_distance", "displacement_score"], ascending=[True, False]
    ).head(top_n)

    invalid_rows = (
        df.loc[df["displacement_pass_valid_inputs"] == 0, cols].copy().head(top_n)
    )

    atr_col = f"atr_{atr_length}"
    if atr_col in df.columns:
        atr_values = df[atr_col].to_numpy(dtype=float)
        invalid_atr = (
            df.loc[~np.isfinite(atr_values) | (atr_values <= 0), cols]
            .copy()
            .head(top_n)
        )
    else:
        invalid_atr = pd.DataFrame(columns=cols)

    return {
        "strongest": flagged.reset_index(drop=True),
        "near_miss": near_miss.reset_index(drop=True),
        "invalid_inputs": invalid_rows.reset_index(drop=True),
        "invalid_atr": invalid_atr.reset_index(drop=True),
    }


def extract_displacement_research_audit_tables(
    research_table: pd.DataFrame,
    *,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    if research_table.empty:
        return {
            "strongest_quality": pd.DataFrame(),
            "strongest_follow_through": pd.DataFrame(),
            "strongest_failures": pd.DataFrame(),
        }

    cols = _research_audit_columns(research_table)
    strongest_quality = (
        research_table.sort_values(
            ["displacement_quality_score", "displacement_body_atr"],
            ascending=False,
        )
        .loc[:, cols]
        .head(top_n)
        .reset_index(drop=True)
    )
    strongest_follow_through = (
        research_table.sort_values(
            ["displacement_follow_through_score", "displacement_mfe_5_atr"],
            ascending=False,
        )
        .loc[:, cols]
        .head(top_n)
        .reset_index(drop=True)
    )
    strongest_failures = (
        research_table.sort_values(
            ["displacement_failure_severity_score", "displacement_mae_5_atr"],
            ascending=False,
        )
        .loc[:, cols]
        .head(top_n)
        .reset_index(drop=True)
    )
    return {
        "strongest_quality": strongest_quality,
        "strongest_follow_through": strongest_follow_through,
        "strongest_failures": strongest_failures,
    }


def summarize_displacement(
    df: pd.DataFrame,
    *,
    atr_length: int = 14,
    body_atr_mult: float = 1.35,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
    top_n: int = 20,
    research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    atr_col = f"atr_{atr_length}"
    if atr_col in df.columns:
        atr_values = df[atr_col].to_numpy(dtype=float)
        invalid_atr_rows = int((~np.isfinite(atr_values) | (atr_values <= 0)).sum())
    else:
        invalid_atr_rows = 0

    zero_range_rows = int(
        (
            (~np.isfinite(df["displacement_range"].to_numpy(dtype=float)))
            | (df["displacement_range"].to_numpy(dtype=float) <= 0)
        ).sum()
    )

    detector_tables = extract_displacement_audit_tables(
        df,
        top_n=top_n,
        body_atr_mult=body_atr_mult,
        close_extreme_frac=close_extreme_frac,
        min_body_frac=min_body_frac,
        max_opposite_wick_frac=max_opposite_wick_frac,
        atr_length=atr_length,
    )

    summary: dict[str, object] = {
        "rows": int(len(df)),
        "eligible_rows": int(df["displacement_pass_valid_inputs"].sum()),
        "displacement_rows": int(df["displacement_flag"].sum()),
        "bullish_rows": int(df["displacement_bull"].sum()),
        "bearish_rows": int(df["displacement_bear"].sum()),
        "invalid_atr_rows": invalid_atr_rows,
        "zero_range_rows": zero_range_rows,
        "warmup_rows": int((df["displacement_atr_ready"] == 0).sum()),
        "thresholds": {
            "atr_length": atr_length,
            "body_atr_mult": body_atr_mult,
            "close_extreme_frac": close_extreme_frac,
            "min_body_frac": min_body_frac,
            "max_opposite_wick_frac": max_opposite_wick_frac,
        },
        "pass_counts": {
            "body_atr": int(df["displacement_pass_body_atr"].sum()),
            "body_frac": int(df["displacement_pass_body_frac"].sum()),
            "close_to_extreme": int(df["displacement_pass_close_to_extreme"].sum()),
            "opposite_wick": int(df["displacement_pass_opposite_wick"].sum()),
        },
        "metric_stats": {
            "body_atr": _continuous_stats(df["displacement_body_atr"]),
            "range_atr": _continuous_stats(df["displacement_range_atr"]),
            "body_frac": _continuous_stats(df["displacement_body_frac"]),
            "close_to_extreme_frac": _continuous_stats(
                df["displacement_close_to_extreme_frac"]
            ),
            "opposite_wick_frac": _continuous_stats(
                df["displacement_opposite_wick_frac"]
            ),
            "score": _continuous_stats(df["displacement_score"]),
        },
        "audit_preview_counts": {
            "strongest": int(len(detector_tables["strongest"])),
            "near_miss": int(len(detector_tables["near_miss"])),
            "invalid_inputs": int(len(detector_tables["invalid_inputs"])),
            "invalid_atr": int(len(detector_tables["invalid_atr"])),
        },
    }

    overlap_summary = summarize_displacement_overlap(df)
    if overlap_summary:
        summary["overlap_summary"] = overlap_summary

    if research_table is not None:
        research_summary = summarize_displacement_research(research_table)
        summary["research_summary"] = research_summary
        research_tables = extract_displacement_research_audit_tables(
            research_table, top_n=top_n
        )
        summary["research_audit_preview_counts"] = {
            key: int(len(value)) for key, value in research_tables.items()
        }
        summary["candidate_comparison"] = build_displacement_candidate_comparison(
            detector_summary=summary,
            research_summary=research_summary,
            overlap_summary=overlap_summary,
        )
    elif overlap_summary:
        summary["candidate_comparison"] = build_displacement_candidate_comparison(
            detector_summary=summary,
            research_summary=None,
            overlap_summary=overlap_summary,
        )

    return summary


def validate_displacement(
    df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Displacement Validation",
    top_n: int = 20,
    atr_length: int = 14,
    body_atr_mult: float = 1.35,
    close_extreme_frac: float = 0.20,
    min_body_frac: float = 0.60,
    max_opposite_wick_frac: float = 0.20,
    research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    audit_df = full_df if full_df is not None else df
    if research_table is None:
        research_table = build_displacement_research_table(
            audit_df, atr_length=atr_length
        )

    detector_tables = extract_displacement_audit_tables(
        audit_df,
        top_n=top_n,
        atr_length=atr_length,
        body_atr_mult=body_atr_mult,
        close_extreme_frac=close_extreme_frac,
        min_body_frac=min_body_frac,
        max_opposite_wick_frac=max_opposite_wick_frac,
    )
    research_tables = extract_displacement_research_audit_tables(
        research_table if research_table is not None else pd.DataFrame(),
        top_n=top_n,
    )
    overlap_tables = extract_displacement_overlap_tables(audit_df)

    summary = summarize_displacement(
        audit_df,
        top_n=top_n,
        atr_length=atr_length,
        body_atr_mult=body_atr_mult,
        close_extreme_frac=close_extreme_frac,
        min_body_frac=min_body_frac,
        max_opposite_wick_frac=max_opposite_wick_frac,
        research_table=research_table,
    )
    if research_table is not None:
        summary["global_research_summary"] = summarize_displacement_research(
            research_table
        )

    x_values = df["timestamp"] if "timestamp" in df.columns else df.index
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#0f9d58",
            decreasing_line_color="#c5221f",
        )
    )

    hover_cols = [
        "displacement_body_atr",
        "displacement_range_atr",
        "displacement_body_frac",
        "displacement_close_to_extreme_frac",
        "displacement_opposite_wick_frac",
        "displacement_score",
    ]
    hover_data = df[hover_cols].to_numpy(dtype=float)
    hovertemplate = (
        "Time=%{x}<br>"
        "Price=%{y:.5f}<br>"
        "body_atr=%{customdata[0]:.3f}<br>"
        "range_atr=%{customdata[1]:.3f}<br>"
        "body_frac=%{customdata[2]:.3f}<br>"
        "close_to_extreme=%{customdata[3]:.3f}<br>"
        "opp_wick=%{customdata[4]:.3f}<br>"
        "score=%{customdata[5]:.3f}<extra></extra>"
    )

    bull = df["displacement_bull"] == 1
    bear = df["displacement_bear"] == 1
    for side_mask, name, symbol, y_col in (
        (bull, "Bull Displacement", "triangle-up", "high"),
        (bear, "Bear Displacement", "triangle-down", "low"),
    ):
        subset = df.loc[side_mask]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values[side_mask],
                y=subset[y_col],
                mode="markers",
                name=name,
                marker=dict(
                    symbol=symbol,
                    size=11,
                    color=subset["displacement_score"],
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    line=dict(color="#111111", width=0.5),
                    showscale=(name == "Bull Displacement"),
                    colorbar=dict(title="Score"),
                ),
                customdata=hover_data[side_mask],
                hovertemplate=hovertemplate,
            )
        )

    fig.update_layout(
        title=title,
        height=760,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
        ),
    )

    html_path: str | None = None
    if outpath is not None:
        out_file = Path(outpath)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_file)
        html_path = str(out_file)

    return {
        "summary": summary,
        "tables": {
            "detector": detector_tables,
            "research": research_tables,
            "overlap": overlap_tables,
        },
        "figure": fig,
        "html_path": html_path,
        "research_table": research_table,
    }
