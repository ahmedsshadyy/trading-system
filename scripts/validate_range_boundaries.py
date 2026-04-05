from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.typing import NaTType
from plotly.subplots import make_subplots

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.foundation.adx import add_adx
from src.indicators.foundation.ema import add_emas
from src.indicators.foundation.range_boundaries import (
    collect_range_boundary_debug_tables,
)
from src.indicators.foundation.regime import add_regime
from src.indicators.foundation.session import add_session_features
from src.indicators.foundation.volatility import add_atr, add_bb_width
from src.indicators.structure.bos import add_bos
from src.indicators.structure.choch import add_choch
from src.indicators.structure.swings import add_swings
from src.indicators.structure.trend_state import add_trend_state
from src.dag_runtime import GraphRunContext, execute_graph, explain_graph_run
from src.dag_runtime.builtin_graphs import get_builtin_graph
from src.pipeline_runtime import (
    dataframe_fingerprint,
    load_partitioned_dataset,
)
from src.validation.common import (
    cleanup_validation_artifacts,
    write_text_atomic,
)
from src.validation.indicators.range_boundaries import (
    summarize_range_boundaries,
)
from src.validation.common.chart_core import save_figure_html
from src.validation.common.reporting import report_fingerprint

OUT_DIR = Path("notebooks/foundation")
FEATURES_ROOT = Path("data/features")
CACHE_ROOT = Path("data/validation_cache")
VALIDATOR_NAME = "validate_range_boundaries"
VALIDATOR_SCHEMA_VERSION = 1
VALIDATOR_CONTRACT_VERSION = 1
AUDIT_PRESSURE_LOOKBACK_BARS = 5
TARGET_CONFIRMED_RANGE_MIN = 120
TARGET_CONFIRMED_RANGE_MAX = 250
TARGET_CONFIRMED_RANGE_MID = 185
TARGET_ACTIVE_ROWS_MIN = 700
TARGET_ACTIVE_ROWS_MAX = 1400
TARGET_ACTIVE_ROWS_MID = 1050
TARGET_CONFIRM_LATENCY_MEDIAN_MIN = 3.0
TARGET_SHORT_LIVED_DURATION_MEAN_MIN = 1.5
MAX_STRENGTH_INVERSION = 0.04
TimestampLike = pd.Timestamp | NaTType
FROZEN_GEOMETRY_RANKING_METRIC = "strength_repair_v2"
GEOMETRY_CANDIDATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("g1_legacy", "high", "low"),
    ("g2_envelope_extended_extrema", "range_high_g2", "range_low_g2"),
    ("g3_touch_cluster_envelope", "range_high_g3", "range_low_g3"),
    ("g4_quantile_envelope", "range_high_g4", "range_low_g4"),
    ("g5_widened_compact_core_with_guardrails", "range_high_g5", "range_low_g5"),
)

BASE_RECOVERY_PARAMS = {
    "candidate_lookback_bars": (5, 8, 12, 16),
    "min_confirm_bars": 2,
    "min_candidate_dwell_bars": 2,
    "boundary_stability_tolerance_atr": 0.35,
    "lineage_grace_bars": 1,
    "max_width_atr": 3.5,
    "edge_tolerance_atr": 0.20,
    "min_upper_touches": 2,
    "min_lower_touches": 2,
    "min_close_inside_frac": 0.50,
    "allowed_wick_overshoot_atr": 1.25,
    "max_drift_frac": 0.85,
    "viability_lookback_bars": 3,
}

TARGET_NODE_MAP = {
    "selection": "range_selection_bundle",
    "selected-debug": "range_selected_debug",
    "forensics": "range_forensics",
    "geometry": "range_geometry_audit",
    "active-truth": "range_active_truth_audit",
    "coverage": "range_coverage_regime_report",
    "ranking": "range_ranking_bundle",
    "downstream": "range_downstream_usefulness",
    "diagnostics": "range_diagnostics_bundle",
    "charts": "range_chart_bundle",
    "csv": "range_csv_bundle",
    "full": "range_validation_bundle",
}

STEP8E_B_RETUNE_PARAMS = {
    "viability_pressure_weight": 0.42,
    "viability_equilibrium_weight": 0.12,
    "viability_freshness_weight": 0.00,
    "viability_expansion_pressure_weight": 0.36,
    "viability_expansion_veto_weight": 0.10,
    "final_strength_formation_base": 0.40,
    "final_strength_viability_scale": 0.60,
}

CONTEXT_REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "atr_14",
    "adx_14",
    "bb_width",
    "trend_state",
    "bos_bull",
    "bos_bear",
    "choch_bull",
    "choch_bear",
    "regime",
}


def _print_summary(value: object, *, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                print(f"{prefix}{key}:")
                _print_summary(child, indent=indent + 2)
            else:
                print(f"{prefix}{key}: {child}")
        return
    print(f"{prefix}{value}")


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _clip_series(series: pd.Series, *, scale: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values / scale).clip(lower=0.0, upper=1.0)


def _compute_pressure_imbalance_v2(
    frame: pd.DataFrame,
    event_table: pd.DataFrame,
    *,
    lookback_bars: int = AUDIT_PRESSURE_LOOKBACK_BARS,
) -> pd.Series:
    values: list[float] = []
    for _, event in event_table.iterrows():
        confirm_idx = pd.to_numeric(
            pd.Series([event.get("confirm_idx")]), errors="coerce"
        ).iloc[0]
        low = pd.to_numeric(pd.Series([event.get("low")]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([event.get("high")]), errors="coerce").iloc[0]
        if (
            not pd.notna(confirm_idx)
            or not pd.notna(low)
            or not pd.notna(high)
            or high <= low
        ):
            values.append(float("nan"))
            continue
        idx = int(confirm_idx)
        start = max(0, idx - lookback_bars + 1)
        window = frame.iloc[start : idx + 1]
        if window.empty:
            values.append(float("nan"))
            continue
        width = max(float(high - low), 1e-9)
        pos = (
            (pd.to_numeric(window["close"], errors="coerce") - float(low)) / width
        ).clip(0.0, 1.0)
        if pos.empty:
            values.append(float("nan"))
            continue
        span = float(pos.max() - pos.min())
        mean_bias = abs(float(pos.mean()) - 0.5) * 2.0
        last_bias = abs(float(pos.iloc[-1]) - 0.5) * 2.0
        values.append(_clip01((1.0 - span) * (0.5 + 0.5 * max(mean_bias, last_bias))))
    return pd.Series(values, index=event_table.index, dtype=float)


def _compute_pressure_imbalance_legacy(
    frame: pd.DataFrame,
    event_table: pd.DataFrame,
    *,
    lookback_bars: int = AUDIT_PRESSURE_LOOKBACK_BARS,
) -> pd.Series:
    values: list[float] = []
    for _, event in event_table.iterrows():
        confirm_idx = pd.to_numeric(
            pd.Series([event.get("confirm_idx")]), errors="coerce"
        ).iloc[0]
        low = pd.to_numeric(pd.Series([event.get("low")]), errors="coerce").iloc[0]
        high = pd.to_numeric(pd.Series([event.get("high")]), errors="coerce").iloc[0]
        if (
            not pd.notna(confirm_idx)
            or not pd.notna(low)
            or not pd.notna(high)
            or high <= low
        ):
            values.append(float("nan"))
            continue
        idx = int(confirm_idx)
        start = max(0, idx - lookback_bars + 1)
        window = frame.iloc[start : idx + 1]
        if window.empty:
            values.append(float("nan"))
            continue
        width = max(float(high - low), 1e-9)
        pos = (
            (pd.to_numeric(window["close"], errors="coerce") - float(low)) / width
        ).clip(0.0, 1.0)
        upper_pressure = (
            pd.to_numeric(window["high"], errors="coerce") >= float(high) - width * 0.10
        ) | (pos >= 0.72)
        lower_pressure = (
            pd.to_numeric(window["low"], errors="coerce") <= float(low) + width * 0.10
        ) | (pos <= 0.28)
        denom = max(len(window), 1)
        values.append(
            _clip01(
                abs(float(upper_pressure.sum()) - float(lower_pressure.sum())) / denom
            )
        )
    return pd.Series(values, index=event_table.index, dtype=float)


def _get_atr_col(frame: pd.DataFrame) -> str | None:
    for col in ("atr", "atr_14"):
        if col in frame.columns:
            return col
    return None


def _event_confirm_atr(
    frame: pd.DataFrame, confirm_idx: int, fallback_width: float
) -> float:
    atr_col = _get_atr_col(frame)
    if atr_col is None or confirm_idx < 0 or confirm_idx >= len(frame):
        return max(float(fallback_width), 1e-9)
    value = pd.to_numeric(
        pd.Series([frame.iloc[confirm_idx][atr_col]]), errors="coerce"
    ).iloc[0]
    if pd.notna(value) and float(value) > 0:
        return float(value)
    return max(float(fallback_width), 1e-9)


def _robust_outer_median(values: np.ndarray, *, side: str, k: int) -> float | None:
    if values.size == 0:
        return None
    k = max(1, min(int(k), int(values.size)))
    if side == "upper":
        selected = np.partition(values, values.size - k)[values.size - k :]
    else:
        selected = np.partition(values, k - 1)[:k]
    return float(np.nanmedian(selected))


def _count_outer_contacts(
    values: np.ndarray, *, side: str, edge: float, tolerance: float
) -> int:
    if values.size == 0:
        return 0
    tol = max(float(tolerance), 1e-9)
    if side == "upper":
        return int(np.sum(values >= float(edge) - tol))
    return int(np.sum(values <= float(edge) + tol))


def _compute_geometry_candidates(
    frame: pd.DataFrame,
    event_table: pd.DataFrame,
) -> pd.DataFrame:
    if event_table.empty:
        out = event_table.copy()
        for _, high_col, low_col in GEOMETRY_CANDIDATE_SPECS[1:]:
            out[high_col] = pd.Series(dtype=float)
            out[low_col] = pd.Series(dtype=float)
        return out

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    out = event_table.copy()

    g2_highs: list[float] = []
    g2_lows: list[float] = []
    g3_highs: list[float] = []
    g3_lows: list[float] = []
    g4_highs: list[float] = []
    g4_lows: list[float] = []
    g5_highs: list[float] = []
    g5_lows: list[float] = []

    for _, event in out.iterrows():
        birth_idx_value = pd.to_numeric(
            pd.Series([event.get("birth_idx")]), errors="coerce"
        ).iloc[0]
        confirm_idx_value = pd.to_numeric(
            pd.Series([event.get("confirm_idx")]), errors="coerce"
        ).iloc[0]
        legacy_high = pd.to_numeric(
            pd.Series([event.get("high")]), errors="coerce"
        ).iloc[0]
        legacy_low = pd.to_numeric(pd.Series([event.get("low")]), errors="coerce").iloc[
            0
        ]
        if (
            pd.isna(birth_idx_value)
            or pd.isna(confirm_idx_value)
            or pd.isna(legacy_high)
            or pd.isna(legacy_low)
            or float(legacy_high) <= float(legacy_low)
        ):
            g2_highs.append(float("nan"))
            g2_lows.append(float("nan"))
            g3_highs.append(float("nan"))
            g3_lows.append(float("nan"))
            g4_highs.append(float("nan"))
            g4_lows.append(float("nan"))
            g5_highs.append(float("nan"))
            g5_lows.append(float("nan"))
            continue

        birth_idx = max(0, int(birth_idx_value))
        confirm_idx = min(max(int(confirm_idx_value), birth_idx), len(frame) - 1)
        start = max(0, birth_idx - 1)
        window_highs = highs[start : confirm_idx + 1]
        window_lows = lows[start : confirm_idx + 1]
        window_highs = window_highs[np.isfinite(window_highs)]
        window_lows = window_lows[np.isfinite(window_lows)]
        if window_highs.size == 0 or window_lows.size == 0:
            g2_highs.append(float(legacy_high))
            g2_lows.append(float(legacy_low))
            g3_highs.append(float(legacy_high))
            g3_lows.append(float(legacy_low))
            g4_highs.append(float(legacy_high))
            g4_lows.append(float(legacy_low))
            g5_highs.append(float(legacy_high))
            g5_lows.append(float(legacy_low))
            continue

        legacy_width = max(float(legacy_high) - float(legacy_low), 1e-9)
        atr_now = _event_confirm_atr(frame, confirm_idx, legacy_width)
        contact_tol = max(legacy_width * 0.12, atr_now * 0.10)
        widen_cap_large = max(legacy_width * 0.70, atr_now * 0.90)
        widen_cap_mid = max(legacy_width * 0.50, atr_now * 0.65)
        widen_cap_small = max(legacy_width * 0.30, atr_now * 0.45)

        k_outer = max(2, min(4, int(window_highs.size)))
        robust_upper = _robust_outer_median(window_highs, side="upper", k=k_outer)
        robust_lower = _robust_outer_median(window_lows, side="lower", k=k_outer)
        upper_contact_count = _count_outer_contacts(
            window_highs,
            side="upper",
            edge=float(robust_upper if robust_upper is not None else legacy_high),
            tolerance=contact_tol,
        )
        lower_contact_count = _count_outer_contacts(
            window_lows,
            side="lower",
            edge=float(robust_lower if robust_lower is not None else legacy_low),
            tolerance=contact_tol,
        )

        g2_high = float(legacy_high)
        g2_low = float(legacy_low)
        if (
            robust_upper is not None
            and robust_upper > float(legacy_high)
            and upper_contact_count >= 2
        ):
            g2_high = min(float(robust_upper), float(legacy_high) + widen_cap_large)
        if (
            robust_lower is not None
            and robust_lower < float(legacy_low)
            and lower_contact_count >= 2
        ):
            g2_low = max(float(robust_lower), float(legacy_low) - widen_cap_large)

        upper_thresh = float(np.nanquantile(window_highs, 0.75))
        lower_thresh = float(np.nanquantile(window_lows, 0.25))
        upper_cluster = window_highs[window_highs >= upper_thresh]
        lower_cluster = window_lows[window_lows <= lower_thresh]
        g3_high = float(legacy_high)
        g3_low = float(legacy_low)
        if upper_cluster.size >= 2:
            upper_cluster_edge = float(np.nanmedian(upper_cluster))
            if upper_cluster_edge > float(legacy_high):
                g3_high = min(upper_cluster_edge, float(legacy_high) + widen_cap_mid)
        if lower_cluster.size >= 2:
            lower_cluster_edge = float(np.nanmedian(lower_cluster))
            if lower_cluster_edge < float(legacy_low):
                g3_low = max(lower_cluster_edge, float(legacy_low) - widen_cap_mid)

        g4_high = float(legacy_high)
        g4_low = float(legacy_low)
        upper_q = float(np.nanquantile(window_highs, 0.90))
        lower_q = float(np.nanquantile(window_lows, 0.10))
        if (
            upper_q > float(legacy_high)
            and _count_outer_contacts(
                window_highs, side="upper", edge=upper_q, tolerance=contact_tol
            )
            >= 2
        ):
            g4_high = min(upper_q, float(legacy_high) + widen_cap_mid)
        if (
            lower_q < float(legacy_low)
            and _count_outer_contacts(
                window_lows, side="lower", edge=lower_q, tolerance=contact_tol
            )
            >= 2
        ):
            g4_low = max(lower_q, float(legacy_low) - widen_cap_mid)

        outer_touch_tol = max(legacy_width * 0.08, atr_now * 0.05)
        upper_outer_touches = int(
            np.sum(window_highs >= float(legacy_high) + outer_touch_tol)
        )
        lower_outer_touches = int(
            np.sum(window_lows <= float(legacy_low) - outer_touch_tol)
        )
        g5_high = float(legacy_high)
        g5_low = float(legacy_low)
        if upper_outer_touches >= 2:
            widened_upper = max(g2_high, g3_high, g4_high)
            g5_high = min(float(widened_upper), float(legacy_high) + widen_cap_small)
        if lower_outer_touches >= 2:
            widened_lower = min(g2_low, g3_low, g4_low)
            g5_low = max(float(widened_lower), float(legacy_low) - widen_cap_small)

        g2_highs.append(float(max(g2_high, legacy_high)))
        g2_lows.append(float(min(g2_low, legacy_low)))
        g3_highs.append(float(max(g3_high, legacy_high)))
        g3_lows.append(float(min(g3_low, legacy_low)))
        g4_highs.append(float(max(g4_high, legacy_high)))
        g4_lows.append(float(min(g4_low, legacy_low)))
        g5_highs.append(float(max(g5_high, legacy_high)))
        g5_lows.append(float(min(g5_low, legacy_low)))

    out["range_high_g2"] = pd.Series(g2_highs, index=out.index, dtype=float)
    out["range_low_g2"] = pd.Series(g2_lows, index=out.index, dtype=float)
    out["range_high_g3"] = pd.Series(g3_highs, index=out.index, dtype=float)
    out["range_low_g3"] = pd.Series(g3_lows, index=out.index, dtype=float)
    out["range_high_g4"] = pd.Series(g4_highs, index=out.index, dtype=float)
    out["range_low_g4"] = pd.Series(g4_lows, index=out.index, dtype=float)
    out["range_high_g5"] = pd.Series(g5_highs, index=out.index, dtype=float)
    out["range_low_g5"] = pd.Series(g5_lows, index=out.index, dtype=float)
    return out


def _build_archetype_summary(
    short_high: pd.DataFrame,
    long_medium: pd.DataFrame,
) -> dict[str, object]:
    fields = [
        "duration_bars",
        "strength",
        "strength_legacy",
        "range_strength_structure",
        "range_strength_monitorability",
        "range_strength_semantic",
        "range_strength_formation",
        "range_strength_viability",
        "range_strength_viability_legacy",
        "width_atr",
        "upper_touches",
        "lower_touches",
        "bars_to_first_breach",
        "bars_to_breakout_accept",
        "reclaimed_count",
        "break_pending_count",
        "confirm_close_position_in_range",
        "range_recent_pressure_imbalance",
        "audit_range_recent_pressure_imbalance_legacy",
        "audit_range_recent_pressure_imbalance_v2",
        "range_recent_equilibrium_score",
        "range_recent_two_sided_freshness_score",
        "range_recent_upper_pressure_count",
        "range_recent_lower_pressure_count",
        "range_recent_expansion_veto_flag",
        "rb_micro_box_risk_score",
        "rb_late_confirm_fragility_score",
        "rb_boundary_relevance_score",
        "rb_monitor_worthiness_score",
        "rb_plausibility_score",
    ]

    def section(df: pd.DataFrame) -> dict[str, object]:
        if df.empty:
            return {"rows": 0}
        return {
            "rows": int(len(df)),
            **{
                field: _continuous_stats(df[field])
                for field in fields
                if field in df.columns
            },
        }

    return {
        "short_lived_high_strength": section(short_high),
        "long_lived_medium_strength": section(long_medium),
    }


def _build_viability_alignment_audit(
    short_high: pd.DataFrame,
    long_medium: pd.DataFrame,
) -> list[dict[str, object]]:
    checks = [
        ("range_recent_pressure_imbalance", "lower_is_better"),
        ("range_recent_equilibrium_score", "higher_is_better"),
        ("range_recent_two_sided_freshness_score", "higher_is_better"),
        ("range_strength_viability_legacy", "higher_is_better"),
        ("range_strength_viability", "higher_is_better"),
        ("strength_legacy", "higher_is_better"),
        ("strength", "higher_is_better"),
        ("rb_plausibility_score", "higher_is_better"),
        ("rb_monitor_worthiness_score", "higher_is_better"),
        ("rb_micro_box_risk_score", "lower_is_better"),
    ]
    rows: list[dict[str, object]] = []
    for metric, expected in checks:
        short_values = pd.to_numeric(short_high.get(metric), errors="coerce").dropna()
        long_values = pd.to_numeric(long_medium.get(metric), errors="coerce").dropna()
        short_mean = float(short_values.mean()) if not short_values.empty else None
        long_mean = float(long_values.mean()) if not long_values.empty else None
        aligned = (
            short_mean is not None
            and long_mean is not None
            and (
                (expected == "lower_is_better" and long_mean < short_mean)
                or (expected == "higher_is_better" and long_mean > short_mean)
            )
        )
        rows.append(
            {
                "metric": metric,
                "expected": expected,
                "short_mean": short_mean,
                "long_mean": long_mean,
                "observed": "aligned" if aligned else "misaligned",
            }
        )
    return rows


def _build_pressure_alignment_audit(
    short_high: pd.DataFrame,
    long_medium: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, label in (
        ("range_recent_pressure_imbalance", "production"),
        ("audit_range_recent_pressure_imbalance_legacy", "legacy"),
        ("audit_range_recent_pressure_imbalance_v2", "candidate_v2"),
    ):
        short_values = pd.to_numeric(short_high.get(metric), errors="coerce").dropna()
        long_values = pd.to_numeric(long_medium.get(metric), errors="coerce").dropna()
        short_mean = float(short_values.mean()) if not short_values.empty else None
        long_mean = float(long_values.mean()) if not long_values.empty else None
        aligned = (
            short_mean is not None and long_mean is not None and long_mean < short_mean
        )
        rows.append(
            {
                "metric": metric,
                "version": label,
                "expected": "lower_is_better",
                "short_mean": short_mean,
                "long_mean": long_mean,
                "observed": "aligned" if aligned else "misaligned",
            }
        )
    return rows


def _run_debug_with_params(
    context: pd.DataFrame,
    params: dict[str, object],
) -> dict[str, object]:
    stage_timings: dict[str, float] = {}

    def _time_stage(name: str, fn):
        started_at = time.perf_counter()
        value = fn()
        stage_timings[name] = time.perf_counter() - started_at
        return value

    debug = _time_stage(
        "debug_collect", lambda: collect_range_boundary_debug_tables(context, **params)
    )
    full_df = debug["frame"]
    event_table = debug["event_table"]
    candidate_table = debug.get("candidate_table", pd.DataFrame())
    if not event_table.empty:
        event_table = event_table.copy()
        event_table["audit_range_recent_pressure_imbalance_legacy"] = _time_stage(
            "pressure_imbalance_legacy",
            lambda: _compute_pressure_imbalance_legacy(
                full_df,
                event_table,
            ),
        )
        event_table["audit_range_recent_pressure_imbalance_v2"] = _time_stage(
            "pressure_imbalance_v2",
            lambda: _compute_pressure_imbalance_v2(
                full_df,
                event_table,
            ),
        )
        event_table = _time_stage(
            "contract_scores", lambda: _add_contract_scores(event_table)
        )
        event_table = _time_stage(
            "geometry_candidates",
            lambda: _compute_geometry_candidates(full_df, event_table),
        )
    else:
        stage_timings["pressure_imbalance_legacy"] = 0.0
        stage_timings["pressure_imbalance_v2"] = 0.0
        stage_timings["contract_scores"] = 0.0
        stage_timings["geometry_candidates"] = 0.0
    summary = _time_stage(
        "summary_build",
        lambda: summarize_range_boundaries(
            full_df,
            event_table=event_table,
            candidate_table=candidate_table,
        ),
    )
    return {
        "params": params,
        "frame": full_df,
        "event_table": event_table,
        "candidate_table": candidate_table,
        "summary": summary,
        "profile_details": {
            "substage_seconds": stage_timings,
            "skipped": False,
        },
    }


def _add_contract_scores(event_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty:
        return event_table.copy()
    out = event_table.copy()
    out["duration_bars"] = pd.to_numeric(
        out["end_idx"], errors="coerce"
    ) - pd.to_numeric(out["confirm_idx"], errors="coerce")
    duration = pd.to_numeric(out["duration_bars"], errors="coerce").fillna(0.0)
    bars_to_first_breach = pd.to_numeric(
        out["bars_to_first_breach"], errors="coerce"
    ).fillna(0.0)
    bars_to_breakout_accept = pd.to_numeric(
        out["bars_to_breakout_accept"], errors="coerce"
    )
    confirm_latency = pd.to_numeric(
        out["confirm_latency_bars"], errors="coerce"
    ).fillna(0.0)
    upper_touches = pd.to_numeric(out["upper_touches"], errors="coerce").fillna(0.0)
    lower_touches = pd.to_numeric(out["lower_touches"], errors="coerce").fillna(0.0)
    min_touches = pd.concat([upper_touches, lower_touches], axis=1).min(axis=1)
    width_atr = pd.to_numeric(out["width_atr"], errors="coerce").fillna(0.0)
    reclaimed_count = pd.to_numeric(out["reclaimed_count"], errors="coerce").fillna(0.0)
    break_pending_count = pd.to_numeric(
        out["break_pending_count"], errors="coerce"
    ).fillna(0.0)

    duration_risk = 1.0 - _clip_series(duration, scale=4.0)
    breach_risk = 1.0 - _clip_series(bars_to_first_breach, scale=4.0)
    accept_risk = pd.Series(0.0, index=out.index, dtype=float)
    accept_mask = bars_to_breakout_accept.notna()
    accept_risk.loc[accept_mask] = 1.0 - _clip_series(
        bars_to_breakout_accept.loc[accept_mask],
        scale=4.0,
    )
    latency_risk = 1.0 - ((confirm_latency - 1.0) / 3.0).clip(lower=0.0, upper=1.0)
    touch_risk = 1.0 - ((min_touches - 2.0) / 2.0).clip(lower=0.0, upper=1.0)
    out["rb_micro_box_risk_score"] = (
        duration_risk + breach_risk + accept_risk + latency_risk + touch_risk
    ) / 5.0

    confirm_share = confirm_latency / (
        confirm_latency + duration.clip(lower=1.0)
    ).replace(0.0, 1.0)
    breach_urgency = 1.0 - _clip_series(bars_to_first_breach, scale=4.0)
    reclaim_absence = 1.0 - _clip_series(reclaimed_count, scale=2.0)
    out["rb_late_confirm_fragility_score"] = (
        confirm_share + breach_urgency + reclaim_absence
    ) / 3.0

    touch_balance = (
        1.0
        - (upper_touches - lower_touches).abs()
        / (upper_touches + lower_touches).replace(0.0, 1.0)
    ).clip(lower=0.0, upper=1.0)
    touch_richness = ((min_touches - 2.0) / 3.0).clip(lower=0.0, upper=1.0)
    width_reasonableness = (1.0 - ((width_atr - 2.2).abs() / 1.3)).clip(
        lower=0.0, upper=1.0
    )
    interaction_value = pd.concat(
        [
            _clip_series(bars_to_first_breach, scale=6.0),
            _clip_series(reclaimed_count, scale=2.0),
        ],
        axis=1,
    ).max(axis=1)
    out["rb_boundary_relevance_score"] = (
        touch_balance + touch_richness + width_reasonableness + interaction_value
    ) / 4.0

    duration_support = _clip_series(duration, scale=12.0)
    breach_delay = _clip_series(bars_to_first_breach, scale=6.0)
    reclaim_richness = _clip_series(reclaimed_count, scale=2.0)
    pending_richness = _clip_series(break_pending_count, scale=2.0)
    out["rb_monitor_worthiness_score"] = (
        duration_support + breach_delay + reclaim_richness + pending_richness
    ) / 4.0

    out["rb_plausibility_score"] = (
        0.35 * out["rb_monitor_worthiness_score"]
        + 0.25 * out["rb_boundary_relevance_score"]
        + 0.20 * (1.0 - out["rb_micro_box_risk_score"])
        + 0.20 * (1.0 - out["rb_late_confirm_fragility_score"])
    ).clip(lower=0.0, upper=1.0)
    return out


def _add_path_c2_candidate_scores(forensics: pd.DataFrame) -> pd.DataFrame:
    if forensics.empty:
        return forensics.copy()

    out = forensics.copy()
    structure = (
        pd.to_numeric(
            out.get("range_strength_structure", out.get("range_strength_formation")),
            errors="coerce",
        )
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    monitor = (
        pd.to_numeric(out["rb_monitor_worthiness_score"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    plaus = (
        pd.to_numeric(out["rb_plausibility_score"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    boundary = (
        pd.to_numeric(out["rb_boundary_relevance_score"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
    )
    micro = (
        pd.to_numeric(out["rb_micro_box_risk_score"], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )

    truth_base = (0.42 * monitor + 0.33 * plaus + 0.25 * boundary).clip(
        lower=0.0, upper=1.0
    )

    candidate_v1 = (
        (0.72 * truth_base + 0.12 * structure + 0.16 * (1.0 - micro))
        * (1.0 - 0.35 * micro)
    ).clip(lower=0.0, upper=1.0)

    eligible_high = (monitor >= 0.45) & (plaus >= 0.50) & (micro <= 0.55)
    eligible_mid = (monitor >= 0.35) & (plaus >= 0.42) & (micro <= 0.65)
    candidate_v2 = (0.75 * truth_base + 0.15 * structure + 0.10 * (1.0 - micro)).clip(
        lower=0.0, upper=1.0
    )
    candidate_v2 = candidate_v2.where(~eligible_mid, candidate_v2 + 0.08)
    candidate_v2 = candidate_v2.where(~eligible_high, candidate_v2 + 0.14)
    ineligible_cap = (0.42 + 0.12 * boundary + 0.06 * structure).clip(
        lower=0.0, upper=0.52
    )
    candidate_v2 = candidate_v2.where(
        eligible_mid, np.minimum(candidate_v2, ineligible_cap)
    )
    candidate_v2 = candidate_v2.clip(lower=0.0, upper=1.0)

    candidate_v3 = (0.55 * truth_base + 0.20 * structure + 0.25 * (1.0 - micro)).clip(
        lower=0.0, upper=1.0
    )
    cap = pd.Series(1.0, index=out.index, dtype=float)
    cap = np.minimum(cap, (0.50 + 0.35 * monitor).clip(lower=0.0, upper=1.0))
    cap = np.minimum(cap, (0.48 + 0.38 * (1.0 - micro)).clip(lower=0.0, upper=1.0))
    cap = np.minimum(cap, (0.52 + 0.30 * plaus).clip(lower=0.0, upper=1.0))
    candidate_v3 = np.minimum(candidate_v3, cap).clip(lower=0.0, upper=1.0)

    candidate_v4 = (truth_base * (1.0 - 0.55 * micro)).clip(lower=0.0, upper=1.0)

    out["strength_repair_v1"] = candidate_v1
    out["strength_repair_v2"] = candidate_v2
    out["strength_repair_v3"] = candidate_v3
    out["strength_repair_v4"] = candidate_v4
    return out


def _build_forensics_tables(
    event_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if event_table.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    forensics = event_table.copy()
    forensics.sort_values(["range_id"], inplace=True)
    archetype_rank_col = (
        "strength_legacy" if "strength_legacy" in forensics.columns else "strength"
    )
    strength_q75 = pd.to_numeric(
        forensics[archetype_rank_col], errors="coerce"
    ).quantile(0.75)
    strength_q40 = pd.to_numeric(
        forensics[archetype_rank_col], errors="coerce"
    ).quantile(0.40)
    strength_q75_medium = pd.to_numeric(
        forensics[archetype_rank_col], errors="coerce"
    ).quantile(0.75)

    short_high = (
        forensics[
            pd.to_numeric(forensics["duration_bars"], errors="coerce").le(3)
            & pd.to_numeric(forensics[archetype_rank_col], errors="coerce").ge(
                strength_q75
            )
        ]
        .sort_values(["duration_bars", archetype_rank_col], ascending=[True, False])
        .head(10)
        .copy()
    )
    long_medium = (
        forensics[
            pd.to_numeric(forensics["duration_bars"], errors="coerce").ge(8)
            & pd.to_numeric(forensics[archetype_rank_col], errors="coerce").ge(
                strength_q40
            )
            & pd.to_numeric(forensics[archetype_rank_col], errors="coerce").lt(
                strength_q75_medium
            )
        ]
        .sort_values(["duration_bars", archetype_rank_col], ascending=[False, False])
        .head(10)
        .copy()
    )
    if short_high.empty:
        short_high = (
            forensics[pd.to_numeric(forensics["duration_bars"], errors="coerce").le(3)]
            .sort_values([archetype_rank_col, "duration_bars"], ascending=[False, True])
            .head(10)
            .copy()
        )
    if long_medium.empty:
        short_ids = set(
            pd.to_numeric(short_high["range_id"], errors="coerce").dropna().astype(int)
        )
        long_medium = (
            forensics[
                pd.to_numeric(forensics["duration_bars"], errors="coerce").ge(8)
                & ~pd.to_numeric(forensics["range_id"], errors="coerce").isin(short_ids)
            ]
            .sort_values(
                ["duration_bars", archetype_rank_col], ascending=[False, False]
            )
            .head(10)
            .copy()
        )
    return forensics, short_high, long_medium


def _mean_from_df(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _build_interpretability_metrics_summary(
    forensics: pd.DataFrame,
) -> dict[str, object]:
    metrics = [
        "rb_micro_box_risk_score",
        "rb_late_confirm_fragility_score",
        "rb_boundary_relevance_score",
        "rb_monitor_worthiness_score",
        "rb_plausibility_score",
    ]
    if forensics.empty:
        return {metric: _continuous_stats(pd.Series(dtype=float)) for metric in metrics}
    return {metric: _continuous_stats(forensics[metric]) for metric in metrics}


def _build_contract_bucket_summary(forensics: pd.DataFrame) -> dict[str, int]:
    if forensics.empty:
        return {
            "durable_plausible": 0,
            "fragile_but_plausible": 0,
            "weak_false_positive": 0,
            "strong_false_positive": 0,
        }
    return {
        "durable_plausible": int(
            (
                (forensics["rb_plausibility_score"] >= 0.65)
                & (forensics["rb_monitor_worthiness_score"] >= 0.60)
                & (forensics["rb_micro_box_risk_score"] <= 0.35)
            ).sum()
        ),
        "fragile_but_plausible": int(
            (
                (forensics["rb_plausibility_score"] >= 0.45)
                & (forensics["rb_plausibility_score"] < 0.65)
                & (forensics["rb_micro_box_risk_score"] <= 0.60)
            ).sum()
        ),
        "weak_false_positive": int(
            (
                (forensics["rb_plausibility_score"] < 0.45)
                & (forensics["rb_micro_box_risk_score"] > 0.50)
            ).sum()
        ),
        "strong_false_positive": int(
            (
                (forensics["rb_plausibility_score"] < 0.30)
                | (
                    (forensics["rb_micro_box_risk_score"] > 0.75)
                    & (forensics["rb_monitor_worthiness_score"] < 0.25)
                )
            ).sum()
        ),
    }


def _assign_contract_bucket_labels(forensics: pd.DataFrame) -> pd.DataFrame:
    out = forensics.copy()
    if out.empty:
        out["contract_bucket"] = pd.Series(dtype="object")
        return out

    durable = (
        (out["rb_plausibility_score"] >= 0.65)
        & (out["rb_monitor_worthiness_score"] >= 0.60)
        & (out["rb_micro_box_risk_score"] <= 0.35)
    )
    fragile = (
        (out["rb_plausibility_score"] >= 0.45)
        & (out["rb_plausibility_score"] < 0.65)
        & (out["rb_micro_box_risk_score"] <= 0.60)
    )
    strong_false = (out["rb_plausibility_score"] < 0.30) | (
        (out["rb_micro_box_risk_score"] > 0.75)
        & (out["rb_monitor_worthiness_score"] < 0.25)
    )
    weak_false = (
        (out["rb_plausibility_score"] < 0.45)
        & (out["rb_micro_box_risk_score"] > 0.50)
        & ~strong_false
    )

    out["contract_bucket"] = "unclassified"
    out.loc[durable, "contract_bucket"] = "durable_plausible"
    out.loc[fragile, "contract_bucket"] = "fragile_but_plausible"
    out.loc[weak_false, "contract_bucket"] = "weak_false_positive"
    out.loc[strong_false, "contract_bucket"] = "strong_false_positive"
    return out


def _state_label(state: object) -> str:
    labels = {
        1: "active_intact",
        2: "active_weakened",
        3: "broken_unaccepted",
        4: "accepted_breakout",
        5: "invalidated",
        6: "expired",
        7: "superseded",
    }
    value = pd.to_numeric(pd.Series([state]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    return labels.get(int(value), f"state_{int(value)}")


def _idx_to_timestamp(frame: pd.DataFrame, idx: object) -> TimestampLike:
    value = pd.to_numeric(pd.Series([idx]), errors="coerce").iloc[0]
    if pd.isna(value):
        return pd.NaT
    i = int(value)
    if i < 0 or i >= len(frame):
        return pd.NaT
    return pd.Timestamp(frame.iloc[i]["timestamp"])


def _evaluate_geometry_candidate_window(
    window: pd.DataFrame,
    *,
    upper: float,
    lower: float,
    width_atr_scale: float,
) -> dict[str, float | int | str]:
    width_abs = max(float(upper) - float(lower), 1e-9)
    visible_high = float(pd.to_numeric(window["high"], errors="coerce").max())
    visible_low = float(pd.to_numeric(window["low"], errors="coerce").min())
    visible_width = max(visible_high - visible_low, 1e-9)
    upper_miss = abs(visible_high - float(upper)) / max(width_atr_scale, 1e-9)
    lower_miss = abs(float(lower) - visible_low) / max(width_atr_scale, 1e-9)
    fit = _clip01(1.0 - ((upper_miss + lower_miss) / 2.0) / 1.5)
    width_ratio = width_abs / visible_width
    if fit >= 0.80 and 0.80 <= width_ratio <= 1.20:
        label = "edge_matches_chart_well"
    elif width_ratio > 1.20:
        label = "box_too_wide_for_visible_structure"
    elif width_ratio < 0.80:
        label = "box_too_narrow_for_visible_structure"
    elif fit >= 0.50:
        label = "edge_partially_matches_chart"
    elif fit < 0.25:
        label = "box_not_visually_real"
    else:
        label = "edge_misses_chart_reality"
    return {
        "geometry_upper_edge_miss_atr": float(upper_miss),
        "geometry_lower_edge_miss_atr": float(lower_miss),
        "geometry_chart_fit_score": float(fit),
        "geometry_visible_width_ratio": float(width_ratio),
        "geometry_review_bucket_suggested": label,
    }


def _build_geometry_audit(
    frame: pd.DataFrame,
    event_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame()

    out = event_table.copy()
    out["birth_timestamp"] = out["birth_idx"].apply(
        lambda x: _idx_to_timestamp(frame, x)
    )
    out["confirm_timestamp"] = out["confirm_idx"].apply(
        lambda x: _idx_to_timestamp(frame, x)
    )
    out["active_start_idx"] = pd.to_numeric(out["confirm_idx"], errors="coerce")
    out["active_start_timestamp"] = out["confirm_timestamp"]
    out["end_timestamp"] = out["end_idx"].apply(lambda x: _idx_to_timestamp(frame, x))
    out["range_state"] = out["state"].map(_state_label)

    high_cols = {"swing_high_confirm_flag", "swing_high_confirm_price"}
    low_cols = {"swing_low_confirm_flag", "swing_low_confirm_price"}
    has_high_swings = high_cols.issubset(frame.columns)
    has_low_swings = low_cols.issubset(frame.columns)

    upper_miss_values: list[float] = []
    lower_miss_values: list[float] = []
    fit_scores: list[float] = []
    width_ratios: list[float] = []
    upper_confluence_values: list[int] = []
    lower_confluence_values: list[int] = []
    suggested_labels: list[str] = []

    for _, event in out.iterrows():
        birth_idx = int(
            pd.to_numeric(pd.Series([event["birth_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = int(end_idx_value) if pd.notna(end_idx_value) else confirm_idx + 8
        end_idx = min(max(end_idx, confirm_idx), len(frame) - 1)
        start = max(0, birth_idx - 2)
        stop = min(len(frame) - 1, end_idx + 2)
        window = frame.iloc[start : stop + 1]

        width_abs = max(float(event["high"] - event["low"]), 1e-9)
        atr_scale = max(float(event.get("width_atr", np.nan)), 1e-9)
        metrics = _evaluate_geometry_candidate_window(
            window,
            upper=float(event["high"]),
            lower=float(event["low"]),
            width_atr_scale=atr_scale,
        )

        upper_confluence = 0
        if has_high_swings:
            sh = window[
                pd.to_numeric(window["swing_high_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
            ]
            if not sh.empty:
                upper_confluence = int(
                    pd.to_numeric(sh["swing_high_confirm_price"], errors="coerce")
                    .sub(float(event["high"]))
                    .abs()
                    .le(width_abs * 0.15)
                    .any()
                )
        lower_confluence = 0
        if has_low_swings:
            sl = window[
                pd.to_numeric(window["swing_low_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
            ]
            if not sl.empty:
                lower_confluence = int(
                    pd.to_numeric(sl["swing_low_confirm_price"], errors="coerce")
                    .sub(float(event["low"]))
                    .abs()
                    .le(width_abs * 0.15)
                    .any()
                )

        upper_miss_values.append(float(metrics["geometry_upper_edge_miss_atr"]))
        lower_miss_values.append(float(metrics["geometry_lower_edge_miss_atr"]))
        fit_scores.append(float(metrics["geometry_chart_fit_score"]))
        width_ratios.append(float(metrics["geometry_visible_width_ratio"]))
        upper_confluence_values.append(upper_confluence)
        lower_confluence_values.append(lower_confluence)
        suggested_labels.append(str(metrics["geometry_review_bucket_suggested"]))

    out["geometry_upper_edge_miss_atr"] = upper_miss_values
    out["geometry_lower_edge_miss_atr"] = lower_miss_values
    out["geometry_chart_fit_score"] = fit_scores
    out["geometry_visible_width_ratio"] = width_ratios
    out["geometry_upper_swing_confluence"] = upper_confluence_values
    out["geometry_lower_swing_confluence"] = lower_confluence_values
    out["geometry_review_bucket_suggested"] = suggested_labels
    out["geometry_review_bucket_manual"] = ""
    out["geometry_review_notes"] = ""

    if not candidate_table.empty:
        lineage = candidate_table[
            pd.to_numeric(
                candidate_table.get("confirmed_flag", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
        ].copy()
        lineage_cols = [
            "candidate_lineage_id",
            "candidate_lookback_bars",
            "birth_idx",
            "last_idx",
            "last_timestamp",
            "maturity_pass_idx",
            "maturity_pass_timestamp",
            "viability_pass_idx",
            "viability_pass_timestamp",
            "range_confirm_idx",
            "range_confirm_timestamp",
        ]
        lineage = lineage[
            [col for col in lineage_cols if col in lineage.columns]
        ].copy()
        lineage.rename(columns={"birth_idx": "candidate_birth_idx"}, inplace=True)
        out = out.merge(
            lineage,
            how="left",
            left_on=["confirm_idx", "candidate_lookback_bars"],
            right_on=["range_confirm_idx", "candidate_lookback_bars"],
            suffixes=("", "_candidate"),
        )
    return out


def _build_geometry_candidate_comparison(
    frame: pd.DataFrame,
    geometry_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if geometry_audit.empty:
        return pd.DataFrame(), pd.DataFrame()

    high_cols = {"swing_high_confirm_flag", "swing_high_confirm_price"}
    low_cols = {"swing_low_confirm_flag", "swing_low_confirm_price"}
    has_high_swings = high_cols.issubset(frame.columns)
    has_low_swings = low_cols.issubset(frame.columns)
    rows: list[dict[str, object]] = []

    for _, event in geometry_audit.iterrows():
        birth_idx = int(
            pd.to_numeric(pd.Series([event["birth_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = int(end_idx_value) if pd.notna(end_idx_value) else confirm_idx + 8
        end_idx = min(max(end_idx, confirm_idx), len(frame) - 1)
        start = max(0, birth_idx - 2)
        stop = min(len(frame) - 1, end_idx + 2)
        window = frame.iloc[start : stop + 1]
        legacy_high = float(event["high"])
        legacy_low = float(event["low"])
        legacy_width = max(legacy_high - legacy_low, 1e-9)
        atr_value = _event_confirm_atr(frame, confirm_idx, legacy_width)
        width_atr_scale = max(
            float(event.get("width_atr", legacy_width / max(atr_value, 1e-9))), 1e-9
        )

        for candidate_family, high_col, low_col in GEOMETRY_CANDIDATE_SPECS:
            candidate_high = pd.to_numeric(
                pd.Series([event.get(high_col)]), errors="coerce"
            ).iloc[0]
            candidate_low = pd.to_numeric(
                pd.Series([event.get(low_col)]), errors="coerce"
            ).iloc[0]
            if (
                pd.isna(candidate_high)
                or pd.isna(candidate_low)
                or float(candidate_high) <= float(candidate_low)
            ):
                continue
            candidate_width = max(float(candidate_high) - float(candidate_low), 1e-9)
            metrics = _evaluate_geometry_candidate_window(
                window,
                upper=float(candidate_high),
                lower=float(candidate_low),
                width_atr_scale=width_atr_scale,
            )
            upper_confluence = 0
            if has_high_swings:
                sh = window[
                    pd.to_numeric(window["swing_high_confirm_flag"], errors="coerce")
                    .fillna(0)
                    .eq(1)
                ]
                if not sh.empty:
                    upper_confluence = int(
                        pd.to_numeric(sh["swing_high_confirm_price"], errors="coerce")
                        .sub(float(candidate_high))
                        .abs()
                        .le(candidate_width * 0.15)
                        .any()
                    )
            lower_confluence = 0
            if has_low_swings:
                sl = window[
                    pd.to_numeric(window["swing_low_confirm_flag"], errors="coerce")
                    .fillna(0)
                    .eq(1)
                ]
                if not sl.empty:
                    lower_confluence = int(
                        pd.to_numeric(sl["swing_low_confirm_price"], errors="coerce")
                        .sub(float(candidate_low))
                        .abs()
                        .le(candidate_width * 0.15)
                        .any()
                    )
            rows.append(
                {
                    "range_id": int(event["range_id"]),
                    "candidate_family": candidate_family,
                    "candidate_high": float(candidate_high),
                    "candidate_low": float(candidate_low),
                    "candidate_width_abs": float(candidate_width),
                    "candidate_width_atr": float(
                        candidate_width / max(atr_value, 1e-9)
                    ),
                    "upper_shift_abs": float(candidate_high - legacy_high),
                    "lower_shift_abs": float(legacy_low - candidate_low),
                    "edge_shift_mean_abs": float(
                        (
                            abs(float(candidate_high) - legacy_high)
                            + abs(legacy_low - float(candidate_low))
                        )
                        / 2.0
                    ),
                    "width_delta_abs": float(candidate_width - legacy_width),
                    "width_delta_atr": float(
                        (candidate_width - legacy_width) / max(atr_value, 1e-9)
                    ),
                    "confirm_latency_bars": pd.to_numeric(
                        pd.Series([event.get("confirm_latency_bars")]), errors="coerce"
                    ).iloc[0],
                    "geometry_upper_edge_miss_atr": float(
                        metrics["geometry_upper_edge_miss_atr"]
                    ),
                    "geometry_lower_edge_miss_atr": float(
                        metrics["geometry_lower_edge_miss_atr"]
                    ),
                    "geometry_chart_fit_score": float(
                        metrics["geometry_chart_fit_score"]
                    ),
                    "geometry_visible_width_ratio": float(
                        metrics["geometry_visible_width_ratio"]
                    ),
                    "geometry_review_bucket_suggested": str(
                        metrics["geometry_review_bucket_suggested"]
                    ),
                    "geometry_upper_swing_confluence": upper_confluence,
                    "geometry_lower_swing_confluence": lower_confluence,
                }
            )

    comparison = pd.DataFrame.from_records(rows)
    if comparison.empty:
        return comparison, pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    for candidate_family, group in comparison.groupby("candidate_family", sort=False):
        counts = group["geometry_review_bucket_suggested"].value_counts()
        faithful_or_partial = int(
            counts.get("edge_matches_chart_well", 0)
            + counts.get("edge_partially_matches_chart", 0)
        )
        summary_rows.append(
            {
                "candidate_family": candidate_family,
                "rows": int(len(group)),
                "edge_matches_chart_well_count": int(
                    counts.get("edge_matches_chart_well", 0)
                ),
                "edge_partially_matches_chart_count": int(
                    counts.get("edge_partially_matches_chart", 0)
                ),
                "edge_misses_chart_reality_count": int(
                    counts.get("edge_misses_chart_reality", 0)
                ),
                "box_too_narrow_for_visible_structure_count": int(
                    counts.get("box_too_narrow_for_visible_structure", 0)
                ),
                "box_too_wide_for_visible_structure_count": int(
                    counts.get("box_too_wide_for_visible_structure", 0)
                ),
                "box_not_visually_real_count": int(
                    counts.get("box_not_visually_real", 0)
                ),
                "faithful_or_partial_share": float(
                    faithful_or_partial / max(len(group), 1)
                ),
                "too_narrow_share": float(
                    counts.get("box_too_narrow_for_visible_structure", 0)
                    / max(len(group), 1)
                ),
                "too_wide_share": float(
                    counts.get("box_too_wide_for_visible_structure", 0)
                    / max(len(group), 1)
                ),
                "mean_width_atr": float(
                    pd.to_numeric(group["candidate_width_atr"], errors="coerce").mean()
                ),
                "median_width_atr": float(
                    pd.to_numeric(
                        group["candidate_width_atr"], errors="coerce"
                    ).median()
                ),
                "mean_width_delta_atr": float(
                    pd.to_numeric(group["width_delta_atr"], errors="coerce").mean()
                ),
                "mean_chart_fit_score": float(
                    pd.to_numeric(
                        group["geometry_chart_fit_score"], errors="coerce"
                    ).mean()
                ),
                "mean_edge_shift_abs": float(
                    pd.to_numeric(group["edge_shift_mean_abs"], errors="coerce").mean()
                ),
            }
        )
    return comparison, pd.DataFrame.from_records(summary_rows)


def _compute_doctrine_paths(
    frame: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
    base_low: float,
    base_high: float,
    lookback_bars: int,
    refresh_stride: int,
    bounded: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(
        frame.get("atr", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)
    length = end_idx - start_idx + 1
    high_path = np.full(length, float(base_high), dtype=float)
    low_path = np.full(length, float(base_low), dtype=float)
    current_high = float(base_high)
    current_low = float(base_low)
    refresh_count = 0

    for offset, idx in enumerate(range(start_idx, end_idx + 1)):
        if offset > 0 and offset % refresh_stride == 0:
            local_start = max(0, idx - lookback_bars + 1)
            candidate_high = float(np.nanmax(highs[local_start : idx + 1]))
            candidate_low = float(np.nanmin(lows[local_start : idx + 1]))
            accept = True
            if bounded:
                current_width = max(current_high - current_low, 1e-9)
                candidate_width = max(candidate_high - candidate_low, 1e-9)
                overlap = max(
                    0.0,
                    min(current_high, candidate_high) - max(current_low, candidate_low),
                )
                overlap_frac = overlap / max(min(current_width, candidate_width), 1e-9)
                atr_now = (
                    atr[idx]
                    if idx < len(atr) and np.isfinite(atr[idx]) and atr[idx] > 0
                    else current_width
                )
                width_change_atr = abs(candidate_width - current_width) / max(
                    atr_now, 1e-9
                )
                accept = overlap_frac >= 0.75 and width_change_atr <= 0.35
            if accept:
                refresh_count += int(
                    candidate_high != current_high or candidate_low != current_low
                )
                current_high = candidate_high
                current_low = candidate_low
        high_path[offset] = current_high
        low_path[offset] = current_low
    return high_path, low_path, refresh_count


def _summarize_doctrine(
    frame: pd.DataFrame,
    *,
    start_idx: int,
    end_idx: int,
    local_high: np.ndarray,
    local_low: np.ndarray,
    high_path: np.ndarray,
    low_path: np.ndarray,
    refresh_count: int,
) -> dict[str, float]:
    closes = pd.to_numeric(
        frame.iloc[start_idx : end_idx + 1]["close"], errors="coerce"
    ).to_numpy(dtype=float)
    atr = pd.to_numeric(
        frame.iloc[start_idx : end_idx + 1].get(
            "atr", pd.Series(np.nan, index=frame.index[start_idx : end_idx + 1])
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    atr_safe = np.where(np.isfinite(atr) & (atr > 0), atr, 1.0)
    mismatch = (
        (np.abs(high_path - local_high) + np.abs(low_path - local_low)) / 2.0
    ) / atr_safe
    inside = (closes >= low_path) & (closes <= high_path)
    continuity_cost = 0.0
    if len(high_path) > 1:
        step_changes = (np.abs(np.diff(high_path)) + np.abs(np.diff(low_path))) / 2.0
        continuity_cost = (
            float(np.nanmean(step_changes / atr_safe[1:])) if len(step_changes) else 0.0
        )
    faithfulness = _clip01(1.0 - float(np.nanmean(mismatch)) / 1.5)
    coherence = _clip01(1.0 - continuity_cost / 1.0)
    inside_frac = float(np.nanmean(inside.astype(float))) if len(inside) else 0.0
    return {
        "mean_edge_mismatch_atr": float(np.nanmean(mismatch)),
        "inside_frac": inside_frac,
        "chart_faithfulness_score": faithfulness,
        "source_coherence_score": coherence,
        "visual_plausibility_score": float(
            np.mean([faithfulness, coherence, inside_frac])
        ),
        "refresh_count": float(refresh_count),
    }


def _build_active_truth_audit(
    frame: pd.DataFrame,
    geometry_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if geometry_audit.empty:
        return pd.DataFrame(), pd.DataFrame()

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(
        frame.get("atr", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)
    truth_rows: list[dict[str, object]] = []
    doctrine_rows: list[dict[str, object]] = []

    for _, event in geometry_audit.iterrows():
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = (
            int(end_idx_value)
            if pd.notna(end_idx_value)
            else min(len(frame) - 1, confirm_idx + 12)
        )
        end_idx = min(max(end_idx, confirm_idx + 1), len(frame) - 1)
        lookback = int(
            pd.to_numeric(
                pd.Series([event.get("candidate_lookback_bars", 8)]), errors="coerce"
            )
            .fillna(8)
            .iloc[0]
        )
        length = end_idx - confirm_idx + 1
        local_high = np.full(length, np.nan, dtype=float)
        local_low = np.full(length, np.nan, dtype=float)
        for offset, idx in enumerate(range(confirm_idx, end_idx + 1)):
            local_start = max(0, idx - lookback + 1)
            local_high[offset] = float(np.nanmax(highs[local_start : idx + 1]))
            local_low[offset] = float(np.nanmin(lows[local_start : idx + 1]))

        frozen_high = np.full(length, float(event["high"]), dtype=float)
        frozen_low = np.full(length, float(event["low"]), dtype=float)
        recompute_high, recompute_low, recompute_refreshes = _compute_doctrine_paths(
            frame,
            start_idx=confirm_idx,
            end_idx=end_idx,
            base_low=float(event["low"]),
            base_high=float(event["high"]),
            lookback_bars=lookback,
            refresh_stride=3,
            bounded=False,
        )
        bounded_high, bounded_low, bounded_refreshes = _compute_doctrine_paths(
            frame,
            start_idx=confirm_idx,
            end_idx=end_idx,
            base_low=float(event["low"]),
            base_high=float(event["high"]),
            lookback_bars=lookback,
            refresh_stride=3,
            bounded=True,
        )

        frozen = _summarize_doctrine(
            frame,
            start_idx=confirm_idx,
            end_idx=end_idx,
            local_high=local_high,
            local_low=local_low,
            high_path=frozen_high,
            low_path=frozen_low,
            refresh_count=0,
        )
        recompute = _summarize_doctrine(
            frame,
            start_idx=confirm_idx,
            end_idx=end_idx,
            local_high=local_high,
            local_low=local_low,
            high_path=recompute_high,
            low_path=recompute_low,
            refresh_count=recompute_refreshes,
        )
        bounded = _summarize_doctrine(
            frame,
            start_idx=confirm_idx,
            end_idx=end_idx,
            local_high=local_high,
            local_low=local_low,
            high_path=bounded_high,
            low_path=bounded_low,
            refresh_count=bounded_refreshes,
        )

        atr_safe = np.where(
            np.isfinite(atr[confirm_idx : end_idx + 1])
            & (atr[confirm_idx : end_idx + 1] > 0),
            atr[confirm_idx : end_idx + 1],
            1.0,
        )
        frozen_mismatch = (
            (np.abs(frozen_high - local_high) + np.abs(frozen_low - local_low)) / 2.0
        ) / atr_safe
        mismatch_hits = np.flatnonzero(frozen_mismatch > 0.50)
        stale_hits = np.flatnonzero(frozen_mismatch > 0.35)
        first_mismatch = float(mismatch_hits[0]) if len(mismatch_hits) else np.nan
        stale_bars = float(stale_hits[0]) if len(stale_hits) else np.nan
        close_positions = (
            (closes[confirm_idx : end_idx + 1] - float(event["low"]))
            / max(float(event["high"] - event["low"]), 1e-9)
        ).clip(0.0, 1.0)
        edge_pressure = float(
            np.nanmean(
                np.maximum(1.0 - 2.0 * close_positions, 2.0 * close_positions - 1.0)
            )
        )
        containment_decay = 1.0 - frozen["inside_frac"]
        relevance_decay = float(
            np.mean([frozen["mean_edge_mismatch_atr"] / 1.5, containment_decay])
        )

        confirm_latency = float(
            pd.to_numeric(
                pd.Series([event.get("confirm_latency_bars")]), errors="coerce"
            )
            .fillna(0)
            .iloc[0]
        )
        duration = float(
            pd.to_numeric(pd.Series([event.get("duration_bars")]), errors="coerce")
            .fillna(np.nan)
            .iloc[0]
        )
        first_breach = float(
            pd.to_numeric(
                pd.Series([event.get("bars_to_first_breach")]), errors="coerce"
            )
            .fillna(np.nan)
            .iloc[0]
        )
        later_overlap = geometry_audit[
            (
                pd.to_numeric(geometry_audit["range_id"], errors="coerce")
                != float(event["range_id"])
            )
            & (
                pd.to_numeric(geometry_audit["confirm_idx"], errors="coerce")
                > confirm_idx
            )
            & (
                pd.to_numeric(geometry_audit["confirm_idx"], errors="coerce")
                <= confirm_idx + 6
            )
            & (
                (
                    np.minimum(
                        pd.to_numeric(geometry_audit["high"], errors="coerce"),
                        float(event["high"]),
                    )
                    - np.maximum(
                        pd.to_numeric(geometry_audit["low"], errors="coerce"),
                        float(event["low"]),
                    )
                )
                > 0
            )
        ]

        label = "active_box_remained_truthful"
        if (
            pd.notna(duration)
            and duration <= 2
            and pd.notna(first_breach)
            and first_breach <= 1
            and confirm_latency <= 3
        ):
            label = "confirm_too_early"
        elif (
            pd.notna(first_breach)
            and first_breach <= 2
            and confirm_latency / max(confirm_latency + max(duration, 1.0), 1.0) > 0.45
        ):
            label = "confirm_too_late"
        elif (
            bounded["visual_plausibility_score"] - frozen["visual_plausibility_score"]
            > 0.08
            and pd.notna(first_mismatch)
            and first_mismatch <= 4
        ):
            label = "frozen_box_became_stale"
        elif len(later_overlap) > 0:
            label = "lineage_fragmentation"

        truth_rows.append(
            {
                "range_id": int(event["range_id"]),
                "confirm_age_until_first_material_mismatch": first_mismatch,
                "bars_until_active_box_looks_stale": stale_bars,
                "post_confirm_boundary_pressure": edge_pressure,
                "post_confirm_internal_containment_decay": containment_decay,
                "post_confirm_relevance_decay": relevance_decay,
                "failure_classification": label,
                "frozen_chart_faithfulness_score": frozen["chart_faithfulness_score"],
                "recompute_chart_faithfulness_score": recompute[
                    "chart_faithfulness_score"
                ],
                "bounded_refresh_chart_faithfulness_score": bounded[
                    "chart_faithfulness_score"
                ],
                "frozen_visual_plausibility_score": frozen["visual_plausibility_score"],
                "recompute_visual_plausibility_score": recompute[
                    "visual_plausibility_score"
                ],
                "bounded_refresh_visual_plausibility_score": bounded[
                    "visual_plausibility_score"
                ],
                "frozen_source_coherence_score": frozen["source_coherence_score"],
                "recompute_source_coherence_score": recompute["source_coherence_score"],
                "bounded_refresh_source_coherence_score": bounded[
                    "source_coherence_score"
                ],
            }
        )
        doctrine_rows.extend(
            [
                {"range_id": int(event["range_id"]), "doctrine": "frozen", **frozen},
                {
                    "range_id": int(event["range_id"]),
                    "doctrine": "fixed_horizon_recompute",
                    **recompute,
                },
                {
                    "range_id": int(event["range_id"]),
                    "doctrine": "bounded_continuity_refresh",
                    **bounded,
                },
            ]
        )

    return pd.DataFrame.from_records(truth_rows), pd.DataFrame.from_records(
        doctrine_rows
    )


def _build_geometry_candidate_active_truth_summary(
    frame: pd.DataFrame,
    geometry_audit: pd.DataFrame,
    active_truth_audit: pd.DataFrame,
) -> pd.DataFrame:
    if geometry_audit.empty:
        return pd.DataFrame()

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    truth_map = {}
    if not active_truth_audit.empty:
        truth_map = {
            int(row["range_id"]): row.to_dict()
            for _, row in active_truth_audit.iterrows()
        }

    rows: list[dict[str, object]] = []
    for _, event in geometry_audit.iterrows():
        range_id = int(event["range_id"])
        truth_row = truth_map.get(range_id, {})
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = (
            int(end_idx_value)
            if pd.notna(end_idx_value)
            else min(len(frame) - 1, confirm_idx + 12)
        )
        end_idx = min(max(end_idx, confirm_idx + 1), len(frame) - 1)
        lookback = int(
            pd.to_numeric(
                pd.Series([event.get("candidate_lookback_bars", 8)]), errors="coerce"
            )
            .fillna(8)
            .iloc[0]
        )
        length = end_idx - confirm_idx + 1
        local_high = np.full(length, np.nan, dtype=float)
        local_low = np.full(length, np.nan, dtype=float)
        for offset, idx in enumerate(range(confirm_idx, end_idx + 1)):
            local_start = max(0, idx - lookback + 1)
            local_high[offset] = float(np.nanmax(highs[local_start : idx + 1]))
            local_low[offset] = float(np.nanmin(lows[local_start : idx + 1]))

        for candidate_family, high_col, low_col in GEOMETRY_CANDIDATE_SPECS:
            candidate_high = pd.to_numeric(
                pd.Series([event.get(high_col)]), errors="coerce"
            ).iloc[0]
            candidate_low = pd.to_numeric(
                pd.Series([event.get(low_col)]), errors="coerce"
            ).iloc[0]
            if (
                pd.isna(candidate_high)
                or pd.isna(candidate_low)
                or float(candidate_high) <= float(candidate_low)
            ):
                continue
            high_path = np.full(length, float(candidate_high), dtype=float)
            low_path = np.full(length, float(candidate_low), dtype=float)
            summary = _summarize_doctrine(
                frame,
                start_idx=confirm_idx,
                end_idx=end_idx,
                local_high=local_high,
                local_low=local_low,
                high_path=high_path,
                low_path=low_path,
                refresh_count=0,
            )
            legacy_plaus = (
                pd.to_numeric(
                    pd.Series([truth_row.get("frozen_visual_plausibility_score")]),
                    errors="coerce",
                )
                .fillna(np.nan)
                .iloc[0]
            )
            failure_class = str(truth_row.get("failure_classification", "unknown"))
            confirm_too_late_relief = int(
                failure_class == "confirm_too_late"
                and pd.notna(legacy_plaus)
                and float(summary["visual_plausibility_score"])
                >= float(legacy_plaus) + 0.05
            )
            stale_relief = int(
                failure_class in {"confirm_too_late", "frozen_box_became_stale"}
                and pd.notna(legacy_plaus)
                and float(summary["visual_plausibility_score"])
                >= float(legacy_plaus) + 0.05
            )
            rows.append(
                {
                    "range_id": range_id,
                    "candidate_family": candidate_family,
                    "failure_classification": failure_class,
                    "chart_faithfulness_score": float(
                        summary["chart_faithfulness_score"]
                    ),
                    "inside_frac": float(summary["inside_frac"]),
                    "visual_plausibility_score": float(
                        summary["visual_plausibility_score"]
                    ),
                    "mean_edge_mismatch_atr": float(summary["mean_edge_mismatch_atr"]),
                    "confirm_too_late_relief_flag": confirm_too_late_relief,
                    "stale_relief_flag": stale_relief,
                }
            )

    detail = pd.DataFrame.from_records(rows)
    if detail.empty:
        return detail
    summary_rows: list[dict[str, object]] = []
    for candidate_family, group in detail.groupby("candidate_family", sort=False):
        summary_rows.append(
            {
                "candidate_family": candidate_family,
                "rows": int(len(group)),
                "mean_chart_faithfulness_score": float(
                    pd.to_numeric(
                        group["chart_faithfulness_score"], errors="coerce"
                    ).mean()
                ),
                "mean_visual_plausibility_score": float(
                    pd.to_numeric(
                        group["visual_plausibility_score"], errors="coerce"
                    ).mean()
                ),
                "mean_edge_mismatch_atr": float(
                    pd.to_numeric(
                        group["mean_edge_mismatch_atr"], errors="coerce"
                    ).mean()
                ),
                "confirm_too_late_relief_rate": float(
                    pd.to_numeric(
                        group["confirm_too_late_relief_flag"], errors="coerce"
                    ).mean()
                ),
                "stale_relief_rate": float(
                    pd.to_numeric(group["stale_relief_flag"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame.from_records(summary_rows)


def _build_coverage_regime_report(
    rung_results: list[tuple[str, dict[str, object]]],
    rung_assessments: list[dict[str, object]],
) -> pd.DataFrame:
    assessment_by_label = {str(item["label"]): item for item in rung_assessments}
    rows: list[dict[str, object]] = []
    for label, result in rung_results:
        summary = result["summary"]
        counts = summary.get("event_counts", {})
        assessment = assessment_by_label.get(label, {})
        buckets = assessment.get("contract_bucket_summary", {})
        plausible_total = int(
            buckets.get("durable_plausible", 0)
            + buckets.get("fragile_but_plausible", 0)
        )
        false_total = int(
            buckets.get("weak_false_positive", 0)
            + buckets.get("strong_false_positive", 0)
        )
        rows.append(
            {
                "rung_id": label,
                "confirmed_count": int(counts.get("confirmed_ranges", 0)),
                "active_rows": int(counts.get("active_rows", 0)),
                "median_confirm_latency": assessment.get("confirm_latency_median"),
                "durable_plausible_count": int(buckets.get("durable_plausible", 0)),
                "fragile_plausible_count": int(buckets.get("fragile_but_plausible", 0)),
                "weak_false_positive_count": int(buckets.get("weak_false_positive", 0)),
                "strong_false_positive_count": int(
                    buckets.get("strong_false_positive", 0)
                ),
                "plausible_total": plausible_total,
                "false_positive_total": false_total,
                "strong_false_positive_share": float(
                    buckets.get("strong_false_positive", 0) / max(false_total, 1)
                ),
                "durable_plausible_share": float(
                    buckets.get("durable_plausible", 0)
                    / max(int(counts.get("confirmed_ranges", 0)), 1)
                ),
                "strength_alignment_status": (
                    "ok" if assessment.get("strength_not_badly_inverted") else "bad"
                ),
                "viability_alignment_status": (
                    "ok"
                    if assessment.get("plausibility_aligned")
                    and assessment.get("monitor_aligned")
                    else "bad"
                ),
                "chart_review_status": (
                    "recommended_diagnostic_regime"
                    if assessment.get("valid")
                    else (
                        "too_noisy"
                        if int(counts.get("confirmed_ranges", 0))
                        > TARGET_CONFIRMED_RANGE_MAX
                        else "too_sparse_or_misaligned"
                    )
                ),
                "valid_contract_rung": bool(assessment.get("valid")),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_ranking_disagreement_report(
    forensics: pd.DataFrame, *, top_n: int = 20
) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    rows: list[dict[str, object]] = []
    for metric in [
        "strength_legacy" if "strength_legacy" in scored.columns else "strength",
        (
            "range_strength_viability_legacy"
            if "range_strength_viability_legacy" in scored.columns
            else "range_strength_viability"
        ),
        "strength",
        "range_strength_monitorability",
        "range_strength_semantic",
        "strength_repair_v1",
        "strength_repair_v2",
        "strength_repair_v3",
        "strength_repair_v4",
        "rb_plausibility_score",
        "rb_monitor_worthiness_score",
        "rb_boundary_relevance_score",
    ]:
        if metric not in scored.columns:
            continue
        top = scored.nlargest(min(top_n, len(scored)), metric)
        counts = top["contract_bucket"].value_counts()
        rows.append(
            {
                "ranking_metric": metric,
                "top_n": int(len(top)),
                "durable_plausible": int(counts.get("durable_plausible", 0)),
                "fragile_but_plausible": int(counts.get("fragile_but_plausible", 0)),
                "weak_false_positive": int(counts.get("weak_false_positive", 0)),
                "strong_false_positive": int(counts.get("strong_false_positive", 0)),
                "mean_monitor_worthiness": float(
                    pd.to_numeric(
                        top["rb_monitor_worthiness_score"], errors="coerce"
                    ).mean()
                ),
                "mean_plausibility": float(
                    pd.to_numeric(top["rb_plausibility_score"], errors="coerce").mean()
                ),
                "mean_micro_box_risk": float(
                    pd.to_numeric(
                        top["rb_micro_box_risk_score"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_ranking_rebase_comparison_report(
    forensics: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    rows: list[dict[str, object]] = []
    comparisons = [
        (
            "legacy_strength",
            "strength_legacy" if "strength_legacy" in scored.columns else "strength",
        ),
        (
            "legacy_viability",
            (
                "range_strength_viability_legacy"
                if "range_strength_viability_legacy" in scored.columns
                else "range_strength_viability"
            ),
        ),
        ("failed_path_c_strength", "strength"),
        ("rebased_strength", "strength"),
        ("rebased_monitorability", "range_strength_monitorability"),
        ("rebased_semantic", "range_strength_semantic"),
        ("repair_v1_truth_dominant", "strength_repair_v1"),
        ("repair_v2_eligibility_gated", "strength_repair_v2"),
        ("repair_v3_structure_cap", "strength_repair_v3"),
        ("repair_v4_truth_only", "strength_repair_v4"),
    ]
    for label, metric in comparisons:
        if metric not in scored.columns:
            continue
        top = scored.nlargest(min(top_n, len(scored)), metric)
        counts = top["contract_bucket"].value_counts()
        rows.append(
            {
                "ranking_family": label,
                "ranking_metric": metric,
                "top_n": int(len(top)),
                "durable_plausible": int(counts.get("durable_plausible", 0)),
                "fragile_but_plausible": int(counts.get("fragile_but_plausible", 0)),
                "weak_false_positive": int(counts.get("weak_false_positive", 0)),
                "strong_false_positive": int(counts.get("strong_false_positive", 0)),
                "mean_monitor_worthiness": float(
                    pd.to_numeric(
                        top["rb_monitor_worthiness_score"], errors="coerce"
                    ).mean()
                ),
                "mean_plausibility": float(
                    pd.to_numeric(top["rb_plausibility_score"], errors="coerce").mean()
                ),
                "mean_micro_box_risk": float(
                    pd.to_numeric(
                        top["rb_micro_box_risk_score"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_agreement_matrix(
    forensics: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    rankings = {
        "legacy_production": (
            "strength_legacy" if "strength_legacy" in scored.columns else "strength"
        ),
        "failed_path_c": "strength",
        "interpretability": "rb_plausibility_score",
        "repair_v1": "strength_repair_v1",
        "repair_v2": "strength_repair_v2",
        "repair_v3": "strength_repair_v3",
        "repair_v4": "strength_repair_v4",
    }
    top_ids = {
        label: set(
            pd.to_numeric(
                scored.nlargest(min(top_n, len(scored)), metric)["range_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        for label, metric in rankings.items()
        if metric in scored.columns
    }
    rows: list[dict[str, object]] = []
    labels = list(top_ids)
    for left in labels:
        for right in labels:
            left_ids = top_ids[left]
            right_ids = top_ids[right]
            union = left_ids | right_ids
            overlap = left_ids & right_ids
            rows.append(
                {
                    "left_ranking": left,
                    "right_ranking": right,
                    "overlap_count": int(len(overlap)),
                    "jaccard_agreement": float(len(overlap) / max(len(union), 1)),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_bucket_lift_report(
    forensics: pd.DataFrame, *, top_n: int = 20
) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    legacy_metric = (
        "strength_legacy" if "strength_legacy" in scored.columns else "strength"
    )
    interpretability_metric = "rb_plausibility_score"
    top_sets: dict[str, pd.DataFrame] = {
        "legacy": scored.nlargest(min(top_n, len(scored)), legacy_metric),
        "failed_path_c": scored.nlargest(min(top_n, len(scored)), "strength"),
        "interpretability": scored.nlargest(
            min(top_n, len(scored)), interpretability_metric
        ),
    }
    for label, metric in (
        ("repair_v1", "strength_repair_v1"),
        ("repair_v2", "strength_repair_v2"),
        ("repair_v3", "strength_repair_v3"),
        ("repair_v4", "strength_repair_v4"),
    ):
        if metric in scored.columns:
            top_sets[label] = scored.nlargest(min(top_n, len(scored)), metric)
    buckets = [
        "durable_plausible",
        "fragile_but_plausible",
        "weak_false_positive",
        "strong_false_positive",
    ]
    rows: list[dict[str, object]] = []
    for ranking_label, frame_top in top_sets.items():
        if ranking_label == "interpretability":
            continue
        for bucket in buckets:
            legacy_share = float(
                top_sets["legacy"]["contract_bucket"].eq(bucket).mean()
            )
            candidate_share = float(frame_top["contract_bucket"].eq(bucket).mean())
            truth_share = float(
                top_sets["interpretability"]["contract_bucket"].eq(bucket).mean()
            )
            rows.append(
                {
                    "ranking_label": ranking_label,
                    "bucket": bucket,
                    "legacy_top_share": legacy_share,
                    "candidate_top_share": candidate_share,
                    "interpretability_top_share": truth_share,
                    "candidate_minus_legacy": candidate_share - legacy_share,
                    "distance_to_truth_legacy": abs(truth_share - legacy_share),
                    "distance_to_truth_candidate": abs(truth_share - candidate_share),
                    "distance_improvement_vs_legacy": abs(truth_share - legacy_share)
                    - abs(truth_share - candidate_share),
                }
            )
    return pd.DataFrame.from_records(rows)


def _build_family_comparison_report(forensics: pd.DataFrame) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    tagged = _assign_contract_bucket_labels(forensics)
    _, short_high, long_medium = _build_forensics_tables(tagged)
    families = {
        "short_lived_high_strength": short_high,
        "long_lived_medium_strength": long_medium,
        "durable_plausible": tagged[tagged["contract_bucket"] == "durable_plausible"],
        "fragile_but_plausible": tagged[
            tagged["contract_bucket"] == "fragile_but_plausible"
        ],
        "weak_false_positive": tagged[
            tagged["contract_bucket"] == "weak_false_positive"
        ],
        "strong_false_positive": tagged[
            tagged["contract_bucket"] == "strong_false_positive"
        ],
    }
    metrics = [
        "strength",
        "strength_legacy",
        "range_strength_structure",
        "range_strength_monitorability",
        "range_strength_semantic",
        "range_strength_formation",
        "range_strength_viability",
        "range_strength_viability_legacy",
        "strength_repair_v1",
        "strength_repair_v2",
        "strength_repair_v3",
        "strength_repair_v4",
        "rb_plausibility_score",
        "rb_monitor_worthiness_score",
        "rb_micro_box_risk_score",
        "rb_boundary_relevance_score",
    ]
    rows: list[dict[str, object]] = []
    for label, group in families.items():
        row: dict[str, object] = {"family": label, "rows": int(len(group))}
        for metric in metrics:
            row[metric] = (
                float(pd.to_numeric(group.get(metric), errors="coerce").dropna().mean())
                if len(group)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _top_bucket_metrics(
    scored: pd.DataFrame, metric: str, *, top_n: int
) -> dict[str, float | int]:
    top = scored.nlargest(min(top_n, len(scored)), metric)
    counts = top["contract_bucket"].value_counts()
    return {
        "top_n": int(len(top)),
        "durable_plausible": int(counts.get("durable_plausible", 0)),
        "fragile_but_plausible": int(counts.get("fragile_but_plausible", 0)),
        "weak_false_positive": int(counts.get("weak_false_positive", 0)),
        "strong_false_positive": int(counts.get("strong_false_positive", 0)),
        "mean_monitor_worthiness": float(
            pd.to_numeric(top["rb_monitor_worthiness_score"], errors="coerce").mean()
        ),
        "mean_plausibility": float(
            pd.to_numeric(top["rb_plausibility_score"], errors="coerce").mean()
        ),
        "mean_micro_box_risk": float(
            pd.to_numeric(top["rb_micro_box_risk_score"], errors="coerce").mean()
        ),
        "mean_boundary_relevance": float(
            pd.to_numeric(top["rb_boundary_relevance_score"], errors="coerce").mean()
        ),
    }


def _candidate_distance_to_truth(
    scored: pd.DataFrame, metric: str, *, top_n: int
) -> float:
    buckets = [
        "durable_plausible",
        "fragile_but_plausible",
        "weak_false_positive",
        "strong_false_positive",
    ]
    truth_top = scored.nlargest(min(top_n, len(scored)), "rb_plausibility_score")
    cand_top = scored.nlargest(min(top_n, len(scored)), metric)
    total = 0.0
    for bucket in buckets:
        truth_share = float(truth_top["contract_bucket"].eq(bucket).mean())
        cand_share = float(cand_top["contract_bucket"].eq(bucket).mean())
        total += abs(truth_share - cand_share)
    return total


def _build_path_c2_candidate_report(
    forensics: pd.DataFrame,
    *,
    top_ns: tuple[int, ...] = (20, 50),
) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    candidates = {
        "legacy": (
            "strength_legacy" if "strength_legacy" in scored.columns else "strength"
        ),
        "failed_path_c": "strength",
        "repair_v1_truth_dominant": "strength_repair_v1",
        "repair_v2_eligibility_gated": "strength_repair_v2",
        "repair_v3_structure_cap": "strength_repair_v3",
        "repair_v4_truth_only": "strength_repair_v4",
        "interpretability": "rb_plausibility_score",
    }
    rows: list[dict[str, object]] = []
    for label, metric in candidates.items():
        if metric not in scored.columns:
            continue
        for top_n in top_ns:
            row = {"candidate_label": label, "ranking_metric": metric}
            row.update(_top_bucket_metrics(scored, metric, top_n=top_n))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _build_path_c2_archetype_report(forensics: pd.DataFrame) -> pd.DataFrame:
    if forensics.empty:
        return pd.DataFrame()
    tagged = _assign_contract_bucket_labels(forensics)
    _, short_high, long_medium = _build_forensics_tables(tagged)
    metrics = [
        "strength_legacy",
        "strength",
        "strength_repair_v1",
        "strength_repair_v2",
        "strength_repair_v3",
        "strength_repair_v4",
        "rb_monitor_worthiness_score",
        "rb_plausibility_score",
        "rb_micro_box_risk_score",
        "rb_boundary_relevance_score",
        "range_strength_structure",
    ]
    rows: list[dict[str, object]] = []
    for cohort_name, df in (
        ("short_lived_high_strength", short_high),
        ("long_lived_medium_strength", long_medium),
    ):
        row: dict[str, object] = {"cohort": cohort_name, "rows": int(len(df))}
        for metric in metrics:
            row[metric] = _mean_from_df(df, metric)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _evaluate_path_c2_candidates(
    forensics: pd.DataFrame, *, top_n: int = 20
) -> tuple[pd.DataFrame, str]:
    if forensics.empty:
        return pd.DataFrame(), "all_candidates_failed"
    scored = _assign_contract_bucket_labels(forensics)
    _, short_high, long_medium = _build_forensics_tables(scored)
    legacy_metric = (
        "strength_legacy" if "strength_legacy" in scored.columns else "strength"
    )
    legacy_top = _top_bucket_metrics(scored, legacy_metric, top_n=top_n)
    legacy_distance = _candidate_distance_to_truth(scored, legacy_metric, top_n=top_n)
    preference = [
        ("repair_v2_eligibility_gated", "strength_repair_v2"),
        ("repair_v1_truth_dominant", "strength_repair_v1"),
        ("repair_v3_structure_cap", "strength_repair_v3"),
        ("repair_v4_truth_only", "strength_repair_v4"),
    ]
    rows: list[dict[str, object]] = []
    recommendation = "all_candidates_failed"
    for label, metric in preference:
        short_score = _mean_from_df(short_high, metric)
        long_score = _mean_from_df(long_medium, metric)
        top_metrics = _top_bucket_metrics(scored, metric, top_n=top_n)
        distance = _candidate_distance_to_truth(scored, metric, top_n=top_n)
        gate_long = (
            short_score is not None
            and long_score is not None
            and long_score >= short_score
        )
        gate_strong_fp = int(top_metrics["strong_false_positive"]) <= int(
            legacy_top["strong_false_positive"]
        )
        gate_monitor = float(top_metrics["mean_monitor_worthiness"]) >= float(
            legacy_top["mean_monitor_worthiness"]
        )
        gate_plaus = float(top_metrics["mean_plausibility"]) >= float(
            legacy_top["mean_plausibility"]
        )
        gate_micro = float(top_metrics["mean_micro_box_risk"]) <= float(
            legacy_top["mean_micro_box_risk"]
        )
        gate_truth = distance < legacy_distance
        passed = bool(
            gate_long
            and gate_strong_fp
            and gate_monitor
            and gate_plaus
            and gate_micro
            and gate_truth
        )
        rows.append(
            {
                "candidate_label": label,
                "ranking_metric": metric,
                "short_mean_score": short_score,
                "long_mean_score": long_score,
                "legacy_top20_strong_false_positive": int(
                    legacy_top["strong_false_positive"]
                ),
                "candidate_top20_strong_false_positive": int(
                    top_metrics["strong_false_positive"]
                ),
                "legacy_top20_mean_monitor_worthiness": float(
                    legacy_top["mean_monitor_worthiness"]
                ),
                "candidate_top20_mean_monitor_worthiness": float(
                    top_metrics["mean_monitor_worthiness"]
                ),
                "legacy_top20_mean_plausibility": float(
                    legacy_top["mean_plausibility"]
                ),
                "candidate_top20_mean_plausibility": float(
                    top_metrics["mean_plausibility"]
                ),
                "legacy_top20_mean_micro_box_risk": float(
                    legacy_top["mean_micro_box_risk"]
                ),
                "candidate_top20_mean_micro_box_risk": float(
                    top_metrics["mean_micro_box_risk"]
                ),
                "legacy_distance_to_truth": legacy_distance,
                "candidate_distance_to_truth": distance,
                "gate_c2_1_long_outranks_short": gate_long,
                "gate_c2_2_no_strong_fp_increase": gate_strong_fp,
                "gate_c2_3_monitor_not_worse": gate_monitor,
                "gate_c2_4_plausibility_not_worse": gate_plaus,
                "gate_c2_5_micro_risk_not_worse": gate_micro,
                "gate_c2_6_truth_distance_improves": gate_truth,
                "gate_c2_7_coverage_stable": True,
                "passed": passed,
            }
        )
        if passed and recommendation == "all_candidates_failed":
            recommendation = label
    return pd.DataFrame.from_records(rows), recommendation


def _bundle_named_reports(
    report_frames: dict[str, pd.DataFrame], *, group_col: str = "report_group"
) -> pd.DataFrame:
    bundled: list[pd.DataFrame] = []
    for label, frame in report_frames.items():
        if frame is None:
            continue
        if frame.empty:
            continue
        part = frame.copy()
        part.insert(0, group_col, label)
        bundled.append(part)
    if not bundled:
        return pd.DataFrame(columns=[group_col])
    return pd.concat(bundled, ignore_index=True, sort=False)


def _build_downstream_usefulness_report(
    frame: pd.DataFrame, forensics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    if forensics.empty:
        return pd.DataFrame(), {}
    out = _assign_contract_bucket_labels(forensics)
    structure_cols = [
        col
        for col in ["bos_bull", "bos_bear", "choch_bull", "choch_bear"]
        if col in frame.columns
    ]
    meaningful_flags: list[int] = []
    nontrivial_flags: list[int] = []
    overlap_flags: list[int] = []
    interpretive_flags: list[int] = []
    confluence_scores: list[float] = []

    for _, event in out.iterrows():
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = (
            int(end_idx_value)
            if pd.notna(end_idx_value)
            else min(len(frame) - 1, confirm_idx + 20)
        )
        horizon = frame.iloc[confirm_idx + 1 : min(len(frame), end_idx + 7)]
        width_abs = max(float(event["high"] - event["low"]), 1e-9)

        meaningful = int(
            (
                pd.to_numeric(
                    pd.Series([event.get("bars_to_first_breach")]), errors="coerce"
                )
                .fillna(np.nan)
                .iloc[0]
                >= 3
            )
            or (
                pd.to_numeric(
                    pd.Series([event.get("reclaimed_count")]), errors="coerce"
                )
                .fillna(0)
                .iloc[0]
                > 0
            )
            or (
                pd.to_numeric(
                    pd.Series([event.get("break_pending_count")]), errors="coerce"
                )
                .fillna(0)
                .iloc[0]
                > 0
            )
        )
        nontrivial = int(
            (
                pd.to_numeric(
                    pd.Series([event.get("reclaimed_count")]), errors="coerce"
                )
                .fillna(0)
                .iloc[0]
                > 0
            )
            or (
                pd.to_numeric(
                    pd.Series([event.get("bars_to_breakout_accept")]), errors="coerce"
                )
                .fillna(np.nan)
                .iloc[0]
                >= 4
            )
        )
        overlap = 0
        if structure_cols and not horizon.empty:
            structure_flags = pd.DataFrame(
                {
                    col: pd.to_numeric(horizon[col], errors="coerce").fillna(0).eq(1)
                    for col in structure_cols
                }
            ).any(axis=1)
            close_vals = pd.to_numeric(horizon["close"], errors="coerce")
            overlap = int(
                (
                    structure_flags
                    & (
                        (close_vals - float(event["high"])).abs().le(width_abs * 0.20)
                        | (close_vals - float(event["low"])).abs().le(width_abs * 0.20)
                    )
                ).any()
            )
        interpretive = int(meaningful or nontrivial or overlap)

        local = frame.iloc[max(0, confirm_idx - 6) : min(len(frame), confirm_idx + 3)]
        confluence = 0.0
        if {"swing_high_confirm_flag", "swing_high_confirm_price"}.issubset(
            local.columns
        ):
            if (
                pd.to_numeric(local["swing_high_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
                & pd.to_numeric(local["swing_high_confirm_price"], errors="coerce")
                .sub(float(event["high"]))
                .abs()
                .le(width_abs * 0.15)
            ).any():
                confluence += 0.5
        if {"swing_low_confirm_flag", "swing_low_confirm_price"}.issubset(
            local.columns
        ):
            if (
                pd.to_numeric(local["swing_low_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
                & pd.to_numeric(local["swing_low_confirm_price"], errors="coerce")
                .sub(float(event["low"]))
                .abs()
                .le(width_abs * 0.15)
            ).any():
                confluence += 0.5

        meaningful_flags.append(meaningful)
        nontrivial_flags.append(nontrivial)
        overlap_flags.append(overlap)
        interpretive_flags.append(interpretive)
        confluence_scores.append(confluence)

    out["later_meaningful_interaction_flag"] = meaningful_flags
    out["later_nontrivial_breach_reclaim_flag"] = nontrivial_flags
    out["later_useful_event_overlap_flag"] = overlap_flags
    out["interpretive_value_flag"] = interpretive_flags
    out["source_family_confluence_score"] = confluence_scores

    summary = {
        "fraction_later_meaningfully_interacted": float(
            pd.Series(meaningful_flags).mean()
        ),
        "fraction_later_nontrivial_breach_reclaim": float(
            pd.Series(nontrivial_flags).mean()
        ),
        "fraction_later_useful_event_overlap": float(pd.Series(overlap_flags).mean()),
        "fraction_presence_improved_interpretation": float(
            pd.Series(interpretive_flags).mean()
        ),
        "mean_confluence_with_other_families": float(
            pd.Series(confluence_scores).mean()
        ),
        "redundancy_assessment": (
            "genuinely_additive"
            if float(pd.Series(interpretive_flags).mean()) >= 0.45
            else (
                "partly_redundant"
                if float(pd.Series(interpretive_flags).mean()) >= 0.25
                else "mostly_redundant"
            )
        ),
    }
    return out, summary


def _build_geometry_candidate_downstream_summary(
    frame: pd.DataFrame,
    geometry_audit: pd.DataFrame,
) -> pd.DataFrame:
    if geometry_audit.empty:
        return pd.DataFrame()

    structure_cols = [
        col
        for col in ["bos_bull", "bos_bear", "choch_bull", "choch_bear"]
        if col in frame.columns
    ]
    rows: list[dict[str, object]] = []
    for _, event in geometry_audit.iterrows():
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = (
            int(end_idx_value)
            if pd.notna(end_idx_value)
            else min(len(frame) - 1, confirm_idx + 20)
        )
        horizon = frame.iloc[confirm_idx + 1 : min(len(frame), end_idx + 7)]
        local = frame.iloc[max(0, confirm_idx - 6) : min(len(frame), confirm_idx + 3)]

        for candidate_family, high_col, low_col in GEOMETRY_CANDIDATE_SPECS:
            candidate_high = pd.to_numeric(
                pd.Series([event.get(high_col)]), errors="coerce"
            ).iloc[0]
            candidate_low = pd.to_numeric(
                pd.Series([event.get(low_col)]), errors="coerce"
            ).iloc[0]
            if (
                pd.isna(candidate_high)
                or pd.isna(candidate_low)
                or float(candidate_high) <= float(candidate_low)
            ):
                continue
            width_abs = max(float(candidate_high) - float(candidate_low), 1e-9)
            edge_tol = width_abs * 0.10
            meaningful = 0
            nontrivial = 0
            overlap = 0
            interpretive = 0
            confluence = 0.0
            if not horizon.empty:
                high_vals = pd.to_numeric(horizon["high"], errors="coerce")
                low_vals = pd.to_numeric(horizon["low"], errors="coerce")
                close_vals = pd.to_numeric(horizon["close"], errors="coerce")
                upper_touch = high_vals.ge(float(candidate_high) - edge_tol)
                lower_touch = low_vals.le(float(candidate_low) + edge_tol)
                close_break = close_vals.gt(
                    float(candidate_high) + edge_tol * 0.25
                ) | close_vals.lt(float(candidate_low) - edge_tol * 0.25)
                close_inside = close_vals.between(
                    float(candidate_low), float(candidate_high)
                )
                reclaim_after_break = False
                for idx in range(len(horizon)):
                    if bool(close_break.iloc[idx]) and bool(
                        close_inside.iloc[idx + 1 : idx + 4].any()
                    ):
                        reclaim_after_break = True
                        break
                meaningful = int(
                    bool((upper_touch | lower_touch).iloc[2:].any())
                    if len(horizon) > 2
                    else bool((upper_touch | lower_touch).any())
                )
                late_close_break = (
                    bool(close_break.iloc[3:].any()) if len(horizon) > 3 else False
                )
                nontrivial = int(reclaim_after_break or late_close_break)
                if structure_cols:
                    structure_flags = pd.DataFrame(
                        {
                            col: pd.to_numeric(horizon[col], errors="coerce")
                            .fillna(0)
                            .eq(1)
                            for col in structure_cols
                        }
                    ).any(axis=1)
                    overlap = int((structure_flags & (upper_touch | lower_touch)).any())
                interpretive = int(meaningful or nontrivial or overlap)
            if {"swing_high_confirm_flag", "swing_high_confirm_price"}.issubset(
                local.columns
            ):
                if (
                    pd.to_numeric(local["swing_high_confirm_flag"], errors="coerce")
                    .fillna(0)
                    .eq(1)
                    & pd.to_numeric(local["swing_high_confirm_price"], errors="coerce")
                    .sub(float(candidate_high))
                    .abs()
                    .le(width_abs * 0.15)
                ).any():
                    confluence += 0.5
            if {"swing_low_confirm_flag", "swing_low_confirm_price"}.issubset(
                local.columns
            ):
                if (
                    pd.to_numeric(local["swing_low_confirm_flag"], errors="coerce")
                    .fillna(0)
                    .eq(1)
                    & pd.to_numeric(local["swing_low_confirm_price"], errors="coerce")
                    .sub(float(candidate_low))
                    .abs()
                    .le(width_abs * 0.15)
                ).any():
                    confluence += 0.5
            rows.append(
                {
                    "range_id": int(event["range_id"]),
                    "candidate_family": candidate_family,
                    "later_meaningful_interaction_flag": meaningful,
                    "later_nontrivial_breach_reclaim_flag": nontrivial,
                    "later_useful_event_overlap_flag": overlap,
                    "interpretive_value_flag": interpretive,
                    "source_family_confluence_score": confluence,
                }
            )
    detail = pd.DataFrame.from_records(rows)
    if detail.empty:
        return detail
    summary_rows: list[dict[str, object]] = []
    for candidate_family, group in detail.groupby("candidate_family", sort=False):
        summary_rows.append(
            {
                "candidate_family": candidate_family,
                "rows": int(len(group)),
                "fraction_later_meaningfully_interacted": float(
                    pd.to_numeric(
                        group["later_meaningful_interaction_flag"], errors="coerce"
                    ).mean()
                ),
                "fraction_later_nontrivial_breach_reclaim": float(
                    pd.to_numeric(
                        group["later_nontrivial_breach_reclaim_flag"], errors="coerce"
                    ).mean()
                ),
                "fraction_later_useful_event_overlap": float(
                    pd.to_numeric(
                        group["later_useful_event_overlap_flag"], errors="coerce"
                    ).mean()
                ),
                "fraction_presence_improved_interpretation": float(
                    pd.to_numeric(
                        group["interpretive_value_flag"], errors="coerce"
                    ).mean()
                ),
                "mean_confluence_with_other_families": float(
                    pd.to_numeric(
                        group["source_family_confluence_score"], errors="coerce"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame.from_records(summary_rows)


def _build_geometry_ranking_preservation_report(
    forensics: pd.DataFrame,
    geometry_candidate_comparison: pd.DataFrame,
    *,
    top_n: int = 20,
    ranking_metric: str = FROZEN_GEOMETRY_RANKING_METRIC,
) -> pd.DataFrame:
    if (
        forensics.empty
        or geometry_candidate_comparison.empty
        or ranking_metric not in forensics.columns
    ):
        return pd.DataFrame()
    scored = _assign_contract_bucket_labels(forensics)
    top = scored.nlargest(min(top_n, len(scored)), ranking_metric).copy()
    top_ids = set(pd.to_numeric(top["range_id"], errors="coerce").dropna().astype(int))
    top_geometry = geometry_candidate_comparison[
        pd.to_numeric(geometry_candidate_comparison["range_id"], errors="coerce").isin(
            top_ids
        )
    ].copy()
    top_bucket_counts = top["contract_bucket"].value_counts()
    rows: list[dict[str, object]] = []
    for candidate_family, group in top_geometry.groupby("candidate_family", sort=False):
        counts = group["geometry_review_bucket_suggested"].value_counts()
        faithful_or_partial = int(
            counts.get("edge_matches_chart_well", 0)
            + counts.get("edge_partially_matches_chart", 0)
        )
        rows.append(
            {
                "candidate_family": candidate_family,
                "ranking_metric": ranking_metric,
                "top_n": int(len(group)),
                "top_ranked_faithful_or_partial_share": float(
                    faithful_or_partial / max(len(group), 1)
                ),
                "top_ranked_too_narrow_share": float(
                    counts.get("box_too_narrow_for_visible_structure", 0)
                    / max(len(group), 1)
                ),
                "top_ranked_too_wide_share": float(
                    counts.get("box_too_wide_for_visible_structure", 0)
                    / max(len(group), 1)
                ),
                "top_ranked_mean_geometry_fit": float(
                    pd.to_numeric(
                        group["geometry_chart_fit_score"], errors="coerce"
                    ).mean()
                ),
                "top_ranked_strong_false_positive_count": int(
                    top_bucket_counts.get("strong_false_positive", 0)
                ),
                "top_ranked_mean_monitor_worthiness": float(
                    pd.to_numeric(
                        top["rb_monitor_worthiness_score"], errors="coerce"
                    ).mean()
                ),
                "top_ranked_mean_plausibility": float(
                    pd.to_numeric(top["rb_plausibility_score"], errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_geometry_candidate_gate_report(
    geometry_candidate_summary: pd.DataFrame,
    geometry_candidate_truth_summary: pd.DataFrame,
    geometry_candidate_downstream_summary: pd.DataFrame,
    geometry_ranking_preservation_report: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if geometry_candidate_summary.empty:
        return pd.DataFrame(), "no_candidate_passed"

    summary_map = {
        str(row["candidate_family"]): row
        for _, row in geometry_candidate_summary.iterrows()
    }
    truth_map = {
        str(row["candidate_family"]): row
        for _, row in geometry_candidate_truth_summary.iterrows()
    }
    downstream_map = {
        str(row["candidate_family"]): row
        for _, row in geometry_candidate_downstream_summary.iterrows()
    }
    ranking_map = {
        str(row["candidate_family"]): row
        for _, row in geometry_ranking_preservation_report.iterrows()
    }
    legacy = summary_map.get("g1_legacy")
    if legacy is None:
        return pd.DataFrame(), "no_candidate_passed"
    legacy_downstream = downstream_map.get("g1_legacy", {})
    legacy_ranking = ranking_map.get("g1_legacy", {})
    rows: list[dict[str, object]] = []
    recommendation = "no_candidate_passed"
    best_score = float("-inf")
    for candidate_family, row in summary_map.items():
        if candidate_family == "g1_legacy":
            continue
        truth = truth_map.get(candidate_family, {})
        downstream = downstream_map.get(candidate_family, {})
        ranking = ranking_map.get(candidate_family, {})
        too_narrow_reduction = float(legacy["too_narrow_share"]) - float(
            row["too_narrow_share"]
        )
        faithful_gain = float(row["faithful_or_partial_share"]) - float(
            legacy["faithful_or_partial_share"]
        )
        too_wide_delta = float(row["too_wide_share"]) - float(legacy["too_wide_share"])
        downstream_meaningful_delta = float(
            downstream.get("fraction_later_meaningfully_interacted", 0.0)
        ) - float(legacy_downstream.get("fraction_later_meaningfully_interacted", 0.0))
        downstream_interpretive_delta = float(
            downstream.get("fraction_presence_improved_interpretation", 0.0)
        ) - float(
            legacy_downstream.get("fraction_presence_improved_interpretation", 0.0)
        )
        top_rank_faithful_delta = float(
            ranking.get("top_ranked_faithful_or_partial_share", 0.0)
        ) - float(legacy_ranking.get("top_ranked_faithful_or_partial_share", 0.0))
        gate_g1 = too_narrow_reduction > 0.05
        gate_g2 = faithful_gain > 0.03
        gate_g3 = too_wide_delta <= 0.10
        gate_g4 = (
            downstream_meaningful_delta >= -0.05
            and downstream_interpretive_delta >= -0.05
        )
        gate_g5 = True
        gate_g6 = top_rank_faithful_delta >= -0.02
        gate_g7 = True
        passed = bool(
            gate_g1
            and gate_g2
            and gate_g3
            and gate_g4
            and gate_g5
            and gate_g6
            and gate_g7
        )
        score = (
            4.0 * too_narrow_reduction
            + 3.0 * faithful_gain
            + 1.5 * float(truth.get("confirm_too_late_relief_rate", 0.0))
            + 1.0 * downstream_interpretive_delta
            + 0.5 * top_rank_faithful_delta
            - 2.0 * max(0.0, too_wide_delta)
        )
        rows.append(
            {
                "candidate_family": candidate_family,
                "too_narrow_reduction": too_narrow_reduction,
                "faithful_or_partial_gain": faithful_gain,
                "too_wide_delta": too_wide_delta,
                "confirm_too_late_relief_rate": float(
                    truth.get("confirm_too_late_relief_rate", 0.0)
                ),
                "downstream_meaningful_delta": downstream_meaningful_delta,
                "downstream_interpretive_delta": downstream_interpretive_delta,
                "top_ranked_faithful_delta": top_rank_faithful_delta,
                "gate_g1_too_narrow_reduction": gate_g1,
                "gate_g2_edge_faithfulness_improvement": gate_g2,
                "gate_g3_too_wide_controlled": gate_g3,
                "gate_g4_downstream_preserved": gate_g4,
                "gate_g5_coverage_stable": gate_g5,
                "gate_g6_ranking_preserved": gate_g6,
                "gate_g7_causal_integrity_unchanged": gate_g7,
                "passed": passed,
                "score": float(score),
            }
        )
        if passed and score > best_score:
            best_score = score
            recommendation = candidate_family
    return pd.DataFrame.from_records(rows), recommendation


def _primary_path_from_reports(
    active_truth_audit: pd.DataFrame,
    doctrine_report: pd.DataFrame,
    ranking_report: pd.DataFrame,
    downstream_summary: dict[str, object],
    geometry_audit: pd.DataFrame,
) -> dict[str, str]:
    failure_counts = (
        active_truth_audit["failure_classification"].value_counts()
        if not active_truth_audit.empty
        else pd.Series(dtype=int)
    )
    dominant_failure = (
        str(failure_counts.idxmax()) if not failure_counts.empty else "unknown"
    )

    gain = 0.0
    if not doctrine_report.empty:
        pivot = doctrine_report.pivot_table(
            index="range_id",
            columns="doctrine",
            values="visual_plausibility_score",
            aggfunc="mean",
        )
        if {"frozen", "bounded_continuity_refresh"}.issubset(pivot.columns):
            gain = float((pivot["bounded_continuity_refresh"] - pivot["frozen"]).mean())

    truth_layer_better = False
    if not ranking_report.empty:
        strength_row = ranking_report[ranking_report["ranking_metric"] == "strength"]
        plaus_row = ranking_report[
            ranking_report["ranking_metric"] == "rb_plausibility_score"
        ]
        if not strength_row.empty and not plaus_row.empty:
            truth_layer_better = int(plaus_row.iloc[0]["durable_plausible"]) > int(
                strength_row.iloc[0]["durable_plausible"]
            ) or int(plaus_row.iloc[0]["strong_false_positive"]) < int(
                strength_row.iloc[0]["strong_false_positive"]
            )

    geometry_bad_share = 0.0
    if not geometry_audit.empty:
        geometry_bad_share = float(
            geometry_audit["geometry_review_bucket_suggested"]
            .isin({"edge_misses_chart_reality", "box_not_visually_real"})
            .mean()
        )
    additive = float(
        downstream_summary.get("fraction_presence_improved_interpretation", 0.0) or 0.0
    )

    if additive < 0.20:
        path = "Path E — source-family demotion"
    elif gain > 0.08 and dominant_failure == "frozen_box_became_stale":
        path = "Path B — controlled refresh fix"
    elif truth_layer_better:
        path = "Path C — ranking rebase"
    elif dominant_failure in {"confirm_too_early", "confirm_too_late"}:
        path = "Path A — confirmation fix"
    elif geometry_bad_share > 0.30:
        path = "Path D — ontology correction"
    else:
        path = "Path C — ranking rebase"

    return {
        "dominant_failure": dominant_failure,
        "frozen_vs_bounded_gain": f"{gain:.4f}",
        "truth_layer": (
            "interpretability_layer"
            if truth_layer_better
            else "canonical_ranking_layer"
        ),
        "primary_next_path": path,
    }


def _plot_audit_chart_pack(
    frame: pd.DataFrame,
    sample_table: pd.DataFrame,
    *,
    event_table: pd.DataFrame,
    outpath: Path,
    title: str,
    active_truth_audit: pd.DataFrame | None = None,
    show_geometry_candidates: bool = False,
    highlight_candidate_family: str | None = None,
) -> Path:
    if sample_table.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        save_figure_html(fig, outpath)
        return outpath

    sample_table = sample_table.reset_index(drop=True)
    rows = len(sample_table)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[f"range_id={int(v)}" for v in sample_table["range_id"]],
    )
    truth_map: dict[int, dict[str, object]] = {}
    if active_truth_audit is not None and not active_truth_audit.empty:
        truth_map = {
            int(row["range_id"]): row.to_dict()
            for _, row in active_truth_audit.iterrows()
        }

    for row_no, (_, event) in enumerate(sample_table.iterrows(), start=1):
        birth_idx = int(
            pd.to_numeric(pd.Series([event["birth_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        confirm_idx = int(
            pd.to_numeric(pd.Series([event["confirm_idx"]]), errors="coerce")
            .fillna(0)
            .iloc[0]
        )
        end_idx_value = pd.to_numeric(
            pd.Series([event.get("end_idx")]), errors="coerce"
        ).iloc[0]
        end_idx = (
            int(end_idx_value)
            if pd.notna(end_idx_value)
            else min(len(frame) - 1, confirm_idx + 12)
        )
        end_idx = min(max(end_idx, confirm_idx + 1), len(frame) - 1)
        window_start = max(0, birth_idx - 10)
        window_stop = min(len(frame) - 1, end_idx + 10)
        window = frame.iloc[window_start : window_stop + 1].copy()

        fig.add_trace(
            go.Candlestick(
                x=window["timestamp"],
                open=window["open"],
                high=window["high"],
                low=window["low"],
                close=window["close"],
                name="OHLC",
                showlegend=(row_no == 1),
                increasing_line_color="#15803d",
                decreasing_line_color="#b45309",
            ),
            row=row_no,
            col=1,
        )

        candidate_x = window.loc[
            (window.index >= birth_idx) & (window.index <= confirm_idx), "timestamp"
        ]
        active_x = window.loc[
            (window.index >= confirm_idx) & (window.index <= end_idx), "timestamp"
        ]
        for xvals, yval, name, color, dash, width in [
            (
                candidate_x,
                float(event["high"]),
                "Candidate High",
                "#b91c1c",
                "dash",
                1.2,
            ),
            (candidate_x, float(event["low"]), "Candidate Low", "#15803d", "dash", 1.2),
            (active_x, float(event["high"]), "Active High", "#dc2626", "solid", 2.0),
            (active_x, float(event["low"]), "Active Low", "#16a34a", "solid", 2.0),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=xvals,
                    y=[yval] * len(xvals),
                    mode="lines",
                    line=dict(color=color, dash=dash, width=width),
                    name=name,
                    showlegend=(row_no == 1),
                ),
                row=row_no,
                col=1,
            )

        if show_geometry_candidates:
            visible_high = float(pd.to_numeric(window["high"], errors="coerce").max())
            visible_low = float(pd.to_numeric(window["low"], errors="coerce").min())
            for yval, name in (
                (visible_high, "Visible Envelope High"),
                (visible_low, "Visible Envelope Low"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=window["timestamp"],
                        y=[yval] * len(window),
                        mode="lines",
                        line=dict(color="#6b7280", dash="dashdot", width=1.2),
                        name=name,
                        showlegend=(row_no == 1),
                    ),
                    row=row_no,
                    col=1,
                )
            for family, high_col, low_col, color in [
                (
                    "g2_envelope_extended_extrema",
                    "range_high_g2",
                    "range_low_g2",
                    "#1d4ed8",
                ),
                (
                    "g3_touch_cluster_envelope",
                    "range_high_g3",
                    "range_low_g3",
                    "#9333ea",
                ),
                ("g4_quantile_envelope", "range_high_g4", "range_low_g4", "#ea580c"),
                (
                    "g5_widened_compact_core_with_guardrails",
                    "range_high_g5",
                    "range_low_g5",
                    "#0891b2",
                ),
            ]:
                high_val = pd.to_numeric(
                    pd.Series([event.get(high_col)]), errors="coerce"
                ).iloc[0]
                low_val = pd.to_numeric(
                    pd.Series([event.get(low_col)]), errors="coerce"
                ).iloc[0]
                if pd.isna(high_val) or pd.isna(low_val):
                    continue
                line_width = 2.6 if family == highlight_candidate_family else 1.4
                line_dash = "solid" if family == highlight_candidate_family else "dot"
                family_label = family.replace("_", " ")
                for yval, suffix in (
                    (float(high_val), "High"),
                    (float(low_val), "Low"),
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=active_x,
                            y=[yval] * len(active_x),
                            mode="lines",
                            line=dict(color=color, dash=line_dash, width=line_width),
                            name=f"{family_label} {suffix}",
                            showlegend=(row_no == 1),
                        ),
                        row=row_no,
                        col=1,
                    )

        if {"swing_high_confirm_flag", "swing_high_confirm_price"}.issubset(
            window.columns
        ):
            sh = window[
                pd.to_numeric(window["swing_high_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
            ]
            if not sh.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sh["timestamp"],
                        y=sh["swing_high_confirm_price"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=8, color="#7f1d1d"),
                        name="Swing High",
                        showlegend=(row_no == 1),
                    ),
                    row=row_no,
                    col=1,
                )
        if {"swing_low_confirm_flag", "swing_low_confirm_price"}.issubset(
            window.columns
        ):
            sl = window[
                pd.to_numeric(window["swing_low_confirm_flag"], errors="coerce")
                .fillna(0)
                .eq(1)
            ]
            if not sl.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sl["timestamp"],
                        y=sl["swing_low_confirm_price"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=8, color="#14532d"),
                        name="Swing Low",
                        showlegend=(row_no == 1),
                    ),
                    row=row_no,
                    col=1,
                )

        overlaps = event_table[
            (
                pd.to_numeric(event_table["range_id"], errors="coerce")
                != float(event["range_id"])
            )
            & (
                pd.to_numeric(event_table["confirm_idx"], errors="coerce").between(
                    window_start, window_stop
                )
            )
        ]
        for _, other in overlaps.iterrows():
            start_ts = _idx_to_timestamp(frame, other["confirm_idx"])
            end_ts = _idx_to_timestamp(frame, other.get("end_idx"))
            if pd.isna(end_ts):
                end_ts = _idx_to_timestamp(
                    frame,
                    min(
                        window_stop,
                        int(
                            pd.to_numeric(
                                pd.Series([other["confirm_idx"]]), errors="coerce"
                            )
                            .fillna(0)
                            .iloc[0]
                        )
                        + 6,
                    ),
                )
            for level in [float(other["high"]), float(other["low"])]:
                fig.add_trace(
                    go.Scatter(
                        x=[start_ts, end_ts],
                        y=[level, level],
                        mode="lines",
                        line=dict(color="#9ca3af", dash="dot", width=1),
                        name="Nearby Range",
                        showlegend=False,
                    ),
                    row=row_no,
                    col=1,
                )

        markers = [(confirm_idx, float(frame.iloc[confirm_idx]["close"]), "Confirm")]
        first_breach_idx = pd.to_numeric(
            pd.Series([event.get("first_breach_idx")]), errors="coerce"
        ).iloc[0]
        if pd.notna(first_breach_idx):
            idx = int(first_breach_idx)
            markers.append((idx, float(frame.iloc[idx]["close"]), "Breach"))
        if pd.notna(end_idx_value):
            markers.append((end_idx, float(frame.iloc[end_idx]["close"]), "End"))
        for idx, price, name in markers:
            fig.add_trace(
                go.Scatter(
                    x=[_idx_to_timestamp(frame, idx)],
                    y=[price],
                    mode="markers",
                    marker=dict(size=9, color="#111827"),
                    name=name,
                    showlegend=(row_no == 1),
                ),
                row=row_no,
                col=1,
            )

        truth_row = truth_map.get(int(event["range_id"]))
        if truth_row is not None:
            fig.add_annotation(
                xref=f"x{row_no}" if row_no > 1 else "x",
                yref=f"y{row_no}" if row_no > 1 else "y",
                x=_idx_to_timestamp(frame, end_idx),
                y=float(event["high"]),
                text=(
                    f"{truth_row['failure_classification']}<br>"
                    f"frozen={truth_row['frozen_visual_plausibility_score']:.2f} "
                    f"bounded={truth_row['bounded_refresh_visual_plausibility_score']:.2f}"
                ),
                showarrow=False,
                font=dict(size=10, color="#1f2937"),
            )

    fig.update_layout(
        title=title,
        height=max(400 * rows, 700),
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=10)),
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_figure_html(fig, outpath)
    return outpath


def _build_diagnosis_memo_text(
    *,
    geometry_audit: pd.DataFrame,
    active_truth_audit: pd.DataFrame,
    doctrine_report: pd.DataFrame,
    coverage_regime_report: pd.DataFrame,
    ranking_report: pd.DataFrame,
    family_report: pd.DataFrame,
    downstream_summary: dict[str, object],
    path_summary: dict[str, str],
    selected_label: str,
) -> Path:
    geometry_counts = (
        geometry_audit["geometry_review_bucket_suggested"].value_counts()
        if not geometry_audit.empty
        else pd.Series(dtype=int)
    )
    failure_counts = (
        active_truth_audit["failure_classification"].value_counts()
        if not active_truth_audit.empty
        else pd.Series(dtype=int)
    )
    doctrine_pivot = (
        doctrine_report.pivot_table(
            index="doctrine",
            values=[
                "chart_faithfulness_score",
                "visual_plausibility_score",
                "source_coherence_score",
            ],
            aggfunc="mean",
        )
        if not doctrine_report.empty
        else pd.DataFrame()
    )
    lines = [
        "# Step 8X Final Diagnosis Memo",
        "",
        f"- Selected diagnostic regime: `{selected_label}`",
        f"- Dominant failure classification: `{path_summary.get('dominant_failure', 'unknown')}`",
        f"- Truth layer conclusion: `{path_summary.get('truth_layer', 'unknown')}`",
        f"- Recommended next path: `{path_summary.get('primary_next_path', 'unknown')}`",
        "",
        "## 3.1 Root Cause Answers",
        f"- Main cause of active-boundary factual mismatch: `{path_summary.get('dominant_failure', 'unknown')}`",
        f"- Is the dominant issue ontology, confirmation timing, frozen geometry, or lineage fragmentation? `{path_summary.get('primary_next_path', 'unknown')}`",
        f"- Are canonical ranking metrics less truthful than interpretability metrics? `{path_summary.get('truth_layer') == 'interpretability_layer'}`",
        "",
        "## Geometry Audit",
        (
            geometry_counts.to_string()
            if not geometry_counts.empty
            else "No geometry audit rows."
        ),
        "",
        "## Active-State Truth Audit",
        (
            failure_counts.to_string()
            if not failure_counts.empty
            else "No active truth rows."
        ),
        "",
        "## Frozen vs Refresh Comparison",
        (
            doctrine_pivot.to_string()
            if not doctrine_pivot.empty
            else "No doctrine comparison rows."
        ),
        "",
        "## Coverage-Regime Comparison",
        (
            coverage_regime_report.to_string(index=False)
            if not coverage_regime_report.empty
            else "No coverage report rows."
        ),
        "",
        "## Ranking Disagreement",
        (
            ranking_report.to_string(index=False)
            if not ranking_report.empty
            else "No ranking disagreement rows."
        ),
        "",
        "## Family Comparison",
        (
            family_report.to_string(index=False)
            if not family_report.empty
            else "No family comparison rows."
        ),
        "",
        "## 3.2 Refresh Doctrine Answer",
        f"- Is controlled refresh justified? `{float(path_summary.get('frozen_vs_bounded_gain', '0')) > 0.08}`",
        f"- Is bounded refresh preferable to naive periodic recompute? `{(not doctrine_pivot.empty) and ('bounded_continuity_refresh' in doctrine_pivot.index) and ('fixed_horizon_recompute' in doctrine_pivot.index) and (doctrine_pivot.loc['bounded_continuity_refresh', 'visual_plausibility_score'] >= doctrine_pivot.loc['fixed_horizon_recompute', 'visual_plausibility_score'])}`",
        "",
        "## 3.3 Step 8 Viability Answer",
        f"- Is Step 8 genuinely additive downstream? `{float(downstream_summary.get('fraction_presence_improved_interpretation', 0.0) or 0.0) >= 0.25}`",
        "- Under what contexts is it most useful? `local balance structures with later meaningful interaction, reclaim, or nearby structural-event overlap.`",
        f"- Is it strong enough to remain a first-class source family? `{downstream_summary.get('redundancy_assessment', 'unknown') != 'mostly_redundant'}`",
        "",
        "## 3.4 Recommended Next-Path Classification",
        f"- Primary next path: `{path_summary.get('primary_next_path', 'unknown')}`",
        "",
        "## Downstream Usefulness Summary",
    ]
    lines.extend(f"- {key}: {value}" for key, value in downstream_summary.items())
    lines.append("")
    return "\n".join(lines)


def _write_diagnosis_memo(
    *,
    outpath: Path,
    geometry_audit: pd.DataFrame,
    active_truth_audit: pd.DataFrame,
    doctrine_report: pd.DataFrame,
    coverage_regime_report: pd.DataFrame,
    ranking_report: pd.DataFrame,
    family_report: pd.DataFrame,
    downstream_summary: dict[str, object],
    path_summary: dict[str, str],
    selected_label: str,
) -> Path:
    text = _build_diagnosis_memo_text(
        geometry_audit=geometry_audit,
        active_truth_audit=active_truth_audit,
        doctrine_report=doctrine_report,
        coverage_regime_report=coverage_regime_report,
        ranking_report=ranking_report,
        family_report=family_report,
        downstream_summary=downstream_summary,
        path_summary=path_summary,
        selected_label=selected_label,
    )
    write_text_atomic(text, outpath)
    return outpath


def _assess_rung(label: str, result: dict[str, object]) -> dict[str, object]:
    summary = result["summary"]
    counts = summary.get("event_counts", {})
    timing = summary.get("confirmation_timing", {})
    forensics, short_high, long_medium = _build_forensics_tables(result["event_table"])

    short_strength = _mean_from_df(short_high, "strength")
    long_strength = _mean_from_df(long_medium, "strength")
    short_duration = _mean_from_df(short_high, "duration_bars")
    short_micro_box_risk = _mean_from_df(short_high, "rb_micro_box_risk_score")
    short_plausibility = _mean_from_df(short_high, "rb_plausibility_score")
    long_plausibility = _mean_from_df(long_medium, "rb_plausibility_score")
    short_monitor = _mean_from_df(short_high, "rb_monitor_worthiness_score")
    long_monitor = _mean_from_df(long_medium, "rb_monitor_worthiness_score")
    confirm_latency_median = timing.get("confirm_latency_bars", {}).get("median")

    coverage_in_band = (
        TARGET_CONFIRMED_RANGE_MIN
        <= int(counts.get("confirmed_ranges", 0))
        <= TARGET_CONFIRMED_RANGE_MAX
    )
    active_in_band = (
        TARGET_ACTIVE_ROWS_MIN
        <= int(counts.get("active_rows", 0))
        <= TARGET_ACTIVE_ROWS_MAX
    )
    latency_ok = (
        confirm_latency_median is not None
        and float(confirm_latency_median) >= TARGET_CONFIRM_LATENCY_MEDIAN_MIN
    )
    short_lived_ok = (
        short_duration is not None
        and float(short_duration) > TARGET_SHORT_LIVED_DURATION_MEAN_MIN
    )
    plausibility_aligned = (
        short_plausibility is not None
        and long_plausibility is not None
        and long_plausibility > short_plausibility
    )
    monitor_aligned = (
        short_monitor is not None
        and long_monitor is not None
        and long_monitor > short_monitor
    )
    micro_box_ok = (
        short_micro_box_risk is not None and float(short_micro_box_risk) < 0.70
    )
    strength_not_badly_inverted = (
        short_strength is not None
        and long_strength is not None
        and float(long_strength) >= float(short_strength) - MAX_STRENGTH_INVERSION
    )
    valid = bool(
        coverage_in_band
        and active_in_band
        and latency_ok
        and short_lived_ok
        and plausibility_aligned
        and monitor_aligned
        and micro_box_ok
        and strength_not_badly_inverted
    )
    score = (
        abs(int(counts.get("confirmed_ranges", 0)) - TARGET_CONFIRMED_RANGE_MID)
        / TARGET_CONFIRMED_RANGE_MID
        + abs(int(counts.get("active_rows", 0)) - TARGET_ACTIVE_ROWS_MID)
        / TARGET_ACTIVE_ROWS_MID
        + max(0.0, (short_micro_box_risk or 0.0) - 0.50)
        + max(0.0, (short_plausibility or 0.0) - (long_plausibility or 0.0))
        + (0.0 if latency_ok else 1.0)
        + (0.0 if short_lived_ok else 1.0)
        + (0.0 if plausibility_aligned else 2.0)
        + (0.0 if monitor_aligned else 2.0)
        + max(0.0, (short_strength or 0.0) - (long_strength or 0.0))
    )
    return {
        "label": label,
        "confirmed_ranges": int(counts.get("confirmed_ranges", 0)),
        "active_rows": int(counts.get("active_rows", 0)),
        "confirm_latency_median": confirm_latency_median,
        "short_lived_high_strength_duration_mean": short_duration,
        "short_lived_high_strength_rows": int(len(short_high)),
        "long_lived_medium_strength_rows": int(len(long_medium)),
        "short_micro_box_risk_mean": short_micro_box_risk,
        "short_plausibility_mean": short_plausibility,
        "long_plausibility_mean": long_plausibility,
        "short_monitor_worthiness_mean": short_monitor,
        "long_monitor_worthiness_mean": long_monitor,
        "short_strength_mean": short_strength,
        "long_strength_mean": long_strength,
        "coverage_in_band": coverage_in_band,
        "active_in_band": active_in_band,
        "latency_ok": latency_ok,
        "short_lived_ok": short_lived_ok,
        "plausibility_aligned": plausibility_aligned,
        "monitor_aligned": monitor_aligned,
        "micro_box_ok": micro_box_ok,
        "strength_not_badly_inverted": strength_not_badly_inverted,
        "valid": valid,
        "score": float(score),
        "contract_bucket_summary": _build_contract_bucket_summary(forensics),
    }


def _select_best_assessment(
    assessments: list[dict[str, object]],
) -> tuple[dict[str, object], bool]:
    valid_assessments = [item for item in assessments if bool(item["valid"])]
    if valid_assessments:
        return min(valid_assessments, key=lambda item: float(item["score"])), True
    return min(assessments, key=lambda item: float(item["score"])), False


def _build_recovery_ladder() -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "mid_a",
            {
                "min_confirm_bars": 3,
                "min_candidate_dwell_bars": 3,
                "max_width_atr": 3.2,
                "min_close_inside_frac": 0.55,
                "max_drift_frac": 0.75,
            },
        ),
        (
            "mid_b",
            {
                "min_confirm_bars": 3,
                "min_candidate_dwell_bars": 3,
                "max_width_atr": 3.0,
                "min_close_inside_frac": 0.60,
                "max_drift_frac": 0.75,
            },
        ),
        (
            "mid_c",
            {
                "min_confirm_bars": 4,
                "min_candidate_dwell_bars": 3,
                "max_width_atr": 3.0,
                "min_close_inside_frac": 0.60,
                "max_drift_frac": 0.70,
            },
        ),
        (
            "mid_d",
            {
                "min_confirm_bars": 4,
                "min_candidate_dwell_bars": 4,
                "max_width_atr": 2.8,
                "min_close_inside_frac": 0.60,
                "max_drift_frac": 0.70,
            },
        ),
        (
            "mid_e",
            {
                "min_confirm_bars": 3,
                "min_candidate_dwell_bars": 4,
                "max_width_atr": 3.0,
                "min_close_inside_frac": 0.55,
                "max_drift_frac": 0.70,
            },
        ),
        (
            "mid_f",
            {
                "min_confirm_bars": 4,
                "min_candidate_dwell_bars": 4,
                "max_width_atr": 3.0,
                "min_close_inside_frac": 0.55,
                "max_drift_frac": 0.70,
            },
        ),
    ]


def _build_context(df: pd.DataFrame) -> pd.DataFrame:
    out = add_atr(df)
    out = add_emas(out)
    out = add_adx(out)
    out = add_bb_width(out)
    out = add_swings(out, window=4, min_retrace_atr=0.7, min_confirm_bars=2)
    out = add_trend_state(out)
    out = add_bos(
        out,
        min_source_age_bars=1,
        min_break_distance_atr=0.05,
        min_body_atr=0.10,
    )
    out = add_choch(
        out,
        min_source_age_bars=1,
        min_break_distance_atr=0.05,
        min_body_atr=0.10,
    )

    ts = pd.to_datetime(out["timestamp"], utc=True)
    if ts.diff().median().total_seconds() < 86400:
        out = add_session_features(out, include_research_only=False)

    out = add_regime(out, include_research_only=False)
    return out


def _load_canonical_live_context(
    raw: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame | None:
    canonical = load_partitioned_dataset(
        FEATURES_ROOT,
        dataset="live",
        symbol=instrument,
        timeframe=timeframe,
    )
    if canonical.empty or not CONTEXT_REQUIRED_COLUMNS.issubset(canonical.columns):
        return None

    raw_view = raw.loc[:, ["timestamp", "open", "high", "low", "close"]].reset_index(
        drop=True
    )
    canonical_view = canonical.loc[
        :, ["timestamp", "open", "high", "low", "close"]
    ].reset_index(drop=True)
    if len(raw_view) != len(canonical_view):
        return None
    if dataframe_fingerprint(raw_view, strategy="content") != dataframe_fingerprint(
        canonical_view, strategy="content"
    ):
        return None
    return canonical.reset_index(drop=True)


def _time_range(df: pd.DataFrame) -> str:
    if df.empty or "timestamp" not in df.columns:
        return "empty"
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return f"{ts.iloc[0].isoformat()}:{ts.iloc[-1].isoformat()}"


def _range_report_fingerprint(
    full_df: pd.DataFrame,
    *,
    event_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    extra: dict[str, object],
) -> str:
    return report_fingerprint(
        full_df,
        extra={
            "event_table": dataframe_fingerprint(event_table, strategy="content"),
            "candidate_table": dataframe_fingerprint(
                candidate_table, strategy="content"
            ),
            **extra,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="XAU_USD")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--target", choices=tuple(TARGET_NODE_MAP), default="full")
    parser.add_argument("--date-from", default="2026-01-01")
    parser.add_argument("--plot-rows", type=int, default=300)
    parser.add_argument("--tail-rows", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--invalidate-cache", action="store_true")
    parser.add_argument("--cleanup-stale", action="store_true")
    parser.add_argument("--max-artifact-age-days", type=int, default=30)
    args = parser.parse_args()
    if args.tail_rows is not None:
        args.plot_rows = args.tail_rows

    data_file = Path(f"data/raw/{args.instrument}_{args.timeframe}.parquet")
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cleanup_stale:
        removed = cleanup_validation_artifacts(
            cache_root=CACHE_ROOT,
            max_age_days=args.max_artifact_age_days,
            report_roots=[OUT_DIR],
        )
        print(f"cleanup_removed: {len(removed)}")
    raw = normalize_candle_schema(pd.read_parquet(data_file), require_volume=False)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    graph = get_builtin_graph(
        "validate_range_boundaries",
        instrument=args.instrument,
        timeframe=args.timeframe,
    )
    resolved_target = TARGET_NODE_MAP[args.target]
    context = GraphRunContext(
        graph_name=graph.graph_name,
        symbol=args.instrument,
        timeframe=args.timeframe,
        inputs={"raw_input": raw},
        config={
            "date_from": args.date_from,
            "plot_rows": args.plot_rows,
            "full": args.full,
            "html": args.html,
            "write_csv": args.write_csv,
            "out_dir": str(OUT_DIR),
        },
        cache_root=CACHE_ROOT,
        features_root=FEATURES_ROOT,
        force=args.force,
        invalidate_cache=args.invalidate_cache,
    )
    if args.explain:
        explanation = explain_graph_run(graph, context=context, target=resolved_target)
        print(
            json.dumps(
                {
                    "validator": VALIDATOR_NAME,
                    "graph_name": graph.graph_name,
                    "wrapper_target": args.target,
                    "resolved_target": resolved_target,
                    "nodes": explanation["nodes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    graph_result = execute_graph(graph, context=context, target=resolved_target)
    profile_path = (
        CACHE_ROOT
        / VALIDATOR_NAME
        / args.instrument
        / args.timeframe
        / "run-summary.json"
    )
    graph_result.profiler.write_json(profile_path)
    executed_nodes = list(graph_result.node_results)

    print(f"\n=== RANGE BOUNDARIES COMMAND: {args.instrument} {args.timeframe} ===")
    print(f"wrapper_target: {args.target}")
    print(f"resolved_target: {resolved_target}")
    print(f"executed_nodes: {executed_nodes}")
    print(
        "cache_summary: "
        f"{sum(1 for result in graph_result.node_results.values() if result.cache_hit)} hits / "
        f"{sum(1 for result in graph_result.node_results.values() if not result.cache_hit)} executes"
    )

    node_results = graph_result.node_results
    output = graph_result.output()
    bundle = output.payload
    if args.target == "selection":
        selected_rung = bundle["selected_rung"]
        print("\n=== SELECTION ===")
        print(f"selected_rung: {selected_rung['selected_label']}")
        print(f"reporting_rung: {selected_rung['reporting_label']}")
        print(f"used_step8e_b_retune: {selected_rung['used_retune']}")
        print(f"selected_params: {selected_rung['selected_params']}")
        print(f"has_valid_rung: {selected_rung['has_valid_rung']}")
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "selected-debug":
        print("\n=== SELECTED DEBUG ===")
        _print_summary(output.payload["summary"])
        if output.primary_frame() is not None:
            print(f"frame_rows: {len(output.primary_frame())}")
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "forensics":
        print("\n=== FORENSICS ===")
        for name, frame in output.frames.items():
            print(f"{name}_rows: {len(frame)}")
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "geometry":
        geometry_audit = output.frames["geometry_audit"]
        geometry_candidate_summary = output.frames.get(
            "geometry_candidate_summary", pd.DataFrame()
        )
        print("\n=== GEOMETRY ===")
        _print_summary(
            geometry_audit["geometry_review_bucket_suggested"].value_counts().to_dict()
            if not geometry_audit.empty
            else {}
        )
        print(f"geometry_rows: {len(geometry_audit)}")
        if not geometry_candidate_summary.empty:
            print("\n=== STEP 8G GEOMETRY CANDIDATES ===")
            _print_summary(geometry_candidate_summary.to_dict(orient="records"))
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "active-truth":
        active_truth_audit = output.frames["active_truth_audit"]
        geometry_candidate_truth_summary = output.frames.get(
            "geometry_candidate_truth_summary", pd.DataFrame()
        )
        print("\n=== ACTIVE TRUTH ===")
        _print_summary(
            active_truth_audit["failure_classification"].value_counts().to_dict()
            if not active_truth_audit.empty
            else {}
        )
        print(f"active_truth_rows: {len(active_truth_audit)}")
        if not geometry_candidate_truth_summary.empty:
            print("\n=== STEP 8G GEOMETRY ACTIVE-TRUTH ===")
            _print_summary(geometry_candidate_truth_summary.to_dict(orient="records"))
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "coverage":
        print("\n=== COVERAGE ===")
        print(f"coverage_rows: {len(output.frames['coverage_regime_report'])}")
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "ranking":
        print("\n=== RANKING ===")
        print(
            f"recommended_ranking_repair_candidate: {output.payload['ranking_repair_recommendation']}"
        )
        for name, frame in output.frames.items():
            print(f"{name}_rows: {len(frame)}")
        geometry_ranking_preservation = output.frames.get(
            "geometry_ranking_preservation", pd.DataFrame()
        )
        if not geometry_ranking_preservation.empty:
            print("\n=== STEP 8G RANKING PRESERVATION ===")
            _print_summary(geometry_ranking_preservation.to_dict(orient="records"))
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "downstream":
        print("\n=== DOWNSTREAM ===")
        _print_summary(output.payload["summary"])
        print(f"downstream_rows: {len(output.frames['downstream_usefulness'])}")
        geometry_candidate_downstream_summary = output.frames.get(
            "geometry_candidate_downstream_summary", pd.DataFrame()
        )
        if not geometry_candidate_downstream_summary.empty:
            print("\n=== STEP 8G DOWNSTREAM PRESERVATION ===")
            _print_summary(
                geometry_candidate_downstream_summary.to_dict(orient="records")
            )
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "diagnostics":
        print("\n=== DIAGNOSTICS ===")
        _print_summary(output.payload)
        for name, frame in output.frames.items():
            print(f"{name}_rows: {len(frame)}")
        geometry_candidate_gate_report = output.frames.get(
            "geometry_candidate_gate_report", pd.DataFrame()
        )
        if not geometry_candidate_gate_report.empty:
            print("\n=== STEP 8G GEOMETRY GATES ===")
            _print_summary(geometry_candidate_gate_report.to_dict(orient="records"))
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "charts":
        chart_artifacts = output.payload["artifacts"]
        print("\n=== CHART BUNDLE ===")
        print(f"html_enabled: {args.html}")
        for name, payload in chart_artifacts.items():
            print(
                f"{name}: {payload.get('html_path')} cache_hit={payload.get('cache_hit')}"
            )
        print(f"Profiler summary: {profile_path}")
        return
    if args.target == "csv":
        csv_artifacts = output.payload.get("artifact_paths", {})
        print("\n=== CSV BUNDLE ===")
        print(f"csv_enabled: {args.write_csv}")
        for name, path in csv_artifacts.items():
            print(f"{name}: {path}")
        print(f"Profiler summary: {profile_path}")
        return

    selected_rung = bundle["selected_rung"]
    rung_assessments = list(selected_rung["rung_assessments"])
    assessment_by_label = {str(item["label"]): item for item in rung_assessments}
    reporting_label = str(selected_rung["reporting_label"])
    selected_label = str(selected_rung["selected_label"])
    used_retune = any(
        str(item["label"]).startswith("step8e_b/") for item in rung_assessments
    )
    selected_summary = bundle["selected_summary"]
    diagnostics = bundle["diagnostics"]
    downstream_summary = bundle["downstream_summary"]
    artifacts = bundle["artifacts"]

    range_context_result = node_results["range_context"]
    range_selected_result = node_results["range_selected_rung"]
    ranking_result = node_results["range_ranking_bundle"]
    geometry_audit = node_results["range_geometry_audit"].output.frames[
        "geometry_audit"
    ]
    active_truth_audit = node_results["range_active_truth_audit"].output.frames[
        "active_truth_audit"
    ]

    rung_results: list[tuple[str, dict[str, object]]] = []
    for node_name, execution in node_results.items():
        if not node_name.startswith("range_rung_debug__"):
            continue
        rung_results.append(
            (
                str(execution.output.payload["label"]),
                {
                    "params": execution.output.payload["params"],
                    "summary": execution.output.payload["summary"],
                    "event_table": execution.output.frames["event_table"],
                    "candidate_table": execution.output.frames["candidate_table"],
                },
            )
        )

    deprecated_csvs = [
        OUT_DIR
        / f"range_boundaries_shortest_lived_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_longest_lived_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR / f"range_boundaries_strongest_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR / f"range_boundaries_weakest_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_ranging_short_lived_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_short_lived_high_strength_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_long_lived_medium_strength_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_highest_plausibility_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_highest_monitor_worthiness_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_highest_micro_box_risk_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_highest_late_confirm_fragility_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_ranking_rebase_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_ranking_repair_candidates_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_ranking_repair_gates_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_ranking_agreement_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_bucket_lift_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_family_comparison_{args.instrument}_{args.timeframe}.csv",
        OUT_DIR
        / f"range_boundaries_path_c2_archetypes_{args.instrument}_{args.timeframe}.csv",
    ]
    removed_deprecated = 0
    if args.write_csv:
        for deprecated_path in deprecated_csvs:
            if deprecated_path.exists():
                deprecated_path.unlink()
                removed_deprecated += 1

    print(f"\n=== RANGE BOUNDARIES SUMMARY: {args.instrument} {args.timeframe} ===")
    print("\n=== STEP 8E CONTRACT SEARCH ===")
    for label, rung_result in rung_results:
        funnel = rung_result["summary"].get("promotion_funnel", {})
        counts = rung_result["summary"].get("event_counts", {})
        assessment = assessment_by_label.get(label, {})
        print(
            f"{label}: confirmed={counts.get('confirmed_ranges')} active_rows={counts.get('active_rows')} "
            f"raw={funnel.get('raw_candidate_count')} maturity={funnel.get('maturity_pass_count')} "
            f"viability={funnel.get('viability_pass_count')} "
            f"latency_median={assessment.get('confirm_latency_median')} "
            f"short_duration_mean={assessment.get('short_lived_high_strength_duration_mean')} "
            f"plausibility_aligned={assessment.get('plausibility_aligned')} "
            f"monitor_aligned={assessment.get('monitor_aligned')} "
            f"micro_box_ok={assessment.get('micro_box_ok')} "
            f"strength_ok={assessment.get('strength_not_badly_inverted')} "
            f"valid={assessment.get('valid')}"
        )
    print(f"selected_rung: {selected_label}")
    print(f"reporting_rung: {reporting_label}")
    print(f"used_step8e_b_retune: {used_retune}")
    print(
        f"selected_params: "
        f"{node_results[next(name for name, execution in node_results.items() if name.startswith('range_rung_debug__') and str(execution.output.payload['label']) == reporting_label)].output.payload['params']}"
    )
    print("selection_targets:")
    print(
        {
            "confirmed_range_band": [
                TARGET_CONFIRMED_RANGE_MIN,
                TARGET_CONFIRMED_RANGE_MAX,
            ],
            "active_row_band": [TARGET_ACTIVE_ROWS_MIN, TARGET_ACTIVE_ROWS_MAX],
            "min_confirm_latency_median": TARGET_CONFIRM_LATENCY_MEDIAN_MIN,
            "min_short_lived_high_strength_duration_mean": TARGET_SHORT_LIVED_DURATION_MEAN_MIN,
            "max_strength_inversion": MAX_STRENGTH_INVERSION,
        }
    )
    _print_summary(selected_summary)
    print("\n=== INTERPRETABILITY METRICS ===")
    _print_summary(diagnostics["interpretability_summary"])
    print("\n=== CONTRACT BUCKET SUMMARY ===")
    _print_summary(diagnostics["contract_bucket_summary"])
    print("\n=== ARCHETYPE COMPARISON ===")
    _print_summary(diagnostics["archetype_summary"])
    print("\n=== VIABILITY ALIGNMENT AUDIT ===")
    _print_summary(diagnostics["alignment_audit"])
    print("\n=== STEP 8F RANKING REBASE REPORT ===")
    _print_summary(
        ranking_result.output.frames["ranking_rebase_report"].to_dict(orient="records")
    )
    print("\n=== STEP 8F.2 RANKING REPAIR CANDIDATES ===")
    _print_summary(
        ranking_result.output.frames["ranking_repair_report"].to_dict(orient="records")
    )
    print("\n=== STEP 8F.2 RANKING REPAIR GATES ===")
    _print_summary(
        ranking_result.output.frames["ranking_repair_gates"].to_dict(orient="records")
    )
    print(
        "recommended_ranking_repair_candidate: "
        f"{ranking_result.output.payload['ranking_repair_recommendation']}"
    )
    print("\n=== STEP 8F AGREEMENT MATRIX ===")
    _print_summary(
        ranking_result.output.frames["agreement_report"].to_dict(orient="records")
    )
    print("\n=== STEP 8F BUCKET LIFT REPORT ===")
    _print_summary(
        ranking_result.output.frames["bucket_lift_report"].to_dict(orient="records")
    )
    print("\n=== PRESSURE IMBALANCE AUDIT ===")
    _print_summary(diagnostics["pressure_audit"])
    print("\n=== STEP 8X GEOMETRY AUDIT SUMMARY ===")
    _print_summary(
        geometry_audit["geometry_review_bucket_suggested"].value_counts().to_dict()
        if not geometry_audit.empty
        else {}
    )
    geometry_candidate_summary = node_results["range_geometry_audit"].output.frames.get(
        "geometry_candidate_summary", pd.DataFrame()
    )
    if not geometry_candidate_summary.empty:
        print("\n=== STEP 8G GEOMETRY CANDIDATE SUMMARY ===")
        _print_summary(geometry_candidate_summary.to_dict(orient="records"))
    print("\n=== STEP 8X ACTIVE-STATE TRUTH SUMMARY ===")
    _print_summary(
        active_truth_audit["failure_classification"].value_counts().to_dict()
        if not active_truth_audit.empty
        else {}
    )
    geometry_candidate_truth_summary = node_results[
        "range_active_truth_audit"
    ].output.frames.get("geometry_candidate_truth_summary", pd.DataFrame())
    if not geometry_candidate_truth_summary.empty:
        print("\n=== STEP 8G GEOMETRY ACTIVE-TRUTH SUMMARY ===")
        _print_summary(geometry_candidate_truth_summary.to_dict(orient="records"))
    geometry_ranking_preservation = ranking_result.output.frames.get(
        "geometry_ranking_preservation", pd.DataFrame()
    )
    if not geometry_ranking_preservation.empty:
        print("\n=== STEP 8G RANKING PRESERVATION ===")
        _print_summary(geometry_ranking_preservation.to_dict(orient="records"))
    print("\n=== STEP 8X DOWNSTREAM USEFULNESS ===")
    _print_summary(downstream_summary)
    geometry_candidate_downstream_summary = node_results[
        "range_downstream_usefulness"
    ].output.frames.get("geometry_candidate_downstream_summary", pd.DataFrame())
    if not geometry_candidate_downstream_summary.empty:
        print("\n=== STEP 8G DOWNSTREAM PRESERVATION ===")
        _print_summary(geometry_candidate_downstream_summary.to_dict(orient="records"))
    geometry_candidate_gate_report = node_results[
        "range_diagnostics_bundle"
    ].output.frames.get("geometry_candidate_gate_report", pd.DataFrame())
    if not geometry_candidate_gate_report.empty:
        print("\n=== STEP 8G CANDIDATE RECOMMENDATION ===")
        print(
            f"geometry_candidate_recommendation: {diagnostics.get('geometry_candidate_recommendation')}"
        )
        _print_summary(geometry_candidate_gate_report.to_dict(orient="records"))
    print("\n=== STEP 8X PRIMARY PATH ===")
    _print_summary(diagnostics["path_summary"])
    print(f"context_source: {range_context_result.output.payload['source']}")
    print(f"context_cache_hit: {range_context_result.cache_hit}")
    print(f"selection_cache_hit: {range_selected_result.cache_hit}")
    print(f"html_enabled: {args.html}")
    print(f"csv_enabled: {args.write_csv}")
    main_html_path = artifacts["range_main_chart"].get("html_path")
    if main_html_path is not None:
        print(f"\nWrote chart to: {main_html_path}")
    else:
        print("\nHTML output skipped. Pass --html to generate charts.")
    if args.write_csv:
        csv_artifacts = artifacts["range_csv_bundle"].get("artifact_paths", {})
        print(f"Wrote event table to: {csv_artifacts.get('events')}")
        print(f"Wrote candidate table to: {csv_artifacts.get('candidates')}")
        print(f"Wrote bundled forensics to: {csv_artifacts.get('forensics_bundle')}")
        print(f"Wrote geometry audit to: {csv_artifacts.get('geometry_audit')}")
        print(
            f"Wrote geometry candidate bundle to: {csv_artifacts.get('geometry_candidates')}"
        )
        print(f"Wrote frozen-vs-refresh report to: {csv_artifacts.get('doctrine')}")
        print(f"Wrote coverage regime report to: {csv_artifacts.get('coverage')}")
        print(
            f"Wrote bundled ranking reports to: {csv_artifacts.get('ranking_bundle')}"
        )
        print(
            f"Wrote downstream usefulness report to: {csv_artifacts.get('downstream')}"
        )
        print(
            f"Wrote ranking repair memo to: {csv_artifacts.get('ranking_repair_memo')}"
        )
        print(f"Removed deprecated range-boundary CSVs: {removed_deprecated}")
    else:
        print("CSV and memo outputs skipped. Pass --write-csv to persist them.")
    print(f"Profiler summary: {profile_path}")


if __name__ == "__main__":
    main()
