from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.indicators._helpers.arrays import get_atr_array
from src.indicators.smc.equal_hl import (
    EQHL_SELECTOR_MODE_CALIBRATION,
    TRADEABLE_WIDTH_REF,
)

from src.indicators.research.equal_hl_research import (
    build_equal_hl_research_table,
    summarize_equal_hl_research,
)

EQHL_TUNING_BASELINE_PARAMS = {
    "atr_tolerance": 0.12,
    "lookback_swings": 50,
}
EQHL_TUNING_GRID = {
    "atr_tolerance": (0.12, 0.15, 0.18),
    "lookback_swings": (50, 40, 30, 25),
}
EQHL_TUNING_TIME_EXIT_HORIZON = 5
EQHL_TUNING_FIXED_STOP_ATR = 1.0
EQHL_TOUCH_DOMINANCE_WARN_THRESHOLD = 0.90
EQHL_CALIBRATION_TOUCH_WEIGHT_GRID = (0.0, 0.05, 0.10)
EQHL_CALIBRATION_WEIGHT_STEP = 0.10
EQHL_CALIBRATION_REFINED_STEP = 0.05
EQHL_CALIBRATION_REFINED_DELTA = 0.10
EQHL_COMPONENT_SIGN_VARIANTS = ("current", "inverted", "neutralized")
EQHL_V2_ACTIVE_AGE_WEIGHT = 0.15
EQHL_V2_OBJECTIVE_KEYS = (
    "accepted",
    "top_decile_avg_r",
    "top_quartile_avg_r",
    "rank_corr",
    "win_rate_gap",
)
EQHL_THRESHOLD_TOP_BUCKET_TOLERANCE = 0.02
EQHL_THRESHOLD_FORMATION_DELAY_TOLERANCE = 3.0
EQHL_THRESHOLD_ACTIVE_AGE_TOLERANCE = 10.0
EQHL_THRESHOLD_SIDE_COLLAPSE_FLOOR = -0.05
EQHL_WIDTH_REF_PERCENTILE = 0.80
EPS = 1e-12

# ── Frozen economic acceptance criteria ────────────────────────────────────────
# These are the hard targets a score candidate must meet to be considered "done".
# Do not relax these without explicit agreement.
EQHL_ACCEPT_MIN_RANK_CORR = 0.05  # overall Spearman rank corr > 0
EQHL_ACCEPT_MIN_TOP_QUARTILE_AVG_R = 0.0  # top-25% trades must be profitable
EQHL_ACCEPT_MIN_TOP_VS_BOTTOM_AVG_R = (
    0.05  # top-quartile must beat bottom-decile by ≥0.05R
)
EQHL_ACCEPT_MIN_WIN_RATE_GAP = (
    0.03  # top-quartile win-rate must beat bottom-decile by ≥3pp
)
EQHL_ACCEPT_SIDE_RANK_CORR_FLOOR = (
    -0.05
)  # neither EQH nor EQL rank-corr may go below this
EQHL_ACCEPT_TOP_DECILE_IMPROVES_QUARTILE = True  # top-10% avg_r must be ≥ top-25% avg_r

# ── Temporal stability thresholds ──────────────────────────────────────────────
EQHL_STABILITY_IC_THRESHOLD = 0.03  # |rank_corr| floor to call a feature "directional"
EQHL_MONOTONICITY_N_BUCKETS = 5  # quantile buckets for monotonicity testing
EQHL_STABILITY_MIN_HALF_RANK_CORR = (
    0.0  # rank_corr must be non-negative in each time half
)
EQHL_STABILITY_MIN_HALF_TOP_VS_BOTTOM = (
    0.0  # top-quartile avg_r must beat bottom-decile in each half
)


@dataclass(frozen=True)
class EqhlScoreCandidate:
    name: str
    signs: dict[str, str]
    weights: dict[str, float]


def build_eqhl_score_candidate(candidate_row: dict[str, object]) -> EqhlScoreCandidate:
    return EqhlScoreCandidate(
        name=str(candidate_row.get("name", "v2")),
        signs=dict(candidate_row["signs"]),
        weights={
            component: float(candidate_row.get(component, 0.0))
            for component in (
                "width_component",
                "formation_delay_component",
                "wick_ratio_component",
                "atr_percentile_component",
                "distance_at_detect_component",
            )
        },
    )


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


def _detect_rows(df: pd.DataFrame) -> pd.DataFrame:
    def _side_rows(prefix: str) -> pd.DataFrame:
        positions = np.flatnonzero(
            pd.to_numeric(df.get(f"{prefix}_detect_flag"), errors="coerce")
            .fillna(0)
            .to_numpy()
            == 1
        )
        if len(positions) == 0:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "side",
                    "event_id",
                    "cluster_id",
                    "detect_idx",
                    "level",
                    "width_atr",
                    "touch_count",
                    "width_component",
                    "formation_delay_component",
                    "structural_score",
                    "tradeable_live_score",
                    "score",
                    "formation_delay",
                ]
            )
        return pd.DataFrame(
            {
                "timestamp": (
                    pd.to_datetime(df.iloc[positions]["timestamp"], utc=True).to_numpy()
                    if "timestamp" in df.columns
                    else pd.Series(pd.NaT, index=np.arange(len(positions)))
                ),
                "side": prefix,
                "event_id": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_event_id"], errors="coerce"
                ).to_numpy(dtype=float),
                "cluster_id": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_cluster_id_on_detect"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                "detect_idx": positions.astype(int),
                "level": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_level_on_detect"], errors="coerce"
                ).to_numpy(dtype=float),
                "width_atr": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_width_atr_on_detect"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                "touch_count": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_member_count_on_detect"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                "width_component": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_width_component_on_detect"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                "formation_delay_component": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_formation_delay_component_on_detect"],
                    errors="coerce",
                ).to_numpy(dtype=float),
                "structural_score": pd.to_numeric(
                    df.iloc[positions].get(
                        f"{prefix}_structural_score_on_detect",
                        df.iloc[positions][f"{prefix}_score_on_detect"],
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float),
                "tradeable_live_score": pd.to_numeric(
                    df.iloc[positions].get(
                        f"{prefix}_tradeable_live_score_on_detect",
                        df.iloc[positions][f"{prefix}_score_on_detect"],
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float),
                "score": pd.to_numeric(
                    df.iloc[positions][f"{prefix}_score_on_detect"], errors="coerce"
                ).to_numpy(dtype=float),
                "formation_delay": pd.to_numeric(
                    df.iloc[positions].get(
                        f"{prefix}_formation_delay_on_detect",
                        df.iloc[positions][f"{prefix}_detect_idx"]
                        - df.iloc[positions][f"{prefix}_first_member_detect_idx"],
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float),
                "last_member_origin_idx": pd.to_numeric(
                    df.iloc[positions].get(
                        f"{prefix}_last_member_origin_idx",
                        pd.Series(np.nan, index=df.iloc[positions].index),
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float),
                "distance_atr_at_detect": pd.to_numeric(
                    df.iloc[positions].get(
                        f"{prefix}_active_distance_atr",
                        pd.Series(np.nan, index=df.iloc[positions].index),
                    ),
                    errors="coerce",
                ).to_numpy(dtype=float),
            }
        )

    combined = pd.concat([_side_rows("eqh"), _side_rows("eql")], ignore_index=True)
    return combined.sort_values(["detect_idx", "event_id"]).reset_index(drop=True)


def _score_tier(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value >= 0.75:
        return "A"
    if value >= 0.55:
        return "B"
    if value >= 0.35:
        return "C"
    return "D"


def _touch_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 2:
        return "2-touch"
    return "3-touch+"


def _formation_delay_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 10:
        return "0_10"
    if value <= 25:
        return "11_25"
    if value <= 50:
        return "26_50"
    return "51+"


def _active_age_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 20:
        return "0_20"
    if value <= 50:
        return "21_50"
    if value <= 100:
        return "51_100"
    return "101+"


def _freshness_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    return "fresh" if value <= 20 else "stale"


def _classify_feature_stability(
    overall_corr: float,
    first_corr: float,
    second_corr: float,
    *,
    threshold: float = EQHL_STABILITY_IC_THRESHOLD,
) -> str:
    """Classify a feature's temporal stability from rank-corr across time splits.

    Returns one of:
      stable_positive  — directional positive in both halves
      stable_negative  — directional negative in both halves
      unstable         — sign flips between halves
      weak             — |rank_corr| below threshold in all available segments
      degenerate       — no finite correlation available (zero variance)
    """
    half_vals = [v for v in (first_corr, second_corr) if np.isfinite(v)]
    if not half_vals:
        if not np.isfinite(overall_corr):
            return "degenerate"
        return (
            "weak"
            if abs(overall_corr) < threshold
            else ("stable_positive" if overall_corr > 0 else "stable_negative")
        )
    directional = [v for v in half_vals if abs(v) >= threshold]
    if not directional:
        return "weak"
    if len(directional) == 2 and directional[0] > 0 and directional[1] < 0:
        return "unstable"
    if len(directional) == 2 and directional[0] < 0 and directional[1] > 0:
        return "unstable"
    if all(v > 0 for v in half_vals):
        return "stable_positive"
    if all(v < 0 for v in half_vals):
        return "stable_negative"
    return "unstable"


def _monotonicity_ratio(
    feature: pd.Series,
    target: pd.Series,
    *,
    n_buckets: int = EQHL_MONOTONICITY_N_BUCKETS,
) -> float:
    """Fraction of consecutive bucket pairs where avg_r is ascending (positive direction).

    Returns float in [0, 1]: 1.0 = perfectly monotone positive, 0.0 = perfectly inverse.
    Returns NaN if insufficient data.
    """
    frame = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature, errors="coerce"),
            "target": pd.to_numeric(target, errors="coerce"),
        }
    ).dropna()
    if len(frame) < n_buckets * 3:
        return np.nan
    try:
        frame["bucket"] = pd.qcut(
            frame["feature"], n_buckets, labels=False, duplicates="drop"
        )
    except Exception:
        return np.nan
    bucket_avg = frame.groupby("bucket")["target"].mean().sort_index()
    if len(bucket_avg) < 2:
        return np.nan
    vals = bucket_avg.to_numpy()
    n_pairs = len(vals) - 1
    n_ascending = int(np.sum(np.diff(vals) > 0))
    return float(n_ascending / n_pairs)


def _spearman_rank_corr(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame(
        {"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}
    ).dropna()
    if len(frame) < 3:
        return np.nan
    x_rank = frame["x"].rank(method="average")
    y_rank = frame["y"].rank(method="average")
    if x_rank.nunique(dropna=True) <= 1 or y_rank.nunique(dropna=True) <= 1:
        return np.nan
    return float(x_rank.corr(y_rank))


def _clip01_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=0.0, upper=1.0)


# Cap for normalizing distance-at-detect to [0, 1]: 5 ATR covers most meaningful approaches
_DISTANCE_AT_DETECT_CAP_ATR = 5.0


def _wick_ratio_component(series: pd.Series) -> pd.Series:
    """Normalize wick_ratio (already in [0,1]) to a score component."""
    return _clip01_series(pd.to_numeric(series, errors="coerce"))


def _atr_percentile_component(series: pd.Series) -> pd.Series:
    """Normalize atr_percentile (already in [0,1]) to a score component."""
    return _clip01_series(pd.to_numeric(series, errors="coerce"))


def _distance_at_detect_component(series: pd.Series) -> pd.Series:
    """Normalize distance-from-price-to-zone (ATR units) to a score component.

    Higher distance → higher score (price further from zone = cleaner approach setup).
    Capped at _DISTANCE_AT_DETECT_CAP_ATR to avoid outlier dominance.
    """
    values = pd.to_numeric(series, errors="coerce").clip(
        lower=0.0, upper=_DISTANCE_AT_DETECT_CAP_ATR
    )
    return (values / _DISTANCE_AT_DETECT_CAP_ATR).clip(lower=0.0, upper=1.0)


def _calibrated_width_reference(
    width_atr: pd.Series,
    *,
    percentile: float = EQHL_WIDTH_REF_PERCENTILE,
) -> float:
    clean = pd.to_numeric(width_atr, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return float(TRADEABLE_WIDTH_REF)
    ref = float(clean.quantile(percentile))
    if not np.isfinite(ref) or ref <= 0:
        fallback = float(clean.max()) if not clean.empty else float(TRADEABLE_WIDTH_REF)
        if np.isfinite(fallback) and fallback > 0:
            return fallback
        return float(TRADEABLE_WIDTH_REF)
    return ref


def _structural_score_quartiles(structural_score: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(structural_score, errors="coerce")
    result = pd.Series("unknown", index=numeric.index, dtype=object)
    clean = numeric.dropna()
    if clean.empty:
        return result
    ranked = clean.rank(method="first")
    quartiles = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    result.loc[clean.index] = quartiles.astype(str)
    return result


def _subperiod_labels(detect_idx: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(detect_idx, errors="coerce")
    result = pd.Series("unknown", index=numeric.index, dtype=object)
    clean = numeric.dropna().sort_values(kind="mergesort")
    if clean.empty:
        return result
    split = int(np.ceil(len(clean) / 2.0))
    result.loc[clean.index[:split]] = "first_half"
    result.loc[clean.index[split:]] = "second_half"
    return result


def _prepare_calibration_events(
    calibration_events: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    prepared = calibration_events.copy()
    width_reference = _calibrated_width_reference(prepared.get("width_atr"))
    prepared["width_component"] = _clip01_series(
        1.0
        - pd.to_numeric(prepared.get("width_atr"), errors="coerce") / width_reference
    )
    _nan = pd.Series(np.nan, index=prepared.index)
    prepared["wick_ratio_component"] = _wick_ratio_component(
        prepared.get("wick_ratio", _nan)
    )
    prepared["atr_percentile_component"] = _atr_percentile_component(
        prepared.get("atr_percentile", _nan)
    )
    prepared["distance_at_detect_component"] = _distance_at_detect_component(
        prepared.get("distance_atr_at_detect", _nan)
    )
    prepared["structural_score_quartile"] = _structural_score_quartiles(
        prepared.get("structural_score")
    )
    prepared["subperiod"] = _subperiod_labels(prepared.get("detect_idx"))
    return prepared, float(width_reference)


def _prepare_active_snapshots(
    active_snapshots: pd.DataFrame,
    *,
    width_reference: float,
) -> pd.DataFrame:
    prepared = active_snapshots.copy()
    prepared["width_component"] = _clip01_series(
        1.0
        - pd.to_numeric(prepared.get("width_atr"), errors="coerce") / width_reference
    )
    # wick_ratio and atr_percentile are detect-time features not updated per-bar;
    # they will be absent from active_snapshots and handled as neutral in _score_with_candidate.
    # distance_atr is available per-bar and serves as a proxy for distance_at_detect_component.
    _nan = pd.Series(np.nan, index=prepared.index)
    prepared["distance_at_detect_component"] = _distance_at_detect_component(
        prepared.get("distance_atr", _nan)
    )
    return prepared


def _normalize_sign_variant(series: pd.Series, variant: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if variant == "current":
        return values
    if variant == "inverted":
        return 1.0 - values
    if variant == "neutralized":
        return pd.Series(0.5, index=values.index, dtype=float)
    raise ValueError(f"Unknown sign variant: {variant}")


def _component_columns(prefix: str) -> dict[str, str]:
    return {
        "width": f"{prefix}_width_component",
        "formation_delay": f"{prefix}_formation_delay_component",
        "wick_ratio": f"{prefix}_wick_ratio_component",
        "atr_percentile": f"{prefix}_atr_percentile_component",
        "distance_at_detect": f"{prefix}_distance_at_detect_component",
    }


def _score_with_candidate(
    table: pd.DataFrame,
    *,
    candidate: EqhlScoreCandidate,
) -> pd.Series:
    score = pd.Series(0.0, index=table.index, dtype=float)
    total_weight = 0.0
    for component, weight in candidate.weights.items():
        if weight <= 0:
            continue
        col = table.get(component)
        if col is None:
            # Component column absent from this table (e.g. detect-time features in
            # active_snapshots): treat as neutral (0.5) so score stays meaningful.
            col = pd.Series(0.5, index=table.index, dtype=float)
        else:
            col = pd.to_numeric(col, errors="coerce")
        normalized = _normalize_sign_variant(
            col.fillna(0.5), candidate.signs.get(component, "current")
        )
        score = score + weight * normalized
        total_weight += weight
    return score.clip(lower=0.0, upper=1.0)


def apply_eqhl_score_candidate(
    calibration_events: pd.DataFrame,
    *,
    candidate_row: dict[str, object],
    output_column: str = "tradeable_live_score",
    width_reference: float | None = None,
) -> pd.DataFrame:
    if width_reference is None:
        scored, _ = _prepare_calibration_events(calibration_events)
    else:
        scored = calibration_events.copy()
        scored["width_component"] = _clip01_series(
            1.0
            - pd.to_numeric(scored.get("width_atr"), errors="coerce") / width_reference
        )
    scored[output_column] = _score_with_candidate(
        scored,
        candidate=build_eqhl_score_candidate(candidate_row),
    )
    return scored


def _max_drawdown(r_values: pd.Series) -> float:
    clean = pd.to_numeric(r_values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan
    equity = np.cumsum(clean)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity - peaks
    return float(drawdown.min())


def build_equal_hl_trade_table(
    df: pd.DataFrame,
    *,
    research_table: pd.DataFrame | None = None,
    time_exit_horizon: int = EQHL_TUNING_TIME_EXIT_HORIZON,
    fixed_stop_atr: float = EQHL_TUNING_FIXED_STOP_ATR,
    atr_length: int = 14,
) -> pd.DataFrame:
    if research_table is None:
        research_table = build_equal_hl_research_table(df)
    if research_table.empty:
        return pd.DataFrame(
            columns=[
                "eqhl_event_id",
                "detect_idx",
                "side",
                "direction",
                "structural_score",
                "tradeable_live_score",
                "realized_r",
                "exit_reason",
                "exit_idx",
            ]
        )

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    atr = get_atr_array(df, atr_length).astype(float, copy=False)

    rows: list[dict[str, object]] = []
    for event in research_table.to_dict("records"):
        detect_idx = int(event["eqhl_r_detect_idx"])
        exit_limit = detect_idx + int(time_exit_horizon)
        if exit_limit >= len(df):
            continue

        atr_detect = float(atr[detect_idx])
        entry = float(close[detect_idx])
        direction = int(event["eqhl_r_direction"])
        if not np.isfinite(entry) or not np.isfinite(atr_detect) or atr_detect <= 0:
            continue

        risk = float(fixed_stop_atr) * atr_detect
        stop = entry - risk if direction == 1 else entry + risk
        exit_idx = exit_limit
        exit_reason = "time"
        realized_r = np.nan

        for idx in range(detect_idx + 1, exit_limit + 1):
            if direction == 1 and low[idx] <= stop:
                exit_idx = idx
                exit_reason = "stop"
                realized_r = -1.0
                break
            if direction == -1 and high[idx] >= stop:
                exit_idx = idx
                exit_reason = "stop"
                realized_r = -1.0
                break

        if not np.isfinite(realized_r):
            realized_r = float(((close[exit_idx] - entry) * direction) / risk)

        rows.append(
            {
                "eqhl_event_id": int(event["eqhl_event_id"]),
                "detect_idx": detect_idx,
                "side": event["eqhl_r_side"],
                "direction": direction,
                "structural_score": float(event.get("eqhl_r_structural_score", np.nan)),
                "tradeable_live_score": float(
                    event.get("eqhl_r_tradeable_live_score_on_detect", np.nan)
                ),
                "realized_r": realized_r,
                "exit_reason": exit_reason,
                "exit_idx": int(exit_idx),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "eqhl_event_id",
                "detect_idx",
                "side",
                "direction",
                "structural_score",
                "tradeable_live_score",
                "realized_r",
                "exit_reason",
                "exit_idx",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["detect_idx", "eqhl_event_id"])
        .reset_index(drop=True)
    )


def _trade_metrics(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "count": 0,
            "win_rate": np.nan,
            "avg_r": np.nan,
            "expectancy": np.nan,
            "max_drawdown_r": np.nan,
        }
    realized = pd.to_numeric(trades["realized_r"], errors="coerce")
    clean = realized.dropna()
    if clean.empty:
        return {
            "count": int(len(trades)),
            "win_rate": np.nan,
            "avg_r": np.nan,
            "expectancy": np.nan,
            "max_drawdown_r": np.nan,
        }
    return {
        "count": int(clean.count()),
        "win_rate": float((clean > 0).mean()),
        "avg_r": float(clean.mean()),
        "expectancy": float(clean.mean()),
        "max_drawdown_r": _max_drawdown(clean),
    }


def _score_bucket_metrics(
    trades: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, object]:
    scoped = trades.dropna(subset=[score_column]).copy()
    if scoped.empty:
        return {
            "tier_counts": {},
            "top_quartile": _trade_metrics(pd.DataFrame(columns=trades.columns)),
            "top_decile": _trade_metrics(pd.DataFrame(columns=trades.columns)),
            "bottom_decile": _trade_metrics(pd.DataFrame(columns=trades.columns)),
        }
    scoped["score_tier"] = pd.to_numeric(scoped[score_column], errors="coerce").map(
        _score_tier
    )
    q75 = float(scoped[score_column].quantile(0.75))
    q90 = float(scoped[score_column].quantile(0.90))
    q10 = float(scoped[score_column].quantile(0.10))
    return {
        "tier_counts": (
            scoped["score_tier"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
        ),
        "top_quartile": _trade_metrics(scoped.loc[scoped[score_column] >= q75]),
        "top_decile": _trade_metrics(scoped.loc[scoped[score_column] >= q90]),
        "bottom_decile": _trade_metrics(scoped.loc[scoped[score_column] <= q10]),
    }


def _score_distribution_audit(
    calibration_events: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, float | int | bool]:
    scoped = calibration_events.dropna(subset=[score_column]).copy()
    if scoped.empty:
        return {
            "count": 0,
            "unique_score_count": 0,
            "tie_frequency": np.nan,
            "q75_threshold": np.nan,
            "q90_threshold": np.nan,
            "q10_threshold": np.nan,
            "top_quartile_count": 0,
            "top_decile_count": 0,
            "bottom_decile_count": 0,
            "expected_top_quartile_count": 0,
            "expected_top_decile_count": 0,
            "q75_boundary_tie_count": 0,
            "q90_boundary_tie_count": 0,
            "top_quartile_tie_expansion": 0,
            "top_decile_tie_expansion": 0,
            "top_decile_same_threshold_as_top_quartile": False,
        }

    score = pd.to_numeric(scoped[score_column], errors="coerce")
    q75 = float(score.quantile(0.75))
    q90 = float(score.quantile(0.90))
    q10 = float(score.quantile(0.10))
    top_quartile_count = int((score >= q75).sum())
    top_decile_count = int((score >= q90).sum())
    bottom_decile_count = int((score <= q10).sum())
    counts = score.value_counts(dropna=False)
    tied_observations = int(counts.loc[counts > 1].sum()) if not counts.empty else 0
    expected_top_quartile = int(np.ceil(len(scoped) * 0.25))
    expected_top_decile = int(np.ceil(len(scoped) * 0.10))
    return {
        "count": int(len(scoped)),
        "unique_score_count": int(score.nunique(dropna=True)),
        "tie_frequency": float(tied_observations / len(scoped)),
        "q75_threshold": q75,
        "q90_threshold": q90,
        "q10_threshold": q10,
        "top_quartile_count": top_quartile_count,
        "top_decile_count": top_decile_count,
        "bottom_decile_count": bottom_decile_count,
        "expected_top_quartile_count": expected_top_quartile,
        "expected_top_decile_count": expected_top_decile,
        "q75_boundary_tie_count": int((score == q75).sum()),
        "q90_boundary_tie_count": int((score == q90).sum()),
        "top_quartile_tie_expansion": int(top_quartile_count - expected_top_quartile),
        "top_decile_tie_expansion": int(top_decile_count - expected_top_decile),
        "top_decile_same_threshold_as_top_quartile": bool(np.isclose(q90, q75)),
    }


def _group_trade_metrics(
    calibration_events: pd.DataFrame,
    *,
    group_column: str,
) -> dict[str, dict[str, float | int]]:
    if calibration_events.empty or group_column not in calibration_events.columns:
        return {}
    scoped = calibration_events.dropna(subset=["realized_r"]).copy()
    if scoped.empty:
        return {}
    rows: dict[str, dict[str, float | int]] = {}
    for value in sorted(scoped[group_column].dropna().unique(), key=str):
        rows[str(value)] = _trade_metrics(scoped.loc[scoped[group_column] == value])
    return rows


def _cross_trade_metrics(
    calibration_events: pd.DataFrame,
    *,
    row_column: str,
    col_column: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    if calibration_events.empty:
        return {}
    scoped = calibration_events.dropna(subset=["realized_r"]).copy()
    if (
        scoped.empty
        or row_column not in scoped.columns
        or col_column not in scoped.columns
    ):
        return {}
    rows: dict[str, dict[str, dict[str, float | int]]] = {}
    for row_value in sorted(scoped[row_column].dropna().unique(), key=str):
        row_slice = scoped.loc[scoped[row_column] == row_value]
        rows[str(row_value)] = {}
        for col_value in sorted(row_slice[col_column].dropna().unique(), key=str):
            rows[str(row_value)][str(col_value)] = _trade_metrics(
                row_slice.loc[row_slice[col_column] == col_value]
            )
    return rows


def _enrich_detect_rows_with_context(
    detect_rows: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add structural-context and formation-quality features to detect-time rows.

    All features are computed strictly from data available at detect_idx — causal.
    New columns added:
      trend_aligned           — 1 if cluster side agrees with trend_state at detect_idx
      prior_bos_distance_atr  — abs distance from zone level to last same-direction BOS level
      prior_choch_distance_atr— same for CHoCH
      swing_magnitude_atr     — prominence_atr of the formation swing at last_member_origin_idx
      wick_ratio              — rejection-wick / total-range at last_member_origin_idx
      atr_percentile          — rolling ATR percentile rank (50-bar window) at detect_idx
      same_side_clusters_nearby_at_detect — count of other active same-side clusters within 2 ATR
    """
    if detect_rows.empty:
        return detect_rows

    out = detect_rows.copy()
    n_df = len(df)

    # Pre-extract numpy arrays from df for fast indexed access
    atr_arr = get_atr_array(df, 14)
    close_arr = df["close"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    open_arr = df["open"].to_numpy(dtype=float)

    # ── trend_aligned ──────────────────────────────────────────────────────────
    if "trend_state" in df.columns:
        trend_state_arr = df["trend_state"].to_numpy(dtype=float)
        aligned = np.full(len(out), np.nan)
        for idx_row, row in enumerate(out.itertuples(index=False)):
            d = int(row.detect_idx)
            if d < 0 or d >= n_df:
                continue
            ts = trend_state_arr[d]
            if not np.isfinite(ts):
                aligned[idx_row] = np.nan
            elif row.side == "eql":
                aligned[idx_row] = 1.0 if ts > 0 else 0.0
            else:  # eqh
                aligned[idx_row] = 1.0 if ts < 0 else 0.0
        out["trend_aligned"] = aligned
    else:
        out["trend_aligned"] = np.nan

    # ── prior_bos_distance_atr and prior_choch_distance_atr ───────────────────
    for event_type in ("bos", "choch"):
        bull_col = f"{event_type}_bull"
        bear_col = f"{event_type}_bear"
        level_col = f"{event_type}_level"
        if bull_col not in df.columns or level_col not in df.columns:
            out[f"prior_{event_type}_distance_atr"] = np.nan
            continue
        bull_flags = pd.to_numeric(df[bull_col], errors="coerce").fillna(0).astype(bool)
        bear_flags = (
            pd.to_numeric(
                df.get(bear_col, pd.Series(0, index=df.index)), errors="coerce"
            )
            .fillna(0)
            .astype(bool)
        )
        levels = pd.to_numeric(df[level_col], errors="coerce")
        last_bull_level = levels.where(bull_flags).ffill().to_numpy(dtype=float)
        last_bear_level = levels.where(bear_flags).ffill().to_numpy(dtype=float)
        dist = np.full(len(out), np.nan)
        for idx_row, row in enumerate(out.itertuples(index=False)):
            d = int(row.detect_idx)
            if d < 0 or d >= n_df:
                continue
            atr_d = float(atr_arr[d])
            if atr_d <= 0 or not np.isfinite(atr_d):
                continue
            zone_level = float(row.level)
            if row.side == "eql":
                ref = last_bull_level[d]
            else:
                ref = last_bear_level[d]
            if np.isfinite(ref):
                dist[idx_row] = abs(zone_level - ref) / atr_d
        out[f"prior_{event_type}_distance_atr"] = dist

    # ── swing_magnitude_atr and wick_ratio ────────────────────────────────────
    sh_prom_atr = df.get(
        "swing_high_prominence_atr", pd.Series(np.nan, index=df.index)
    ).to_numpy(dtype=float)
    sl_prom_atr = df.get(
        "swing_low_prominence_atr", pd.Series(np.nan, index=df.index)
    ).to_numpy(dtype=float)
    swing_mag = np.full(len(out), np.nan)
    wick_ratio = np.full(len(out), np.nan)
    eps = 1e-10
    for idx_row, row in enumerate(out.itertuples(index=False)):
        orig = (
            int(row.last_member_origin_idx)
            if np.isfinite(row.last_member_origin_idx)
            else -1
        )
        d = int(row.detect_idx)
        if orig < 0 or orig >= n_df:
            continue
        atr_o = float(atr_arr[orig])
        if atr_o <= 0 or not np.isfinite(atr_o):
            continue
        if row.side == "eqh":
            swing_mag[idx_row] = float(sh_prom_atr[orig])
            h, l, o, c = (
                float(high_arr[orig]),
                float(low_arr[orig]),
                float(open_arr[orig]),
                float(close_arr[orig]),
            )
            rng = h - l
            upper_wick = h - max(o, c)
            wick_ratio[idx_row] = upper_wick / (rng + eps) if rng > eps else np.nan
        else:
            swing_mag[idx_row] = float(sl_prom_atr[orig])
            h, l, o, c = (
                float(high_arr[orig]),
                float(low_arr[orig]),
                float(open_arr[orig]),
                float(close_arr[orig]),
            )
            rng = h - l
            lower_wick = min(o, c) - l
            wick_ratio[idx_row] = lower_wick / (rng + eps) if rng > eps else np.nan
    out["swing_magnitude_atr"] = swing_mag
    out["wick_ratio"] = wick_ratio

    # ── atr_percentile ────────────────────────────────────────────────────────
    atr_series = pd.Series(atr_arr)
    atr_pct_series = atr_series.rolling(50, min_periods=10).rank(pct=True)
    atr_pct_arr = atr_pct_series.to_numpy(dtype=float)
    atr_pct = np.full(len(out), np.nan)
    for idx_row, row in enumerate(out.itertuples(index=False)):
        d = int(row.detect_idx)
        if 0 <= d < n_df:
            atr_pct[idx_row] = atr_pct_arr[d]
    out["atr_percentile"] = atr_pct

    # ── same_side_clusters_nearby_at_detect ───────────────────────────────────
    # Count ranked active clusters (rank1, rank2) within 2 ATR of the new cluster's level.
    # At detect_idx the new cluster IS the primary active cluster; ranked slots hold others.
    crowding = np.full(len(out), np.nan)
    for idx_row, row in enumerate(out.itertuples(index=False)):
        d = int(row.detect_idx)
        if d < 0 or d >= n_df:
            continue
        atr_d = float(atr_arr[d])
        if atr_d <= 0 or not np.isfinite(atr_d):
            continue
        zone_level = float(row.level)
        radius = 2.0 * atr_d
        count = 0
        for rank in (1, 2):
            col = f"{row.side}_rank{rank}_active_level"
            if col not in df.columns:
                continue
            other_level = float(pd.to_numeric(df[col].iloc[d], errors="coerce"))
            if np.isfinite(other_level) and abs(zone_level - other_level) <= radius:
                count += 1
        crowding[idx_row] = float(count)
    out["same_side_clusters_nearby_at_detect"] = crowding

    return out


def build_equal_hl_calibration_event_table(
    df: pd.DataFrame,
    *,
    research_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    detect_rows = _detect_rows(df)
    detect_rows = _enrich_detect_rows_with_context(detect_rows, df)
    if research_table is None:
        research_table = build_equal_hl_research_table(df)
    trade_table = build_equal_hl_trade_table(df, research_table=research_table)
    if detect_rows.empty or trade_table.empty:
        return pd.DataFrame(
            columns=[
                "eqhl_event_id",
                "side",
                "detect_idx",
                "touch_count",
                "width_atr",
                "formation_delay",
                "structural_score",
                "tradeable_live_score",
                "width_component",
                "formation_delay_component",
                "realized_r",
                "exit_reason",
                "touch_bucket",
                "formation_delay_bucket",
            ]
        )
    merged = trade_table.merge(
        detect_rows.rename(columns={"event_id": "eqhl_event_id"}),
        on=["eqhl_event_id", "side", "detect_idx"],
        how="left",
        suffixes=("", "_detect"),
    )
    merged["touch_bucket"] = merged["touch_count"].map(_touch_bucket)
    merged["formation_delay_bucket"] = merged["formation_delay"].map(
        _formation_delay_bucket
    )
    return merged


def _add_forward_returns(
    active_snapshots: pd.DataFrame,
    df: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (3, 5),
) -> pd.DataFrame:
    """Add per-bar forward returns to each active snapshot row.

    fwd_r_N = direction × (close[t+N] - close[t]) / ATR[t]

    direction = +1 for EQL (demand — expect up), -1 for EQH (supply — expect down).
    This is causally clean: at bar t all inputs are known; outcome is future N bars.
    Rows near the end of the series where t+N is out-of-bounds get NaN.
    """
    if active_snapshots.empty:
        return active_snapshots

    out = active_snapshots.copy()
    close_arr = df["close"].to_numpy(dtype=float)
    atr_arr = get_atr_array(df, 14)
    n_df = len(df)

    # Build a positional index from df so we can map snapshot bar → df position
    df_reset = df.reset_index(drop=True)
    if "timestamp" in df_reset.columns and "timestamp" in out.columns:
        ts_to_pos = {ts: pos for pos, ts in enumerate(df_reset["timestamp"])}
    else:
        ts_to_pos = {}

    directions = out["side"].map({"eql": 1.0, "eqh": -1.0}).to_numpy(dtype=float)

    for N in horizons:
        fwd = np.full(len(out), np.nan)
        for row_i, row in enumerate(out.itertuples(index=False)):
            ts = getattr(row, "timestamp", None)
            pos = ts_to_pos.get(ts, None)
            if pos is None:
                continue
            if pos + N >= n_df:
                continue
            atr_t = float(atr_arr[pos])
            if atr_t <= 0 or not np.isfinite(atr_t):
                continue
            ret = (close_arr[pos + N] - close_arr[pos]) / atr_t
            fwd[row_i] = directions[row_i] * ret
        out[f"fwd_r_{N}"] = fwd

    return out


def build_equal_hl_active_snapshot_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for prefix in ("eqh", "eql"):
        mask = pd.to_numeric(df.get(f"{prefix}_active"), errors="coerce").fillna(0) == 1
        scoped = df.loc[mask].copy()
        if scoped.empty:
            continue
        scoped = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(scoped["timestamp"], utc=True),
                "side": prefix,
                "cluster_id": pd.to_numeric(
                    scoped[f"{prefix}_active_id"], errors="coerce"
                ),
                "touch_count": pd.to_numeric(
                    scoped[f"{prefix}_active_touch_count"], errors="coerce"
                ),
                "width_atr": pd.to_numeric(
                    scoped[f"{prefix}_active_width_atr"], errors="coerce"
                ),
                "active_age": pd.to_numeric(
                    scoped[f"{prefix}_active_age"], errors="coerce"
                ),
                "formation_delay": pd.to_numeric(
                    scoped[f"{prefix}_active_formation_delay"], errors="coerce"
                ),
                "distance_atr": pd.to_numeric(
                    scoped[f"{prefix}_active_distance_atr"], errors="coerce"
                ),
                "structural_score": pd.to_numeric(
                    scoped[f"{prefix}_active_structural_score"], errors="coerce"
                ),
                "tradeable_live_score": pd.to_numeric(
                    scoped[f"{prefix}_active_tradeable_live_score"], errors="coerce"
                ),
                "width_component": pd.to_numeric(
                    scoped[f"{prefix}_active_width_component"], errors="coerce"
                ),
                "formation_delay_component": pd.to_numeric(
                    scoped[f"{prefix}_active_formation_delay_component"],
                    errors="coerce",
                ),
                "bars_since_touch": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_bars_since_touch",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "touches_since_detect": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_touches_since_detect",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "close_tests": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_close_tests",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "max_pen_atr": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_max_pen_atr",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "far_edge_atr": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_far_edge_atr",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "inside_zone": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_inside_zone",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "signed_dist_atr": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_signed_dist_atr",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "same_side_count_nearby": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_same_side_count_nearby",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
                "nearest_same_side_atr": pd.to_numeric(
                    scoped.get(
                        f"{prefix}_active_nearest_same_side_atr",
                        pd.Series(dtype=float, index=scoped.index),
                    ),
                    errors="coerce",
                ),
            }
        )
        rows.append(scoped)
    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "side",
                "cluster_id",
                "touch_count",
                "width_atr",
                "active_age",
                "formation_delay",
                "distance_atr",
                "structural_score",
                "tradeable_live_score",
                "width_component",
                "formation_delay_component",
                "touch_bucket",
                "active_age_bucket",
                "freshness_bucket",
            ]
        )
    combined = pd.concat(rows, ignore_index=True)
    combined["touch_bucket"] = combined["touch_count"].map(_touch_bucket)
    combined["active_age_bucket"] = combined["active_age"].map(_active_age_bucket)
    combined["freshness_bucket"] = combined["active_age"].map(_freshness_bucket)
    combined = _add_forward_returns(combined, df)
    return combined


def summarize_equal_hl_score_candidate(
    calibration_events: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, object]:
    scoped = calibration_events.dropna(subset=[score_column, "realized_r"]).copy()
    if scoped.empty:
        return {
            "count": 0,
            "top_quartile_avg_r": np.nan,
            "top_decile_avg_r": np.nan,
            "bottom_decile_avg_r": np.nan,
            "top_quartile_win_rate": np.nan,
            "top_decile_win_rate": np.nan,
            "bottom_decile_win_rate": np.nan,
            "eqh_top_quartile_avg_r": np.nan,
            "eql_top_quartile_avg_r": np.nan,
            "eqh_top_quartile_win_rate": np.nan,
            "eql_top_quartile_win_rate": np.nan,
            "rank_corr": np.nan,
            "eqh_rank_corr": np.nan,
            "eql_rank_corr": np.nan,
            "win_rate_gap": np.nan,
        }
    q75 = float(scoped[score_column].quantile(0.75))
    q90 = float(scoped[score_column].quantile(0.90))
    q10 = float(scoped[score_column].quantile(0.10))
    top_quartile = scoped.loc[scoped[score_column] >= q75]
    top_decile = scoped.loc[scoped[score_column] >= q90]
    bottom_decile = scoped.loc[scoped[score_column] <= q10]
    top_q_metrics = _trade_metrics(top_quartile)
    top_d_metrics = _trade_metrics(top_decile)
    bottom_d_metrics = _trade_metrics(bottom_decile)
    eqh_top_q_metrics = _trade_metrics(top_quartile.loc[top_quartile["side"] == "eqh"])
    eql_top_q_metrics = _trade_metrics(top_quartile.loc[top_quartile["side"] == "eql"])
    return {
        "count": int(len(scoped)),
        "top_quartile_avg_r": top_q_metrics["avg_r"],
        "top_decile_avg_r": top_d_metrics["avg_r"],
        "bottom_decile_avg_r": bottom_d_metrics["avg_r"],
        "top_quartile_win_rate": top_q_metrics["win_rate"],
        "top_decile_win_rate": top_d_metrics["win_rate"],
        "bottom_decile_win_rate": bottom_d_metrics["win_rate"],
        "eqh_top_quartile_avg_r": eqh_top_q_metrics["avg_r"],
        "eql_top_quartile_avg_r": eql_top_q_metrics["avg_r"],
        "eqh_top_quartile_win_rate": eqh_top_q_metrics["win_rate"],
        "eql_top_quartile_win_rate": eql_top_q_metrics["win_rate"],
        "rank_corr": _spearman_rank_corr(scoped[score_column], scoped["realized_r"]),
        "eqh_rank_corr": _spearman_rank_corr(
            scoped.loc[scoped["side"] == "eqh", score_column],
            scoped.loc[scoped["side"] == "eqh", "realized_r"],
        ),
        "eql_rank_corr": _spearman_rank_corr(
            scoped.loc[scoped["side"] == "eql", score_column],
            scoped.loc[scoped["side"] == "eql", "realized_r"],
        ),
        "win_rate_gap": (
            float(top_q_metrics["win_rate"]) - float(bottom_d_metrics["win_rate"])
            if np.isfinite(top_q_metrics["win_rate"])
            and np.isfinite(bottom_d_metrics["win_rate"])
            else np.nan
        ),
    }


def _score_candidate_robustness(
    calibration_events: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, dict[str, object]]:
    scoped = calibration_events.dropna(subset=[score_column, "realized_r"]).copy()
    if scoped.empty:
        return {"subperiod": {}, "side": {}}
    return {
        "subperiod": {
            str(label): summarize_equal_hl_score_candidate(
                group, score_column=score_column
            )
            for label, group in scoped.groupby("subperiod", sort=True)
            if label != "unknown"
        },
        "side": {
            str(label): summarize_equal_hl_score_candidate(
                group, score_column=score_column
            )
            for label, group in scoped.groupby("side", sort=True)
        },
    }


def build_equal_hl_candidate_summary(
    calibration_events: pd.DataFrame,
    active_snapshots: pd.DataFrame,
    *,
    score_column: str,
    width_reference: float,
) -> dict[str, object]:
    metrics = summarize_equal_hl_score_candidate(
        calibration_events,
        score_column=score_column,
    )
    return {
        "event_count": int(len(calibration_events)),
        "eqh_count": int((calibration_events["side"] == "eqh").sum()),
        "eql_count": int((calibration_events["side"] == "eql").sum()),
        "formation_delay_mean": float(
            pd.to_numeric(calibration_events["formation_delay"], errors="coerce").mean()
        ),
        "formation_delay_bucket_counts": (
            calibration_events["formation_delay_bucket"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
            if "formation_delay_bucket" in calibration_events.columns
            else {}
        ),
        "active_age_mean": (
            float(pd.to_numeric(active_snapshots["active_age"], errors="coerce").mean())
            if not active_snapshots.empty
            else np.nan
        ),
        "active_age_bucket_counts": (
            active_snapshots["active_age_bucket"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
            if not active_snapshots.empty
            and "active_age_bucket" in active_snapshots.columns
            else {}
        ),
        "structural_score_mean": float(
            pd.to_numeric(
                calibration_events["structural_score"], errors="coerce"
            ).mean()
        ),
        "tradeable_live_score_mean": float(
            pd.to_numeric(calibration_events[score_column], errors="coerce").mean()
        ),
        "width_reference": float(width_reference),
        "score_distribution_audit": _score_distribution_audit(
            calibration_events, score_column=score_column
        ),
        "robustness": _score_candidate_robustness(
            calibration_events, score_column=score_column
        ),
        **metrics,
    }


def build_equal_hl_component_direction_audit(
    calibration_events: pd.DataFrame,
) -> pd.DataFrame:
    if calibration_events.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "component",
                "variant",
                "count",
                "top_quartile_avg_r",
                "top_decile_avg_r",
                "bottom_decile_avg_r",
                "top_quartile_win_rate",
                "bottom_decile_win_rate",
                "rank_corr",
                "variance_zero",
            ]
        )

    segments = {
        "overall": calibration_events,
        "eqh": calibration_events.loc[calibration_events["side"] == "eqh"],
        "eql": calibration_events.loc[calibration_events["side"] == "eql"],
        "2-touch": calibration_events.loc[
            calibration_events["touch_bucket"] == "2-touch"
        ],
        "3-touch+": calibration_events.loc[
            calibration_events["touch_bucket"] == "3-touch+"
        ],
    }
    if "subperiod" in calibration_events.columns:
        first_half = calibration_events.loc[
            calibration_events["subperiod"] == "first_half"
        ]
        second_half = calibration_events.loc[
            calibration_events["subperiod"] == "second_half"
        ]
        if not first_half.empty:
            segments["first_half"] = first_half
        if not second_half.empty:
            segments["second_half"] = second_half

    rows: list[dict[str, object]] = []
    component_names = (
        "width_component",
        "formation_delay_component",
        "wick_ratio_component",
        "atr_percentile_component",
        "distance_at_detect_component",
    )
    for segment_name, segment_df in segments.items():
        if segment_df.empty:
            continue
        for component in component_names:
            variance_zero = bool(
                pd.to_numeric(segment_df[component], errors="coerce").dropna().nunique()
                <= 1
            )
            for variant in EQHL_COMPONENT_SIGN_VARIANTS:
                scored = segment_df.copy()
                scored["candidate_score"] = _normalize_sign_variant(
                    scored[component], variant
                )
                metrics = summarize_equal_hl_score_candidate(
                    scored, score_column="candidate_score"
                )
                rows.append(
                    {
                        "segment": segment_name,
                        "component": component,
                        "variant": variant,
                        **metrics,
                        "variance_zero": variance_zero,
                    }
                )

    if not rows:
        return pd.DataFrame(rows)

    audit = pd.DataFrame(rows)

    # Annotate stability_classification per (component, variant) using time-split rank_corr.
    # Only populated for rows where we can get first_half and second_half values.
    def _lookup_corr(component: str, variant: str, segment: str) -> float:
        mask = (
            (audit["component"] == component)
            & (audit["variant"] == variant)
            & (audit["segment"] == segment)
        )
        hits = audit.loc[mask, "rank_corr"]
        return float(hits.iloc[0]) if len(hits) > 0 else np.nan

    stability_col: list[str] = []
    for _, row in audit.iterrows():
        comp = str(row["component"])
        var = str(row["variant"])
        overall_c = _lookup_corr(comp, var, "overall")
        first_c = _lookup_corr(comp, var, "first_half")
        second_c = _lookup_corr(comp, var, "second_half")
        stability_col.append(_classify_feature_stability(overall_c, first_c, second_c))
    audit["stability_classification"] = stability_col

    return audit


def build_eqhl_feature_monotonicity_audit(
    calibration_events: pd.DataFrame,
    *,
    n_buckets: int = EQHL_MONOTONICITY_N_BUCKETS,
) -> pd.DataFrame:
    """Audit raw features and processed components for monotone signal and temporal stability.

    For each (feature, segment) pair, reports:
      rank_corr, monotonicity_ratio, per-bucket avg_r, stability_classification (overall only).

    Raw features tested: formation_delay, width_atr, touch_count, structural_score, span.
    Component features tested: all five processed components.
    Segments tested: overall, eqh, eql, first_half, second_half.
    """
    if calibration_events.empty or "realized_r" not in calibration_events.columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "segment",
                "count",
                "rank_corr",
                "monotonicity_ratio",
                "stability_classification",
            ]
        )

    component_features = [
        "width_component",
        "formation_delay_component",
        "wick_ratio_component",
        "atr_percentile_component",
        "distance_at_detect_component",
    ]
    all_features = [
        f
        for f in EQHL_AUDIT_RAW_FEATURES + component_features
        if f in calibration_events.columns
    ]
    if not all_features:
        return pd.DataFrame()

    # Build segment map
    segment_map: dict[str, pd.DataFrame] = {"overall": calibration_events}
    if "side" in calibration_events.columns:
        for side_val in ("eqh", "eql"):
            subset = calibration_events.loc[calibration_events["side"] == side_val]
            if not subset.empty:
                segment_map[side_val] = subset
    if "subperiod" in calibration_events.columns:
        for period in ("first_half", "second_half"):
            subset = calibration_events.loc[calibration_events["subperiod"] == period]
            if not subset.empty:
                segment_map[period] = subset

    rows: list[dict[str, object]] = []
    for feature in all_features:
        # Pre-compute half correlations for stability classification
        first_corr = (
            _spearman_rank_corr(
                segment_map.get("first_half", pd.DataFrame()).get(
                    feature, pd.Series(dtype=float)
                ),
                segment_map.get("first_half", pd.DataFrame()).get(
                    "realized_r", pd.Series(dtype=float)
                ),
            )
            if "first_half" in segment_map
            else np.nan
        )
        second_corr = (
            _spearman_rank_corr(
                segment_map.get("second_half", pd.DataFrame()).get(
                    feature, pd.Series(dtype=float)
                ),
                segment_map.get("second_half", pd.DataFrame()).get(
                    "realized_r", pd.Series(dtype=float)
                ),
            )
            if "second_half" in segment_map
            else np.nan
        )
        overall_corr = _spearman_rank_corr(
            calibration_events[feature], calibration_events["realized_r"]
        )
        stability = _classify_feature_stability(overall_corr, first_corr, second_corr)

        for seg_name, seg_df in segment_map.items():
            if seg_df.empty:
                continue
            feat_vals = pd.to_numeric(
                seg_df.get(feature, pd.Series(dtype=float)), errors="coerce"
            )
            tgt_vals = pd.to_numeric(
                seg_df.get("realized_r", pd.Series(dtype=float)), errors="coerce"
            )
            rank_corr = _spearman_rank_corr(feat_vals, tgt_vals)
            mono = _monotonicity_ratio(feat_vals, tgt_vals, n_buckets=n_buckets)

            # Compute per-bucket avg_r and count
            frame = pd.DataFrame({"f": feat_vals, "r": tgt_vals}).dropna()
            bucket_stats: dict[str, object] = {}
            if len(frame) >= n_buckets * 3:
                try:
                    frame["bucket"] = pd.qcut(
                        frame["f"], n_buckets, labels=False, duplicates="drop"
                    )
                    for b_val, grp in frame.groupby("bucket"):
                        bucket_stats[f"bucket_{int(b_val)}_avg_r"] = float(
                            grp["r"].mean()
                        )
                        bucket_stats[f"bucket_{int(b_val)}_count"] = int(len(grp))
                except Exception:
                    pass

            rows.append(
                {
                    "feature": feature,
                    "segment": seg_name,
                    "count": int(len(frame)),
                    "rank_corr": rank_corr,
                    "monotonicity_ratio": mono,
                    # stability_classification only populated on overall row; others left blank
                    "stability_classification": (
                        stability if seg_name == "overall" else ""
                    ),
                    **bucket_stats,
                }
            )

    return pd.DataFrame(rows)


# ── Feature science: outcome enrichment + correlation matrices ──────────────────

# Outcome columns pulled from the research table for correlation analysis
EQHL_AUDIT_OUTCOME_HORIZON = 5
EQHL_AUDIT_OUTCOME_COLS = [
    f"eqhl_r_hold_{EQHL_AUDIT_OUTCOME_HORIZON}",
    f"eqhl_r_failed_{EQHL_AUDIT_OUTCOME_HORIZON}",
    f"eqhl_r_swept_{EQHL_AUDIT_OUTCOME_HORIZON}",
    f"eqhl_r_mfe_{EQHL_AUDIT_OUTCOME_HORIZON}_atr",
    f"eqhl_r_mae_{EQHL_AUDIT_OUTCOME_HORIZON}_atr",
]
EQHL_AUDIT_RAW_FEATURES = [
    # Detect-time features (Stage 1 — scored once at detect_idx)
    "formation_delay",
    "width_atr",
    "touch_count",
    "structural_score",
    "distance_atr_at_detect",
    "trend_aligned",
    "prior_bos_distance_atr",
    "prior_choch_distance_atr",
    "swing_magnitude_atr",
    "wick_ratio",
    "atr_percentile",
    "same_side_clusters_nearby_at_detect",
    # Active-state features (Stage 2 — updated each bar, audited against fwd_r_N)
    "active_age",
    "distance_atr",
    "bars_since_touch",
    "touches_since_detect",
    "max_pen_atr",
    "inside_zone",
    "signed_dist_atr",
    "nearest_same_side_atr",
]
EQHL_AUDIT_COMPONENT_FEATURES = [
    "width_component",
    "formation_delay_component",
    "wick_ratio_component",
    "atr_percentile_component",
    "distance_at_detect_component",
]


def build_eqhl_feature_science_table(
    calibration_events: pd.DataFrame,
    research_table: pd.DataFrame,
) -> pd.DataFrame:
    """Merge calibration events with key research-table outcomes for correlation analysis.

    Returns a wide table with:
    - All calibration event feature columns (raw + component)
    - realized_r
    - hold_H, failed_H, swept_H, mfe_H_atr, mae_H_atr at the audit horizon
    - side, subperiod, formation_delay_bucket, touch_bucket
    """
    if calibration_events.empty:
        return calibration_events.copy()
    if research_table.empty:
        return calibration_events.copy()

    outcome_cols_present = [
        c for c in EQHL_AUDIT_OUTCOME_COLS if c in research_table.columns
    ]
    research_slim = research_table[["eqhl_event_id"] + outcome_cols_present].copy()

    merged = calibration_events.merge(
        research_slim,
        on="eqhl_event_id",
        how="left",
        suffixes=("", "_r"),
    )
    return merged


def build_eqhl_correlation_matrices(
    feature_science_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compute Spearman correlation matrices from the feature science table.

    Returns:
      feature_vs_outcome  — raw features × outcome metrics (realized_r + research outcomes)
      feature_vs_feature  — raw features × all features (pairwise)
      component_vs_outcome — processed components × outcome metrics
    """
    raw_cols = [
        c for c in EQHL_AUDIT_RAW_FEATURES if c in feature_science_table.columns
    ]
    comp_cols = [
        c for c in EQHL_AUDIT_COMPONENT_FEATURES if c in feature_science_table.columns
    ]
    outcome_cols = ["realized_r"] + [
        c for c in EQHL_AUDIT_OUTCOME_COLS if c in feature_science_table.columns
    ]
    all_feature_cols = raw_cols + comp_cols

    numeric = feature_science_table[all_feature_cols + outcome_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    def _spearman_matrix(row_cols: list[str], col_cols: list[str]) -> pd.DataFrame:
        result = pd.DataFrame(index=row_cols, columns=col_cols, dtype=float)
        for r in row_cols:
            for c in col_cols:
                if r == c:
                    result.loc[r, c] = 1.0
                    continue
                r_vals = numeric[r].rename("_r")
                c_vals = numeric[c].rename("_c")
                pair = pd.concat([r_vals, c_vals], axis=1).dropna()
                if (
                    len(pair) < 3
                    or pair["_r"].nunique() <= 1
                    or pair["_c"].nunique() <= 1
                ):
                    result.loc[r, c] = np.nan
                else:
                    r_rank = pair["_r"].rank(method="average")
                    c_rank = pair["_c"].rank(method="average")
                    result.loc[r, c] = float(r_rank.corr(c_rank))
        return result

    return {
        "feature_vs_outcome": _spearman_matrix(raw_cols, outcome_cols),
        "component_vs_outcome": _spearman_matrix(comp_cols, outcome_cols),
        "feature_vs_feature": _spearman_matrix(all_feature_cols, all_feature_cols),
    }


def build_eqhl_univariate_ic_table(
    feature_science_table: pd.DataFrame,
) -> pd.DataFrame:
    """One-row-per-feature summary table: rank_corr across all segments + stability.

    Columns: feature, overall, first_half, second_half, eqh, eql, stability_classification.
    This is the pivot view of build_eqhl_feature_monotonicity_audit for clean printing.
    """
    audit = build_eqhl_feature_monotonicity_audit(feature_science_table)
    if audit.empty:
        return pd.DataFrame()

    all_features = audit["feature"].unique().tolist()
    segments = ["overall", "first_half", "second_half", "eqh", "eql"]
    rows = []
    for feature in all_features:
        feature_rows = audit.loc[audit["feature"] == feature]
        row: dict[str, object] = {"feature": feature}
        for seg in segments:
            seg_row = feature_rows.loc[feature_rows["segment"] == seg]
            row[seg] = (
                float(seg_row["rank_corr"].iloc[0]) if not seg_row.empty else np.nan
            )
        # Stability from overall row
        overall_row = feature_rows.loc[feature_rows["segment"] == "overall"]
        row["monotonicity"] = (
            float(overall_row["monotonicity_ratio"].iloc[0])
            if not overall_row.empty
            else np.nan
        )
        row["stability"] = (
            str(overall_row["stability_classification"].iloc[0])
            if not overall_row.empty
            else ""
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def equal_hl_score_candidate_passes_temporal_stability(
    robustness: dict[str, dict[str, object]],
    *,
    min_half_rank_corr: float = EQHL_STABILITY_MIN_HALF_RANK_CORR,
    min_half_top_vs_bottom: float = EQHL_STABILITY_MIN_HALF_TOP_VS_BOTTOM,
) -> bool:
    """Return True if the candidate score is directionally stable in both time halves.

    Requires:
    - rank_corr >= min_half_rank_corr in first_half AND second_half
    - top_quartile_avg_r - bottom_decile_avg_r >= min_half_top_vs_bottom in both halves
    """
    subperiod = robustness.get("subperiod", {})
    for period in ("first_half", "second_half"):
        m = subperiod.get(period, {})
        if not m or not m.get("count", 0):
            return False
        rank_corr = float(m.get("rank_corr", np.nan))
        if not np.isfinite(rank_corr) or rank_corr < min_half_rank_corr:
            return False
        tq_avg_r = float(m.get("top_quartile_avg_r", np.nan))
        bd_avg_r = float(m.get("bottom_decile_avg_r", np.nan))
        if not np.isfinite(tq_avg_r) or not np.isfinite(bd_avg_r):
            return False
        if tq_avg_r - bd_avg_r < min_half_top_vs_bottom:
            return False
    return True


def equal_hl_score_candidate_passes_acceptance(
    candidate_summary: dict[str, object],
    *,
    structural_benchmark: dict[str, object],
) -> bool:
    top_quartile_avg_r = float(candidate_summary.get("top_quartile_avg_r", np.nan))
    top_decile_avg_r = float(candidate_summary.get("top_decile_avg_r", np.nan))
    bottom_decile_avg_r = float(candidate_summary.get("bottom_decile_avg_r", np.nan))
    top_quartile_win_rate = float(
        candidate_summary.get("top_quartile_win_rate", np.nan)
    )
    bottom_decile_win_rate = float(
        candidate_summary.get("bottom_decile_win_rate", np.nan)
    )
    rank_corr = float(candidate_summary.get("rank_corr", np.nan))
    eqh_rank_corr = float(candidate_summary.get("eqh_rank_corr", np.nan))
    eql_rank_corr = float(candidate_summary.get("eql_rank_corr", np.nan))
    structural_top_quartile = float(
        structural_benchmark.get("top_quartile_avg_r", np.nan)
    )

    # Top-quartile must be profitable
    if (
        not np.isfinite(top_quartile_avg_r)
        or top_quartile_avg_r < EQHL_ACCEPT_MIN_TOP_QUARTILE_AVG_R
    ):
        return False
    # Top-decile must be at least as good as top-quartile (not a flat score distribution)
    if EQHL_ACCEPT_TOP_DECILE_IMPROVES_QUARTILE:
        if not np.isfinite(top_decile_avg_r) or top_decile_avg_r < top_quartile_avg_r:
            return False
    if not np.isfinite(bottom_decile_avg_r):
        return False
    # Top must materially beat bottom
    if (top_quartile_avg_r - bottom_decile_avg_r) < EQHL_ACCEPT_MIN_TOP_VS_BOTTOM_AVG_R:
        return False
    # Win-rate separation
    if not np.isfinite(top_quartile_win_rate) or not np.isfinite(
        bottom_decile_win_rate
    ):
        return False
    if (top_quartile_win_rate - bottom_decile_win_rate) < EQHL_ACCEPT_MIN_WIN_RATE_GAP:
        return False
    # Positive rank correlation
    if not np.isfinite(rank_corr) or rank_corr < EQHL_ACCEPT_MIN_RANK_CORR:
        return False
    # Neither side may collapse
    if np.isfinite(eqh_rank_corr) and eqh_rank_corr < EQHL_ACCEPT_SIDE_RANK_CORR_FLOOR:
        return False
    if np.isfinite(eql_rank_corr) and eql_rank_corr < EQHL_ACCEPT_SIDE_RANK_CORR_FLOOR:
        return False
    if (
        np.isfinite(structural_top_quartile)
        and top_quartile_avg_r < structural_top_quartile
    ):
        return False
    return True


def _candidate_objective_key(
    candidate_summary: dict[str, object],
) -> tuple[object, ...]:
    return tuple(candidate_summary.get(key) for key in EQHL_V2_OBJECTIVE_KEYS)


def _candidate_weight_rows(
    *,
    step: float,
) -> list[dict[str, float]]:
    """Generate weight combinations for the 5 detect-time score components.

    All 5 components sum to 1.0; weights are multiples of `step`.
    Components: width_component, formation_delay_component, wick_ratio_component,
                atr_percentile_component, distance_at_detect_component.

    Excluded from search (fixed at 0.0):
      - touch_component      ≡ touch_count (correlation 1.0 — fully redundant)
      - active_age_component is always 1.0 at detect time (zero variance)
      - structural_component  negative 2H IC (-0.073) — actively hurts score stability
    """
    units = int(round(1.0 / step))
    rows: list[dict[str, float]] = []
    for w_units in range(units + 1):
        for fd_units in range(units - w_units + 1):
            for wr_units in range(units - w_units - fd_units + 1):
                for ap_units in range(units - w_units - fd_units - wr_units + 1):
                    d_units = units - w_units - fd_units - wr_units - ap_units
                    rows.append(
                        {
                            "width_component": w_units * step,
                            "formation_delay_component": fd_units * step,
                            "wick_ratio_component": wr_units * step,
                            "atr_percentile_component": ap_units * step,
                            "distance_at_detect_component": d_units * step,
                        }
                    )
    return rows


def search_equal_hl_tradeable(
    calibration_events: pd.DataFrame,
    *,
    structural_benchmark: dict[str, object],
) -> dict[str, object]:
    if calibration_events.empty:
        empty = pd.DataFrame()
        return {
            "component_direction_audit": empty,
            "coarse_candidates": empty,
            "refined_candidates": empty,
            "selected_candidate": None,
            "width_reference": float(TRADEABLE_WIDTH_REF),
        }

    prepared_events, width_reference = _prepare_calibration_events(calibration_events)
    component_audit = build_equal_hl_component_direction_audit(prepared_events)
    # Fixed-zero components — skip sign search (irrelevant)
    chosen_signs = {
        "touch_component": "current",
        "active_age_component": "current",
        "structural_component": "current",
    }
    for component in (
        "width_component",
        "formation_delay_component",
        "wick_ratio_component",
        "atr_percentile_component",
        "distance_at_detect_component",
    ):
        scoped = component_audit.loc[
            (component_audit["segment"] == "overall")
            & (component_audit["component"] == component)
        ].copy()
        scoped = scoped.loc[scoped["variance_zero"] == False]  # noqa: E712
        if scoped.empty:
            chosen_signs[component] = "current"
            continue
        scoped = scoped.sort_values(
            ["top_decile_avg_r", "top_quartile_avg_r", "rank_corr"],
            ascending=[False, False, False],
        )
        chosen_signs[component] = str(scoped.iloc[0]["variant"])

    coarse_rows: list[dict[str, object]] = []
    for weights in _candidate_weight_rows(step=EQHL_CALIBRATION_WEIGHT_STEP):
        candidate = EqhlScoreCandidate(
            name="coarse",
            signs=chosen_signs,
            weights=weights,
        )
        scoped = prepared_events.copy()
        scoped["tradeable_live_score"] = _score_with_candidate(
            scoped,
            candidate=candidate,
        )
        metrics = summarize_equal_hl_score_candidate(
            scoped, score_column="tradeable_live_score"
        )
        robustness = _score_candidate_robustness(
            scoped, score_column="tradeable_live_score"
        )
        first_m = robustness.get("subperiod", {}).get("first_half", {})
        second_m = robustness.get("subperiod", {}).get("second_half", {})
        metrics.update(
            {
                **weights,
                "accepted": equal_hl_score_candidate_passes_acceptance(
                    metrics,
                    structural_benchmark=structural_benchmark,
                ),
                "temporally_stable": equal_hl_score_candidate_passes_temporal_stability(
                    robustness
                ),
                "first_half_rank_corr": float(first_m.get("rank_corr", np.nan)),
                "second_half_rank_corr": float(second_m.get("rank_corr", np.nan)),
                "first_half_top_q_avg_r": float(
                    first_m.get("top_quartile_avg_r", np.nan)
                ),
                "second_half_top_q_avg_r": float(
                    second_m.get("top_quartile_avg_r", np.nan)
                ),
                "first_half_win_rate_gap": float(first_m.get("win_rate_gap", np.nan)),
                "second_half_win_rate_gap": float(second_m.get("win_rate_gap", np.nan)),
            }
        )
        coarse_rows.append(metrics)

    coarse_df = pd.DataFrame(coarse_rows).sort_values(
        [
            "temporally_stable",
            "accepted",
            "top_decile_avg_r",
            "top_quartile_avg_r",
            "rank_corr",
        ],
        ascending=[False, False, False, False, False],
    )
    if coarse_df.empty:
        return {
            "component_direction_audit": component_audit,
            "coarse_candidates": coarse_df,
            "refined_candidates": pd.DataFrame(),
            "selected_candidate": None,
            "width_reference": float(width_reference),
        }
    best_coarse = coarse_df.iloc[0]

    refined_rows: list[dict[str, object]] = []
    for deltas in itertools.product(
        np.arange(
            -EQHL_CALIBRATION_REFINED_DELTA,
            EQHL_CALIBRATION_REFINED_DELTA + EPS,
            EQHL_CALIBRATION_REFINED_STEP,
        ),
        repeat=5,  # 5 active components: width, formation_delay, wick_ratio, atr_pct, distance
    ):
        weights = {
            "width_component": max(
                float(best_coarse["width_component"] + deltas[0]), 0.0
            ),
            "formation_delay_component": max(
                float(best_coarse["formation_delay_component"] + deltas[1]), 0.0
            ),
            "wick_ratio_component": max(
                float(best_coarse["wick_ratio_component"] + deltas[2]), 0.0
            ),
            "atr_percentile_component": max(
                float(best_coarse["atr_percentile_component"] + deltas[3]), 0.0
            ),
            "distance_at_detect_component": max(
                float(best_coarse["distance_at_detect_component"] + deltas[4]), 0.0
            ),
        }
        total = sum(weights.values())
        if total <= 0:
            continue
        weights = {key: value / total for key, value in weights.items()}
        candidate = EqhlScoreCandidate(
            name="refined",
            signs=chosen_signs,
            weights=weights,
        )
        scoped = prepared_events.copy()
        scoped["tradeable_live_score"] = _score_with_candidate(
            scoped,
            candidate=candidate,
        )
        metrics = summarize_equal_hl_score_candidate(
            scoped, score_column="tradeable_live_score"
        )
        robustness = _score_candidate_robustness(
            scoped, score_column="tradeable_live_score"
        )
        first_m = robustness.get("subperiod", {}).get("first_half", {})
        second_m = robustness.get("subperiod", {}).get("second_half", {})
        metrics.update(
            {
                **weights,
                "accepted": equal_hl_score_candidate_passes_acceptance(
                    metrics,
                    structural_benchmark=structural_benchmark,
                ),
                "temporally_stable": equal_hl_score_candidate_passes_temporal_stability(
                    robustness
                ),
                "first_half_rank_corr": float(first_m.get("rank_corr", np.nan)),
                "second_half_rank_corr": float(second_m.get("rank_corr", np.nan)),
                "first_half_top_q_avg_r": float(
                    first_m.get("top_quartile_avg_r", np.nan)
                ),
                "second_half_top_q_avg_r": float(
                    second_m.get("top_quartile_avg_r", np.nan)
                ),
                "first_half_win_rate_gap": float(first_m.get("win_rate_gap", np.nan)),
                "second_half_win_rate_gap": float(second_m.get("win_rate_gap", np.nan)),
            }
        )
        refined_rows.append(metrics)

    refined_df = pd.DataFrame(refined_rows).sort_values(
        [
            "temporally_stable",
            "accepted",
            "top_decile_avg_r",
            "top_quartile_avg_r",
            "rank_corr",
        ],
        ascending=[False, False, False, False, False],
    )
    selected_source = refined_df if not refined_df.empty else coarse_df
    selected_row = (
        selected_source.iloc[0].to_dict() if not selected_source.empty else None
    )
    if selected_row is not None:
        selected_row["signs"] = chosen_signs
    return {
        "component_direction_audit": component_audit.reset_index(drop=True),
        "coarse_candidates": coarse_df.reset_index(drop=True),
        "refined_candidates": refined_df.reset_index(drop=True),
        "selected_candidate": selected_row,
        "width_reference": float(width_reference),
    }


def extract_equal_hl_tables(
    df: pd.DataFrame,
    *,
    research_table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    detect_rows = _detect_rows(df)
    if research_table is None:
        research_table = build_equal_hl_research_table(df)
    calibration_events = build_equal_hl_calibration_event_table(
        df,
        research_table=research_table,
    )
    calibration_events, width_reference = _prepare_calibration_events(
        calibration_events
    )
    active_snapshots = _prepare_active_snapshots(
        build_equal_hl_active_snapshot_table(df),
        width_reference=width_reference,
    )
    structural_detect_metrics = summarize_equal_hl_score_candidate(
        calibration_events,
        score_column="structural_score",
    )
    score_calibration = search_equal_hl_tradeable(
        calibration_events,
        structural_benchmark=structural_detect_metrics,
    )
    selected_v2 = score_calibration["selected_candidate"]
    calibration_events_export = calibration_events.copy()
    if selected_v2 is not None:
        candidate = EqhlScoreCandidate(
            name="v2",
            signs=dict(selected_v2["signs"]),
            weights={
                component: float(selected_v2[component])
                for component in (
                    "width_component",
                    "formation_delay_component",
                    "wick_ratio_component",
                    "atr_percentile_component",
                    "distance_at_detect_component",
                )
            },
        )
        calibration_events_export["tradeable_live_score"] = _score_with_candidate(
            calibration_events_export,
            candidate=candidate,
        )
        if not active_snapshots.empty:
            active_snapshots = active_snapshots.copy()
            active_snapshots["tradeable_live_score"] = _score_with_candidate(
                active_snapshots,
                candidate=candidate,
            )
    else:
        calibration_events_export["tradeable_live_score"] = np.nan
        if not active_snapshots.empty:
            active_snapshots = active_snapshots.copy()
            active_snapshots["tradeable_live_score"] = np.nan

    strongest = (
        calibration_events_export.sort_values(
            ["structural_score", "tradeable_live_score"],
            ascending=[False, False],
        )
        .head(20)
        .reset_index(drop=True)
    )
    structurally_strongest = (
        calibration_events_export.sort_values(
            ["structural_score", "tradeable_live_score"],
            ascending=[False, False],
        )
        .head(20)
        .reset_index(drop=True)
    )
    freshest_tradeable = (
        calibration_events_export.sort_values(
            ["tradeable_live_score", "formation_delay", "width_atr"],
            ascending=[False, True, True],
        )
        .head(20)
        .reset_index(drop=True)
    )
    borderline_wide = (
        detect_rows.sort_values(["width_atr", "touch_count"], ascending=[False, False])
        .head(20)
        .reset_index(drop=True)
    )

    if not active_snapshots.empty:
        stale_source = active_snapshots.copy()
        score_column = "tradeable_live_score"
        stale_source["tradeable_live_score_selected"] = pd.to_numeric(
            stale_source[score_column], errors="coerce"
        )
        stale_source["score"] = stale_source["tradeable_live_score_selected"]
        stale_source["age"] = pd.to_numeric(stale_source["active_age"], errors="coerce")
        stale_source["level"] = np.nan
        stale = (
            stale_source[
                [
                    "timestamp",
                    "side",
                    "cluster_id",
                    "level",
                    "width_atr",
                    "touch_count",
                    "structural_score",
                    "tradeable_live_score_selected",
                    "score",
                    "age",
                    "formation_delay",
                ]
            ]
            .rename(columns={"tradeable_live_score_selected": "tradeable_live_score"})
            .sort_values(["age", "score"], ascending=[False, False])
            .head(20)
            .reset_index(drop=True)
        )
    else:
        stale = pd.DataFrame(
            columns=[
                "timestamp",
                "side",
                "cluster_id",
                "level",
                "width_atr",
                "touch_count",
                "structural_score",
                "tradeable_live_score",
                "score",
                "age",
                "formation_delay",
            ]
        )

    research_strongest = (
        research_table.sort_values(
            ["eqhl_r_tradeable_score", "eqhl_r_quality_score"],
            ascending=[False, False],
        )
        .head(20)
        .reset_index(drop=True)
    )
    research_failures = (
        research_table.sort_values(
            ["eqhl_r_failure_severity_score", "eqhl_r_mae_5_atr"],
            ascending=[False, False],
        )
        .head(20)
        .reset_index(drop=True)
    )

    feature_monotonicity_audit = build_eqhl_feature_monotonicity_audit(
        calibration_events_export
    )

    return {
        "strongest_clusters": strongest,
        "structurally_strongest_clusters": structurally_strongest,
        "fresh_tradeable_clusters": freshest_tradeable,
        "borderline_wide_clusters": borderline_wide,
        "old_stale_clusters": stale,
        "calibration_events": calibration_events_export,
        "active_snapshots": active_snapshots,
        "component_direction_audit": score_calibration[
            "component_direction_audit"
        ].copy(),
        "feature_monotonicity_audit": feature_monotonicity_audit,
        "coarse_score_candidates": score_calibration["coarse_candidates"].copy(),
        "refined_score_candidates": score_calibration["refined_candidates"].copy(),
        "width_reference": pd.DataFrame(
            [
                {
                    "width_reference": width_reference,
                    "width_reference_percentile": EQHL_WIDTH_REF_PERCENTILE,
                }
            ]
        ),
        "strongest_research": research_strongest,
        "strongest_failures": research_failures,
    }


def summarize_equal_hl(
    df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    analysis_df = full_df if full_df is not None else df
    if research_table is None:
        research_table = build_equal_hl_research_table(analysis_df)
    trade_table = build_equal_hl_trade_table(analysis_df, research_table=research_table)
    calibration_events = build_equal_hl_calibration_event_table(
        analysis_df,
        research_table=research_table,
    )
    calibration_events, width_reference = _prepare_calibration_events(
        calibration_events
    )
    active_snapshots = _prepare_active_snapshots(
        build_equal_hl_active_snapshot_table(analysis_df),
        width_reference=width_reference,
    )

    eqh_detect_count = int(
        pd.to_numeric(analysis_df.get("eqh_detect_flag"), errors="coerce")
        .fillna(0)
        .sum()
    )
    eql_detect_count = int(
        pd.to_numeric(analysis_df.get("eql_detect_flag"), errors="coerce")
        .fillna(0)
        .sum()
    )
    detect_rows = _detect_rows(analysis_df)
    eqh_active_counts = pd.to_numeric(
        analysis_df.get("eqh_active_count"), errors="coerce"
    )
    eql_active_counts = pd.to_numeric(
        analysis_df.get("eql_active_count"), errors="coerce"
    )
    pooled_active_counts = pd.concat(
        [eqh_active_counts, eql_active_counts],
        axis=0,
        ignore_index=True,
    )
    detect_structural = (
        detect_rows["structural_score"]
        if not detect_rows.empty
        else pd.Series(dtype=float)
    )
    detect_tradeable = (
        detect_rows["tradeable_live_score"]
        if not detect_rows.empty
        else pd.Series(dtype=float)
    )
    detect_formation_delay = (
        detect_rows["formation_delay"]
        if not detect_rows.empty
        else pd.Series(dtype=float)
    )
    active_age = pd.concat(
        [
            pd.to_numeric(analysis_df.get("eqh_active_age"), errors="coerce"),
            pd.to_numeric(analysis_df.get("eql_active_age"), errors="coerce"),
        ],
        ignore_index=True,
    )
    active_tradeable = pd.concat(
        [
            pd.to_numeric(
                analysis_df.get("eqh_active_tradeable_live_score"), errors="coerce"
            ),
            pd.to_numeric(
                analysis_df.get("eql_active_tradeable_live_score"), errors="coerce"
            ),
        ],
        ignore_index=True,
    )
    active_structural = pd.concat(
        [
            pd.to_numeric(
                analysis_df.get("eqh_active_structural_score"), errors="coerce"
            ),
            pd.to_numeric(
                analysis_df.get("eql_active_structural_score"), errors="coerce"
            ),
        ],
        ignore_index=True,
    )
    structural_detect_metrics = summarize_equal_hl_score_candidate(
        calibration_events,
        score_column="structural_score",
    )
    score_calibration = search_equal_hl_tradeable(
        calibration_events,
        structural_benchmark=structural_detect_metrics,
    )
    selected_v2 = score_calibration["selected_candidate"]
    if selected_v2 is not None:
        candidate = EqhlScoreCandidate(
            name="v2",
            signs=dict(selected_v2["signs"]),
            weights={
                component: float(selected_v2[component])
                for component in (
                    "width_component",
                    "formation_delay_component",
                    "wick_ratio_component",
                    "atr_percentile_component",
                    "distance_at_detect_component",
                )
            },
        )
        calibration_events = calibration_events.copy()
        calibration_events["tradeable_live_score"] = _score_with_candidate(
            calibration_events,
            candidate=candidate,
        )
        if not active_snapshots.empty:
            active_snapshots = active_snapshots.copy()
            active_snapshots["tradeable_live_score"] = _score_with_candidate(
                active_snapshots,
                candidate=candidate,
            )
        score_metrics = summarize_equal_hl_score_candidate(
            calibration_events,
            score_column="tradeable_live_score",
        )
    else:
        score_metrics = summarize_equal_hl_score_candidate(
            pd.DataFrame(columns=[*calibration_events.columns, "tradeable_live_score"]),
            score_column="tradeable_live_score",
        )
    selected_v2_accepted = bool(selected_v2 is not None and selected_v2.get("accepted"))
    candidate_summary = (
        build_equal_hl_candidate_summary(
            calibration_events,
            active_snapshots,
            score_column="tradeable_live_score",
            width_reference=width_reference,
        )
        if selected_v2_accepted
        else {
            "event_count": int(eqh_detect_count + eql_detect_count),
            "eqh_count": int(eqh_detect_count),
            "eql_count": int(eql_detect_count),
            "formation_delay_mean": (
                float(
                    pd.to_numeric(
                        calibration_events["formation_delay"], errors="coerce"
                    ).mean()
                )
                if not calibration_events.empty
                else np.nan
            ),
            "formation_delay_bucket_counts": (
                calibration_events["formation_delay_bucket"]
                .value_counts(dropna=False)
                .sort_index()
                .astype(int)
                .to_dict()
                if "formation_delay_bucket" in calibration_events.columns
                else {}
            ),
            "active_age_mean": (
                float(
                    pd.to_numeric(
                        active_snapshots["active_age"], errors="coerce"
                    ).mean()
                )
                if not active_snapshots.empty
                else np.nan
            ),
            "active_age_bucket_counts": (
                active_snapshots["active_age_bucket"]
                .value_counts(dropna=False)
                .sort_index()
                .astype(int)
                .to_dict()
                if not active_snapshots.empty
                and "active_age_bucket" in active_snapshots.columns
                else {}
            ),
            "structural_score_mean": (
                float(
                    pd.to_numeric(
                        calibration_events["structural_score"], errors="coerce"
                    ).mean()
                )
                if not calibration_events.empty
                else np.nan
            ),
            "tradeable_live_score_mean": np.nan,
            "width_reference": float(width_reference),
            "score_distribution_audit": _score_distribution_audit(
                calibration_events, score_column="tradeable_live_score"
            ),
            "robustness": {
                "subperiod": {},
                "side": {},
            },
            "count": int(len(calibration_events)),
            "top_quartile_avg_r": np.nan,
            "top_decile_avg_r": np.nan,
            "bottom_decile_avg_r": np.nan,
            "top_quartile_win_rate": np.nan,
            "top_decile_win_rate": np.nan,
            "bottom_decile_win_rate": np.nan,
            "eqh_top_quartile_avg_r": np.nan,
            "eql_top_quartile_avg_r": np.nan,
            "eqh_top_quartile_win_rate": np.nan,
            "eql_top_quartile_win_rate": np.nan,
            "rank_corr": np.nan,
            "eqh_rank_corr": np.nan,
            "eql_rank_corr": np.nan,
            "win_rate_gap": np.nan,
        }
    )
    detect_trade_metrics = _score_bucket_metrics(
        trade_table,
        score_column="tradeable_live_score",
    )
    structural_trade_metrics = _score_bucket_metrics(
        trade_table,
        score_column="structural_score",
    )
    summary = {
        "rows": int(len(analysis_df)),
        "eqh_detect_count": eqh_detect_count,
        "eql_detect_count": eql_detect_count,
        "total_detect_count": eqh_detect_count + eql_detect_count,
        "unique_event_ids": bool(
            detect_rows["event_id"].dropna().nunique() == len(detect_rows)
            if not detect_rows.empty
            else True
        ),
        "one_row_per_detect_idx": bool(
            detect_rows[["detect_idx", "side"]].drop_duplicates().shape[0]
            == len(detect_rows)
            if not detect_rows.empty
            else True
        ),
        "active_count_stats_scope": "pooled_eqh_eql_active_count_observations",
        "eqh_active_count_stats": _continuous_stats(eqh_active_counts),
        "eql_active_count_stats": _continuous_stats(eql_active_counts),
        "pooled_active_count_stats": _continuous_stats(pooled_active_counts),
        "active_count_stats": _continuous_stats(pooled_active_counts),
        "touch_count_stats": (
            _continuous_stats(detect_rows["touch_count"])
            if not detect_rows.empty
            else _continuous_stats(pd.Series(dtype=float))
        ),
        "width_atr_stats": (
            _continuous_stats(detect_rows["width_atr"])
            if not detect_rows.empty
            else _continuous_stats(pd.Series(dtype=float))
        ),
        "formation_delay_stats": _continuous_stats(detect_formation_delay),
        "active_age_stats": _continuous_stats(active_age),
        "structural_score_stats": _continuous_stats(detect_structural),
        "tradeable_live_score_stats": _continuous_stats(detect_tradeable),
        "score_stats": (
            _continuous_stats(detect_rows["score"])
            if not detect_rows.empty
            else _continuous_stats(pd.Series(dtype=float))
        ),
        "active_structural_score_stats": _continuous_stats(active_structural),
        "active_tradeable_live_score_stats": _continuous_stats(active_tradeable),
        "width_reference": width_reference,
        "width_reference_percentile": EQHL_WIDTH_REF_PERCENTILE,
        "tradeable_live_score_tier_counts": detect_trade_metrics["tier_counts"],
        "structural_score_tier_counts": structural_trade_metrics["tier_counts"],
        "top_quartile_trade_metrics": detect_trade_metrics["top_quartile"],
        "top_decile_trade_metrics": detect_trade_metrics["top_decile"],
        "bottom_decile_trade_metrics": detect_trade_metrics["bottom_decile"],
        "structural_top_quartile_trade_metrics": structural_trade_metrics[
            "top_quartile"
        ],
        "side_split_trade_metrics": {
            side: _trade_metrics(trade_table.loc[trade_table["side"] == side])
            for side in ("eqh", "eql")
        },
        "calibration_truths": {
            "broader_inventory_worked": True,
            "structural_score_not_problem": bool(
                np.isfinite(structural_detect_metrics["top_quartile_avg_r"])
                and structural_detect_metrics["top_quartile_avg_r"] > 0
            ),
            "defaults_frozen_for_repair": True,
        },
        "selector_mode": EQHL_SELECTOR_MODE_CALIBRATION,
        "structural_detect_trade_metrics": structural_detect_metrics,
        "tradeable_live_score_metrics": score_metrics,
        "tradeable_live_score_distribution_audit": candidate_summary[
            "score_distribution_audit"
        ],
        "tradeable_live_score_robustness": candidate_summary["robustness"],
        "touch_population_stats": {
            "two_touch_share": (
                float(
                    (
                        pd.to_numeric(detect_rows["touch_count"], errors="coerce") <= 2
                    ).mean()
                )
                if not detect_rows.empty
                else np.nan
            ),
            "three_touch_plus_count": (
                int(
                    (
                        pd.to_numeric(detect_rows["touch_count"], errors="coerce") > 2
                    ).sum()
                )
                if not detect_rows.empty
                else 0
            ),
            "dominance_warning": bool(
                not detect_rows.empty
                and float(
                    (
                        pd.to_numeric(detect_rows["touch_count"], errors="coerce") <= 2
                    ).mean()
                )
                >= EQHL_TOUCH_DOMINANCE_WARN_THRESHOLD
            ),
        },
        "touch_bucket_trade_metrics": {
            bucket: _trade_metrics(
                calibration_events.loc[calibration_events["touch_bucket"] == bucket]
            )
            for bucket in ("2-touch", "3-touch+")
        },
        "formation_delay_bucket_trade_metrics": {
            bucket: _trade_metrics(
                calibration_events.loc[
                    calibration_events["formation_delay_bucket"] == bucket
                ]
            )
            for bucket in sorted(
                calibration_events["formation_delay_bucket"].dropna().unique()
            )
        },
        "formation_delay_bucket_by_side_trade_metrics": _cross_trade_metrics(
            calibration_events,
            row_column="formation_delay_bucket",
            col_column="side",
        ),
        "formation_delay_bucket_by_structural_quartile_trade_metrics": _cross_trade_metrics(
            calibration_events,
            row_column="formation_delay_bucket",
            col_column="structural_score_quartile",
        ),
        "active_age_bucket_snapshot_counts": (
            active_snapshots["active_age_bucket"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
            if not active_snapshots.empty
            else {}
        ),
        "freshness_bucket_snapshot_counts": (
            active_snapshots["freshness_bucket"]
            .value_counts(dropna=False)
            .sort_index()
            .astype(int)
            .to_dict()
            if not active_snapshots.empty
            else {}
        ),
        "research": summarize_equal_hl_research(research_table),
    }
    summary["candidate_comparison"] = {
        **candidate_summary,
        "score_repair_accepted": selected_v2_accepted,
    }
    summary["score_calibration"] = {
        "selected_candidate": selected_v2,
        "selected_candidate_accepted": selected_v2_accepted,
        "width_reference": float(score_calibration["width_reference"]),
        "component_direction_audit_rows": int(
            len(score_calibration["component_direction_audit"])
        ),
        "coarse_candidate_count": int(len(score_calibration["coarse_candidates"])),
        "refined_candidate_count": int(len(score_calibration["refined_candidates"])),
        "temporally_stable_coarse_count": (
            int(
                score_calibration["coarse_candidates"]
                .get("temporally_stable", pd.Series(dtype=bool))
                .sum()
            )
            if not score_calibration["coarse_candidates"].empty
            else 0
        ),
        "temporally_stable_refined_count": (
            int(
                score_calibration["refined_candidates"]
                .get("temporally_stable", pd.Series(dtype=bool))
                .sum()
            )
            if not score_calibration["refined_candidates"].empty
            else 0
        ),
        "selected_temporally_stable": bool(
            selected_v2 is not None and selected_v2.get("temporally_stable", False)
        ),
    }
    summary["feature_monotonicity_audit_summary"] = {
        "stable_positive_features": (
            sorted(
                score_calibration["component_direction_audit"]
                .loc[
                    (
                        score_calibration["component_direction_audit"].get(
                            "stability_classification", pd.Series()
                        )
                        == "stable_positive"
                    )
                    & (
                        score_calibration["component_direction_audit"].get(
                            "segment", pd.Series()
                        )
                        == "overall"
                    )
                    & (
                        score_calibration["component_direction_audit"].get(
                            "variant", pd.Series()
                        )
                        == "current"
                    )
                ]["component"]
                .dropna()
                .unique()
                .tolist()
            )
            if not score_calibration["component_direction_audit"].empty
            else []
        ),
        "unstable_features": (
            sorted(
                score_calibration["component_direction_audit"]
                .loc[
                    (
                        score_calibration["component_direction_audit"].get(
                            "stability_classification", pd.Series()
                        )
                        == "unstable"
                    )
                    & (
                        score_calibration["component_direction_audit"].get(
                            "segment", pd.Series()
                        )
                        == "overall"
                    )
                    & (
                        score_calibration["component_direction_audit"].get(
                            "variant", pd.Series()
                        )
                        == "current"
                    )
                ]["component"]
                .dropna()
                .unique()
                .tolist()
            )
            if not score_calibration["component_direction_audit"].empty
            else []
        ),
    }
    return summary


def _add_active_band(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    prefix: str,
    name: str,
    color: str,
) -> None:
    mask = pd.to_numeric(df.get(f"{prefix}_active"), errors="coerce").fillna(0) == 1
    if not mask.any():
        return
    scoped = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "cluster_id": pd.to_numeric(df[f"{prefix}_active_id"], errors="coerce"),
            "low": pd.to_numeric(df[f"{prefix}_active_low"], errors="coerce"),
            "high": pd.to_numeric(df[f"{prefix}_active_high"], errors="coerce"),
            "level": pd.to_numeric(df[f"{prefix}_active_level"], errors="coerce"),
            "touch_count": pd.to_numeric(
                df[f"{prefix}_active_touch_count"], errors="coerce"
            ),
            "width_atr": pd.to_numeric(
                df[f"{prefix}_active_width_atr"], errors="coerce"
            ),
            "span": pd.to_numeric(
                df.get(f"{prefix}_active_span", pd.Series(np.nan, index=df.index)),
                errors="coerce",
            ),
            "age": pd.to_numeric(df[f"{prefix}_active_age"], errors="coerce"),
            "structural_score": pd.to_numeric(
                df.get(f"{prefix}_active_structural_score"), errors="coerce"
            ),
            "tradeable_live_score": pd.to_numeric(
                df.get(
                    f"{prefix}_active_tradeable_live_score",
                    df[f"{prefix}_active_score"],
                ),
                errors="coerce",
            ),
            "score": pd.to_numeric(df[f"{prefix}_active_score"], errors="coerce"),
            "formation_delay": pd.to_numeric(
                df.get(f"{prefix}_active_formation_delay"), errors="coerce"
            ),
            "swept": pd.to_numeric(df[f"{prefix}_active_swept"], errors="coerce"),
        }
    ).loc[mask]
    segment_key = (
        scoped["cluster_id"].ne(scoped["cluster_id"].shift(fill_value=np.nan))
    ).cumsum()
    show_band_legend = True
    show_line_legend = True
    for _, segment in scoped.groupby(segment_key):
        customdata = np.column_stack(
            [
                segment["cluster_id"].to_numpy(dtype=float),
                segment["touch_count"].to_numpy(dtype=float),
                segment["width_atr"].to_numpy(dtype=float),
                segment["span"].to_numpy(dtype=float),
                segment["age"].to_numpy(dtype=float),
                segment["structural_score"].to_numpy(dtype=float),
                segment["tradeable_live_score"].to_numpy(dtype=float),
                segment["formation_delay"].to_numpy(dtype=float),
                segment["score"].to_numpy(dtype=float),
                segment["swept"].to_numpy(dtype=float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=segment["timestamp"],
                y=segment["high"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                connectgaps=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=segment["timestamp"],
                y=segment["low"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=color.replace("1.0)", "0.12)"),
                hoverinfo="skip",
                name=f"{name} band",
                connectgaps=False,
                showlegend=show_band_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=segment["timestamp"],
                y=segment["level"],
                mode="lines",
                line={"color": color, "width": 2},
                name=f"{name} selected",
                customdata=customdata,
                hovertemplate=(
                    "ts=%{x}<br>"
                    "level=%{y:.5f}<br>"
                    "cluster_id=%{customdata[0]:.0f}<br>"
                    "touches=%{customdata[1]:.0f}<br>"
                    "width_atr=%{customdata[2]:.4f}<br>"
                    "span=%{customdata[3]:.0f}<br>"
                    "age=%{customdata[4]:.0f}<br>"
                    "structural=%{customdata[5]:.4f}<br>"
                    "tradeable=%{customdata[6]:.4f}<br>"
                    "formation_delay=%{customdata[7]:.0f}<br>"
                    "score=%{customdata[8]:.4f}<br>"
                    "swept=%{customdata[9]:.0f}<extra></extra>"
                ),
                connectgaps=False,
                showlegend=show_line_legend,
            )
        )
        show_band_legend = False
        show_line_legend = False


def _add_secondary_active_line(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    prefix: str,
    name: str,
    color: str,
) -> None:
    id_col = f"{prefix}_rank2_active_id"
    level_col = f"{prefix}_rank2_active_level"
    if id_col not in df.columns or level_col not in df.columns:
        return
    scoped = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "cluster_id": pd.to_numeric(df[id_col], errors="coerce"),
            "level": pd.to_numeric(df[level_col], errors="coerce"),
        }
    ).dropna(subset=["cluster_id", "level"])
    if scoped.empty:
        return
    segment_key = (
        scoped["cluster_id"].ne(scoped["cluster_id"].shift(fill_value=np.nan))
    ).cumsum()
    showlegend = True
    for _, segment in scoped.groupby(segment_key):
        cluster_id = int(segment["cluster_id"].iloc[0])
        fig.add_trace(
            go.Scatter(
                x=segment["timestamp"],
                y=segment["level"],
                mode="lines",
                line={"color": color, "width": 1.5, "dash": "dot"},
                name=f"{name} secondary",
                hovertemplate=(
                    "ts=%{x}<br>"
                    "level=%{y:.5f}<br>"
                    f"cluster_id={cluster_id}<extra></extra>"
                ),
                connectgaps=False,
                showlegend=showlegend,
            )
        )
        showlegend = False


def validate_equal_hl(
    df: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Equal H/L Validation",
    research_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    analysis_df = full_df if full_df is not None else df
    if research_table is None:
        research_table = build_equal_hl_research_table(analysis_df)

    summary = summarize_equal_hl(df, full_df=analysis_df, research_table=research_table)
    tables = extract_equal_hl_tables(analysis_df, research_table=research_table)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    if "swing_high_confirm_flag" in df.columns:
        mask = (
            pd.to_numeric(df["swing_high_confirm_flag"], errors="coerce").fillna(0) == 1
        )
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, "swing_high_confirm_price"],
                    mode="markers",
                    marker={"symbol": "triangle-down", "size": 9, "color": "#b91c1c"},
                    name="Confirmed swing highs",
                )
            )
    if "swing_low_confirm_flag" in df.columns:
        mask = (
            pd.to_numeric(df["swing_low_confirm_flag"], errors="coerce").fillna(0) == 1
        )
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, "swing_low_confirm_price"],
                    mode="markers",
                    marker={"symbol": "triangle-up", "size": 9, "color": "#047857"},
                    name="Confirmed swing lows",
                )
            )

    _add_active_band(fig, df, prefix="eqh", name="EQH", color="rgba(185, 28, 28, 1.0)")
    _add_active_band(fig, df, prefix="eql", name="EQL", color="rgba(4, 120, 87, 1.0)")
    _add_secondary_active_line(
        fig, df, prefix="eqh", name="EQH", color="rgba(185, 28, 28, 0.70)"
    )
    _add_secondary_active_line(
        fig, df, prefix="eql", name="EQL", color="rgba(4, 120, 87, 0.70)"
    )

    for prefix, label, color, symbol in (
        ("eqh", "EQH detect", "#7f1d1d", "diamond"),
        ("eql", "EQL detect", "#065f46", "diamond"),
    ):
        mask = (
            pd.to_numeric(df.get(f"{prefix}_detect_flag"), errors="coerce").fillna(0)
            == 1
        )
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, "timestamp"],
                    y=df.loc[mask, f"{prefix}_level_on_detect"],
                    mode="markers",
                    marker={"symbol": symbol, "size": 10, "color": color},
                    name=label,
                )
            )
        swept_mask = (
            pd.to_numeric(df.get(f"{prefix}_swept_flag"), errors="coerce").fillna(0)
            == 1
        )
        if swept_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[swept_mask, "timestamp"],
                    y=df.loc[swept_mask, f"{prefix}_swept_level"],
                    mode="markers",
                    marker={"symbol": "x", "size": 10, "color": color},
                    name=f"{label} swept",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend={"orientation": "h"},
        template="plotly_white",
    )

    html_path = None
    if outpath is not None:
        html_path = Path(outpath)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))

    return {
        "summary": summary,
        "tables": tables,
        "figure": fig,
        "html_path": str(html_path) if html_path is not None else None,
    }


def equal_hl_candidate_passes_acceptance(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> bool:
    if not bool(candidate.get("score_repair_accepted", False)):
        return False
    baseline_events = float(baseline.get("event_count", 0) or 0)
    candidate_events = float(candidate.get("event_count", 0) or 0)
    if baseline_events <= 0 or candidate_events <= baseline_events:
        return False

    candidate_top_q = float(candidate.get("top_quartile_avg_r", np.nan))
    baseline_top_q = float(baseline.get("top_quartile_avg_r", np.nan))
    if np.isfinite(candidate_top_q) and np.isfinite(baseline_top_q):
        if candidate_top_q < (baseline_top_q - EQHL_THRESHOLD_TOP_BUCKET_TOLERANCE):
            return False

    candidate_top_d = float(candidate.get("top_decile_avg_r", np.nan))
    baseline_top_d = float(baseline.get("top_decile_avg_r", np.nan))
    if np.isfinite(candidate_top_d) and np.isfinite(baseline_top_d):
        if candidate_top_d < (baseline_top_d - EQHL_THRESHOLD_TOP_BUCKET_TOLERANCE):
            return False

    candidate_top_q_wr = float(candidate.get("top_quartile_win_rate", np.nan))
    baseline_top_q_wr = float(baseline.get("top_quartile_win_rate", np.nan))
    if np.isfinite(candidate_top_q_wr) and np.isfinite(baseline_top_q_wr):
        if candidate_top_q_wr < (
            baseline_top_q_wr - EQHL_THRESHOLD_TOP_BUCKET_TOLERANCE
        ):
            return False

    candidate_form = float(candidate.get("formation_delay_mean", np.nan))
    baseline_form = float(baseline.get("formation_delay_mean", np.nan))
    if np.isfinite(candidate_form) and np.isfinite(baseline_form):
        if candidate_form > (baseline_form + EQHL_THRESHOLD_FORMATION_DELAY_TOLERANCE):
            return False

    candidate_active_age = float(candidate.get("active_age_mean", np.nan))
    baseline_active_age = float(baseline.get("active_age_mean", np.nan))
    if np.isfinite(candidate_active_age) and np.isfinite(baseline_active_age):
        if candidate_active_age > (
            baseline_active_age + EQHL_THRESHOLD_ACTIVE_AGE_TOLERANCE
        ):
            return False

    bottom_decile = float(candidate.get("bottom_decile_avg_r", np.nan))
    if np.isfinite(candidate_top_d) and np.isfinite(bottom_decile):
        if candidate_top_d <= bottom_decile:
            return False

    for side_key in ("eqh_top_quartile_avg_r", "eql_top_quartile_avg_r"):
        side_value = float(candidate.get(side_key, np.nan))
        baseline_side = float(baseline.get(side_key, np.nan))
        if np.isfinite(side_value) and side_value < EQHL_THRESHOLD_SIDE_COLLAPSE_FLOOR:
            return False
        if np.isfinite(side_value) and np.isfinite(baseline_side):
            if side_value < (baseline_side - 0.05):
                return False

    robustness = candidate.get("robustness", {})
    subperiods = robustness.get("subperiod", {}) if isinstance(robustness, dict) else {}
    for period_key in ("first_half", "second_half"):
        period = subperiods.get(period_key, {})
        if not isinstance(period, dict):
            continue
        rank_corr = float(period.get("rank_corr", np.nan))
        if np.isfinite(rank_corr) and rank_corr < EQHL_ACCEPT_SIDE_RANK_CORR_FLOOR:
            return False
        period_top = float(period.get("top_quartile_avg_r", np.nan))
        period_bottom = float(period.get("bottom_decile_avg_r", np.nan))
        if (
            np.isfinite(period_top)
            and np.isfinite(period_bottom)
            and period_top < period_bottom
        ):
            return False

    return True
