# src/validation/indicators/trend_state.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STATE_NAMES = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}
REGIME_NAMES = {0: "RANGING", 1: "TRANSITIONAL", 2: "TRENDING"}
COMMIT_ENTRY_MIN = 0.62
DIRECTIONAL_EVIDENCE_HIGH = 0.65
STRONG_ENV_ADX_MIN = 0.70
STRONG_ENV_SLOPE_MIN = 0.70
STRONG_ENV_CONTINUITY_MIN = 0.70
STRONG_ENV_COMPRESSION_MAX = 0.30
LOW_COMMIT_GAP_MAX = 0.10
MEDIUM_COMMIT_GAP_MAX = 0.18

TREND_STATE_SOURCE = "src/indicators/structure/trend_state.add_trend_state"
REGIME_SOURCE = "src/indicators/foundation/regime.add_regime"
VALIDATOR_DERIVED_SOURCE = "src/validation/indicators/trend_state.validate_trend_state"


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def _state_runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if len(values) == 0:
        return runs

    start = 0
    cur = int(values[0])

    for i in range(1, len(values)):
        val = int(values[i])
        if val != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = val

    runs.append((start, len(values) - 1, cur))
    return runs


def _add_state_background(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    row: int,
    col: int,
    state_col: str,
    opacity: float,
    max_runs: int = 250,
) -> None:
    if state_col != "trend_state":
        return
    if state_col not in df.columns:
        return

    values = df[state_col].fillna(0).astype(int).to_numpy()
    x = df["timestamp"].reset_index(drop=True)

    runs = _state_runs(values)
    non_zero_runs = [(s, e, v) for s, e, v in runs if v != 0]

    if len(non_zero_runs) > max_runs:
        step = int(np.ceil(len(non_zero_runs) / max_runs))
        non_zero_runs = non_zero_runs[::step]

    for start, end, state in non_zero_runs:
        color = (
            f"rgba(0, 180, 0, {opacity})"
            if state == 1
            else f"rgba(200, 0, 0, {opacity})"
        )
        fig.add_vrect(
            x0=x.iloc[start],
            x1=x.iloc[end],
            fillcolor=color,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )


def _transition_table(series: pd.Series, from_name: str, to_name: str) -> pd.DataFrame:
    prev = pd.to_numeric(series.shift(1), errors="coerce")
    cur = pd.to_numeric(series, errors="coerce")
    changed = prev.notna() & cur.notna() & prev.ne(cur)

    return (
        pd.DataFrame(
            {from_name: prev[changed].astype(int), to_name: cur[changed].astype(int)}
        )
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def _duration_stats(series: pd.Series) -> pd.DataFrame:
    vals = series.fillna(0).astype(int).to_numpy()
    runs = _state_runs(vals)

    rows: list[dict[str, float]] = []
    for start, end, state in runs:
        rows.append({"state": state, "duration": end - start + 1})

    if not rows:
        return pd.DataFrame(columns=["count", "mean", "median", "max"])

    runs_df = pd.DataFrame(rows)
    return (
        runs_df.groupby("state")["duration"]
        .agg(["count", "mean", "median", "max"])
        .sort_index()
    )


def _sample_transition_windows(
    df: pd.DataFrame,
    *,
    transition_col: str,
    n_windows: int,
    pad: int = 6,
) -> list[pd.DataFrame]:
    if transition_col not in df.columns:
        return []

    positions = np.flatnonzero(
        df[transition_col].fillna(0).astype(int).to_numpy() == 1
    ).tolist()
    windows: list[pd.DataFrame] = []

    for pos in positions[:n_windows]:
        start = max(0, pos - pad)
        end = min(len(df), pos + pad + 1)
        win = df.iloc[start:end].copy()
        win["event_row"] = 0
        win.loc[win.index == win.index[min(pos - start, len(win) - 1)], "event_row"] = 1
        windows.append(win)

    return windows


def _segment_stats(series: pd.Series) -> dict[str, object]:
    durations = _duration_stats(series)
    vals = series.fillna(0).astype(int).to_numpy()
    runs = _state_runs(vals)
    run_count = len(runs)
    single_bar_count = sum(1 for start, end, _ in runs if end - start + 1 == 1)
    two_bar_count = sum(1 for start, end, _ in runs if end - start + 1 == 2)
    return {
        "duration_stats": durations,
        "single_bar_segment_count": single_bar_count,
        "single_bar_segment_rate_pct": (
            float(single_bar_count / run_count * 100.0) if run_count else 0.0
        ),
        "two_bar_segment_count": two_bar_count,
        "two_bar_segment_rate_pct": (
            float(two_bar_count / run_count * 100.0) if run_count else 0.0
        ),
    }


def _named_counts(series: pd.Series, labels: dict[int, str]) -> dict[str, int]:
    counts = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {
        labels.get(int(key), str(int(key))): int(value) for key, value in counts.items()
    }


def _named_pct(series: pd.Series, labels: dict[int, str]) -> dict[str, float]:
    counts = (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
    )
    return {
        labels.get(int(key), str(int(key))): float(value * 100.0)
        for key, value in counts.items()
    }


def _confusion_matrix(
    row_series: pd.Series,
    col_series: pd.Series,
    *,
    row_labels: dict[int, str],
    col_labels: dict[int, str],
) -> dict[str, dict[str, int]]:
    rows = pd.to_numeric(row_series, errors="coerce")
    cols = pd.to_numeric(col_series, errors="coerce")
    valid = rows.notna() & cols.notna()
    scoped = pd.DataFrame(
        {"row": rows[valid].astype(int), "col": cols[valid].astype(int)}
    )
    if scoped.empty:
        return {}
    out: dict[str, dict[str, int]] = {}
    for row_key, group in scoped.groupby("row"):
        out[row_labels.get(int(row_key), str(int(row_key)))] = {
            col_labels.get(int(col_key), str(int(col_key))): int(value)
            for col_key, value in group["col"].value_counts().sort_index().items()
        }
    return out


def _state_metric_profile(
    df: pd.DataFrame, state: pd.Series, metric_cols: list[str]
) -> dict[str, dict[str, float | None]]:
    profile: dict[str, dict[str, float | None]] = {}
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        grouped = pd.to_numeric(df[metric], errors="coerce").groupby(state)
        metric_profile: dict[str, float | None] = {}
        for state_key, series in grouped:
            valid = series.dropna()
            metric_profile[STATE_NAMES.get(int(state_key), str(int(state_key)))] = (
                float(valid.mean()) if not valid.empty else None
            )
        profile[metric] = metric_profile
    return profile


def _subset_profile(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    scoped = df.loc[mask].copy()
    if scoped.empty:
        return {"rows": 0}
    out: dict[str, object] = {"rows": int(len(scoped))}
    for col in [
        "trend_confidence",
        "trend_strength_raw",
        "trend_strength_ema",
        "trend_bull_commit_score",
        "trend_bear_commit_score",
        "trend_directional_evidence_score",
        "trend_conf_structure_continuity",
        "trend_conf_neutral_coherence",
        "bars_in_trend_state",
        "trend_persistence_5",
        "trend_persistence_20",
        "adx_strength",
        "ema_slope_strength",
        "structure_continuity",
        "compression_score",
        "regime_confidence",
    ]:
        if col not in scoped.columns:
            continue
        values = pd.to_numeric(scoped[col], errors="coerce").dropna()
        if values.empty:
            continue
        out[col] = {"mean": float(values.mean()), "median": float(values.median())}
    return out


def _numeric_summary(
    df: pd.DataFrame, cols: list[str]
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        out[col] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
        }
    return out


def _state_distribution(series: pd.Series, labels: dict[int, str]) -> dict[str, int]:
    return _named_counts(series, labels)


def _future_window(series: pd.Series, horizon: int) -> pd.DataFrame:
    return pd.concat(
        [
            pd.to_numeric(series.shift(-step), errors="coerce")
            for step in range(1, horizon + 1)
        ],
        axis=1,
    )


def _pct(mask: pd.Series) -> float:
    if int(mask.shape[0]) == 0:
        return 0.0
    return float(mask.mean() * 100.0)


def _bucket_counts(series: pd.Series, labels: list[str]) -> dict[str, int]:
    series_str = series.astype(str)
    return {label: int(series_str.eq(label).sum()) for label in labels}


def validate_trend_state(
    df: pd.DataFrame,
    *,
    outpath: str | Path | None,
    title: str = "Trend State Validation",
    n_windows: int = 5,
    summary_df: pd.DataFrame | None = None,
) -> dict:
    plot_df = _ensure_datetime(df)
    audit_df = _ensure_datetime(summary_df if summary_df is not None else df)

    strict = (
        pd.to_numeric(audit_df["trend_state"], errors="coerce").fillna(0).astype(int)
    )
    bias = (
        pd.to_numeric(audit_df["trend_bias_state"], errors="coerce")
        .fillna(0)
        .astype(int)
        if "trend_bias_state" in audit_df.columns
        else pd.Series(0, index=audit_df.index)
    )
    regime = (
        pd.to_numeric(audit_df["regime"], errors="coerce")
        if "regime" in audit_df.columns
        else pd.Series(np.nan, index=audit_df.index)
    )
    conf = (
        pd.to_numeric(audit_df["trend_confidence"], errors="coerce")
        if "trend_confidence" in audit_df.columns
        else pd.Series(np.nan, index=audit_df.index)
    )

    strict_state_counts = _named_counts(strict, STATE_NAMES)
    strict_state_pct = _named_pct(strict, STATE_NAMES)
    bias_state_counts = _named_counts(bias, STATE_NAMES)
    bias_state_pct = _named_pct(bias, STATE_NAMES)

    transition_count = int(
        audit_df.get("trend_state_changed", pd.Series(0, index=audit_df.index))
        .fillna(0)
        .astype(int)
        .sum()
    )

    avg_bias_score_by_strict_state = (
        {
            STATE_NAMES.get(int(key), str(int(key))): float(value)
            for key, value in audit_df.groupby(strict)["trend_bias_score"]
            .mean()
            .to_dict()
            .items()
        }
        if "trend_bias_score" in audit_df.columns
        else {}
    )

    avg_age_by_strict_state = (
        {
            STATE_NAMES.get(int(key), str(int(key))): float(value)
            for key, value in audit_df.groupby(strict)["trend_state_age"]
            .mean()
            .to_dict()
            .items()
        }
        if "trend_state_age" in audit_df.columns
        else {}
    )

    avg_strength_raw_by_strict_state = (
        {
            STATE_NAMES.get(int(key), str(int(key))): float(value)
            for key, value in audit_df.groupby(strict)["trend_strength_raw"]
            .mean()
            .to_dict()
            .items()
        }
        if "trend_strength_raw" in audit_df.columns
        else {}
    )

    avg_strength_ema_by_strict_state = (
        {
            STATE_NAMES.get(int(key), str(int(key))): float(value)
            for key, value in audit_df.groupby(strict)["trend_strength_ema"]
            .mean()
            .to_dict()
            .items()
        }
        if "trend_strength_ema" in audit_df.columns
        else {}
    )

    strict_neutral_rows = int((strict == 0).sum())
    bias_carry_rows = int(((strict == 0) & (bias != 0)).sum())

    strict_bull_not_ready_rows = (
        int(((strict == 1) & (audit_df["trend_bull_ready"] != 1)).sum())
        if "trend_bull_ready" in audit_df.columns
        else 0
    )
    strict_bear_not_ready_rows = (
        int(((strict == -1) & (audit_df["trend_bear_ready"] != 1)).sum())
        if "trend_bear_ready" in audit_df.columns
        else 0
    )

    confidence_distribution = pd.DataFrame()
    if not conf.dropna().empty:
        confidence_distribution = (
            pd.DataFrame(
                {
                    "trend_state": strict.map(STATE_NAMES),
                    "trend_conf_bucket": pd.cut(
                        conf,
                        bins=[-np.inf, 0.25, 0.45, 0.60, 0.75, np.inf],
                        labels=[
                            "<=0.25",
                            "0.25-0.45",
                            "0.45-0.60",
                            "0.60-0.75",
                            ">=0.75",
                        ],
                    ).astype(str),
                }
            )
            .value_counts()
            .reset_index(name="count")
            .sort_values(["trend_state", "trend_conf_bucket"])
            .reset_index(drop=True)
        )

    transitions_table = _transition_table(strict, "trend_state_from", "trend_state_to")
    bias_transitions_table = _transition_table(bias, "trend_bias_from", "trend_bias_to")
    segment_stats = _segment_stats(strict)

    structure_loss_bull_rows = int(
        audit_df.get("trend_structure_loss_bull", pd.Series(0, index=audit_df.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    structure_loss_bear_rows = int(
        audit_df.get("trend_structure_loss_bear", pd.Series(0, index=audit_df.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    emerging_bull_rows = int(
        audit_df.get("trend_emerging_bull", pd.Series(0, index=audit_df.index))
        .fillna(0)
        .astype(int)
        .sum()
    )
    emerging_bear_rows = int(
        audit_df.get("trend_emerging_bear", pd.Series(0, index=audit_df.index))
        .fillna(0)
        .astype(int)
        .sum()
    )

    regime_phase_counts = (
        audit_df["trend_regime_phase"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        if "trend_regime_phase" in audit_df.columns
        else {}
    )

    confidence_by_state = _state_metric_profile(
        audit_df,
        strict,
        [
            "trend_confidence",
            "trend_conf_structure_continuity",
            "trend_conf_freshness",
            "trend_conf_event_quality",
            "trend_conf_persistence",
            "trend_conf_contradiction_penalty",
            "trend_conf_neutral_coherence",
        ],
    )
    strength_by_state = _state_metric_profile(
        audit_df,
        strict,
        ["trend_strength_raw", "trend_strength_ema"],
    )
    commitment_by_state = _state_metric_profile(
        audit_df,
        strict,
        [
            "trend_bull_commit_score",
            "trend_bear_commit_score",
            "trend_directional_evidence_score",
        ],
    )

    bias_interaction = {
        "trend_x_bias_confusion_matrix": _confusion_matrix(
            strict,
            bias,
            row_labels=STATE_NAMES,
            col_labels=STATE_NAMES,
        ),
        "bias_carry_rows": bias_carry_rows,
        "bias_inherited_rows": int(
            pd.to_numeric(
                audit_df.get(
                    "trend_bias_inherited_flag", pd.Series(0, index=audit_df.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "bias_expired_rows": int(
            pd.to_numeric(
                audit_df.get(
                    "trend_bias_expired_flag", pd.Series(0, index=audit_df.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
        "bias_contradicted_rows": int(
            pd.to_numeric(
                audit_df.get(
                    "trend_bias_contradicted_flag", pd.Series(0, index=audit_df.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .sum()
        ),
    }

    regime_interaction: dict[str, object] = {}
    if "regime" in audit_df.columns:
        regime_interaction = {
            "trend_x_regime_confusion_matrix": _confusion_matrix(
                strict,
                regime,
                row_labels=STATE_NAMES,
                col_labels=REGIME_NAMES,
            )
        }
        if "trend_bias_state" in audit_df.columns:
            regime_interaction["bias_x_regime_confusion_matrix"] = _confusion_matrix(
                bias,
                regime,
                row_labels=STATE_NAMES,
                col_labels=REGIME_NAMES,
            )
        mismatch_mask = strict.eq(0) & regime.eq(2)
        range_mismatch_mask = strict.ne(0) & regime.eq(0)
        regime_interaction["mismatch_profiles"] = {
            "neutral_trend_with_trending_regime": _subset_profile(
                audit_df, mismatch_mask
            ),
            "directional_trend_with_ranging_regime": _subset_profile(
                audit_df, range_mismatch_mask
            ),
        }

    bull_commit = pd.to_numeric(
        audit_df.get(
            "trend_bull_commit_score", pd.Series(np.nan, index=audit_df.index)
        ),
        errors="coerce",
    )
    bear_commit = pd.to_numeric(
        audit_df.get(
            "trend_bear_commit_score", pd.Series(np.nan, index=audit_df.index)
        ),
        errors="coerce",
    )
    directional_evidence = pd.to_numeric(
        audit_df.get(
            "trend_directional_evidence_score",
            pd.Series(np.nan, index=audit_df.index),
        ),
        errors="coerce",
    )
    commit_gap = pd.to_numeric(
        audit_df.get("trend_commit_gap", (bull_commit - bear_commit).abs()),
        errors="coerce",
    )
    adx_strength = pd.to_numeric(
        audit_df.get("adx_strength", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    )
    ema_slope_strength = pd.to_numeric(
        audit_df.get("ema_slope_strength", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    )
    structure_continuity = pd.to_numeric(
        audit_df.get("structure_continuity", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    )
    compression_score = pd.to_numeric(
        audit_df.get("compression_score", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    )
    bars_in_state = pd.to_numeric(
        audit_df.get("bars_in_trend_state", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    )
    trend_event = (
        pd.to_numeric(
            audit_df.get("trend_event", pd.Series(0, index=audit_df.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    contradiction_penalty = pd.to_numeric(
        audit_df.get(
            "trend_conf_contradiction_penalty", pd.Series(np.nan, index=audit_df.index)
        ),
        errors="coerce",
    )
    strong_environment_mask = (
        adx_strength.ge(STRONG_ENV_ADX_MIN)
        & ema_slope_strength.ge(STRONG_ENV_SLOPE_MIN)
        & structure_continuity.ge(STRONG_ENV_CONTINUITY_MIN)
        & compression_score.le(STRONG_ENV_COMPRESSION_MAX)
    )
    medium_environment_mask = (
        adx_strength.ge(0.50)
        & ema_slope_strength.ge(0.50)
        & structure_continuity.ge(0.50)
        & compression_score.le(0.45)
        & ~strong_environment_mask
    )
    neutral_directional_evidence_broad_mask = strict.eq(0) & (
        directional_evidence.ge(DIRECTIONAL_EVIDENCE_HIGH)
        | adx_strength.gt(STRONG_ENV_ADX_MIN)
        | ema_slope_strength.gt(STRONG_ENV_SLOPE_MIN)
        | structure_continuity.gt(STRONG_ENV_CONTINUITY_MIN)
        | pd.to_numeric(
            audit_df.get("trend_strength_ema", pd.Series(0.0, index=audit_df.index)),
            errors="coerce",
        )
        .abs()
        .gt(0.2)
    )
    directional_low_continuity_mask = strict.ne(0) & pd.to_numeric(
        audit_df.get(
            "trend_conf_structure_continuity", pd.Series(np.nan, index=audit_df.index)
        ),
        errors="coerce",
    ).lt(0.35)
    directional_low_conf_mask = strict.ne(0) & pd.to_numeric(
        audit_df.get("trend_confidence", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    ).lt(0.45)
    neutral_high_bull_commit_mask = strict.eq(0) & bull_commit.ge(COMMIT_ENTRY_MIN)
    neutral_high_bear_commit_mask = strict.eq(0) & bear_commit.ge(COMMIT_ENTRY_MIN)
    directional_low_commit_gap_mask = strict.ne(0) & (
        bull_commit - bear_commit
    ).abs().lt(0.18)
    neutral_bias_strength_mask = (
        strict.eq(0)
        & bias.ne(0)
        & pd.to_numeric(
            audit_df.get("trend_strength_ema", pd.Series(0.0, index=audit_df.index)),
            errors="coerce",
        )
        .abs()
        .gt(0.2)
    )
    neutral_mask = strict.eq(0)
    neutral_row_count = int(neutral_mask.sum())
    neutral_directional_evidence_strict_mask = neutral_mask & directional_evidence.ge(
        DIRECTIONAL_EVIDENCE_HIGH
    )
    neutral_overuse_audit = {
        "neutral_row_count": neutral_row_count,
        "neutral_with_high_bull_commit_count": int(neutral_high_bull_commit_mask.sum()),
        "neutral_with_high_bear_commit_count": int(neutral_high_bear_commit_mask.sum()),
        "neutral_with_high_directional_evidence_strict_count": int(
            neutral_directional_evidence_strict_mask.sum()
        ),
        "neutral_with_high_bull_commit_rate_pct": (
            float(neutral_high_bull_commit_mask.sum() / neutral_row_count * 100.0)
            if neutral_row_count
            else 0.0
        ),
        "neutral_with_high_bear_commit_rate_pct": (
            float(neutral_high_bear_commit_mask.sum() / neutral_row_count * 100.0)
            if neutral_row_count
            else 0.0
        ),
        "neutral_with_high_directional_evidence_strict_rate_pct": (
            float(
                neutral_directional_evidence_strict_mask.sum()
                / neutral_row_count
                * 100.0
            )
            if neutral_row_count
            else 0.0
        ),
    }
    neutral_confidence_audit: dict[str, dict[str, float | int | None]] = {}
    for bucket_name, bucket_mask in {
        "low_directional_evidence": neutral_mask & directional_evidence.lt(0.35),
        "medium_directional_evidence": neutral_mask
        & directional_evidence.ge(0.35)
        & directional_evidence.lt(DIRECTIONAL_EVIDENCE_HIGH),
        "high_directional_evidence": neutral_mask
        & directional_evidence.ge(DIRECTIONAL_EVIDENCE_HIGH),
    }.items():
        vals = conf.loc[bucket_mask].dropna()
        neutral_confidence_audit[bucket_name] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()) if not vals.empty else None,
            "median": float(vals.median()) if not vals.empty else None,
        }

    confidence_ordering_check = {
        "neutral_lt_bearish_mean": bool(
            confidence_by_state.get("trend_confidence", {}).get("NEUTRAL", np.inf)
            < confidence_by_state.get("trend_confidence", {}).get("BEARISH", -np.inf)
        ),
        "neutral_lt_bullish_mean": bool(
            confidence_by_state.get("trend_confidence", {}).get("NEUTRAL", np.inf)
            < confidence_by_state.get("trend_confidence", {}).get("BULLISH", -np.inf)
        ),
    }
    confidence_separation_check = {
        "bearish_minus_neutral_mean": (
            float(
                confidence_by_state["trend_confidence"]["BEARISH"]
                - confidence_by_state["trend_confidence"]["NEUTRAL"]
            )
            if "trend_confidence" in confidence_by_state
            and {"BEARISH", "NEUTRAL"}.issubset(confidence_by_state["trend_confidence"])
            else None
        ),
        "bullish_minus_neutral_mean": (
            float(
                confidence_by_state["trend_confidence"]["BULLISH"]
                - confidence_by_state["trend_confidence"]["NEUTRAL"]
            )
            if "trend_confidence" in confidence_by_state
            and {"BULLISH", "NEUTRAL"}.issubset(confidence_by_state["trend_confidence"])
            else None
        ),
    }
    neutral_conf_series = conf.loc[neutral_mask].dropna()
    neutral_confidence_cap_check = {
        "neutral_mean_confidence": (
            float(neutral_conf_series.mean()) if not neutral_conf_series.empty else None
        ),
        "neutral_max_confidence": (
            float(neutral_conf_series.max()) if not neutral_conf_series.empty else None
        ),
        "neutral_max_confidence_le_0_65": (
            bool(neutral_conf_series.max() <= 0.65)
            if not neutral_conf_series.empty
            else True
        ),
    }

    neutral_in_trend_mask = (
        strict.eq(0) & regime.eq(2)
        if "regime" in audit_df.columns
        else pd.Series(False, index=audit_df.index)
    )
    directional_in_range_mask = (
        strict.ne(0) & regime.eq(0)
        if "regime" in audit_df.columns
        else pd.Series(False, index=audit_df.index)
    )
    neutral_with_directional_bias_mask = strict.eq(0) & bias.ne(0)
    neutral_in_strong_env_mask = neutral_in_trend_mask & strong_environment_mask
    stale_neutral_neutral_mask = neutral_mask & bias.eq(0) & bars_in_state.ge(6)
    old_neutral_strong_env_mask = stale_neutral_neutral_mask & strong_environment_mask

    neutral_in_trend_df = audit_df.loc[neutral_in_trend_mask].copy()
    directional_in_range_df = audit_df.loc[directional_in_range_mask].copy()
    neutral_with_directional_bias_df = audit_df.loc[
        neutral_with_directional_bias_mask
    ].copy()
    old_neutral_strong_env_df = audit_df.loc[old_neutral_strong_env_mask].copy()
    audit_df.loc[stale_neutral_neutral_mask].copy()

    neutral_in_trend_audit: dict[str, object] = {
        "row_count": int(len(neutral_in_trend_df)),
        "strong_environment_row_count": int(neutral_in_strong_env_mask.sum()),
        "metrics": _numeric_summary(
            neutral_in_trend_df.assign(
                trend_commit_gap=commit_gap.loc[neutral_in_trend_mask]
            ),
            [
                "bars_in_trend_state",
                "trend_directional_evidence_score",
                "trend_bull_commit_score",
                "trend_bear_commit_score",
                "trend_commit_gap",
                "trend_confidence",
                "adx_strength",
                "ema_slope_strength",
                "structure_continuity",
                "compression_score",
            ],
        ),
        "trend_bias_state_distribution": _state_distribution(
            bias.loc[neutral_in_trend_mask], STATE_NAMES
        ),
        "trend_event_distribution": _state_distribution(
            trend_event.loc[neutral_in_trend_mask],
            {-2: "-2", -1: "-1", 0: "0", 1: "1", 2: "2"},
        ),
    }

    directional_in_range_audit: dict[str, object] = {
        "row_count": int(len(directional_in_range_df)),
        "metrics": _numeric_summary(
            directional_in_range_df.assign(
                trend_commit_gap=commit_gap.loc[directional_in_range_mask]
            ),
            [
                "bars_in_trend_state",
                "trend_persistence_5",
                "trend_persistence_20",
                "trend_directional_evidence_score",
                "trend_bull_commit_score",
                "trend_bear_commit_score",
                "trend_commit_gap",
                "trend_confidence",
                "adx_strength",
                "ema_slope_strength",
                "structure_continuity",
                "compression_score",
            ],
        ),
        "early_state": _subset_profile(
            directional_in_range_df.assign(
                trend_commit_gap=commit_gap.loc[directional_in_range_mask]
            ),
            directional_in_range_df["bars_in_trend_state"].le(3),
        ),
        "mature_state": _subset_profile(
            directional_in_range_df.assign(
                trend_commit_gap=commit_gap.loc[directional_in_range_mask]
            ),
            directional_in_range_df["bars_in_trend_state"].gt(3),
        ),
    }

    dominant_side_series = (
        pd.to_numeric(
            audit_df.get(
                "trend_commit_dominant_side", pd.Series(0, index=audit_df.index)
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    bull_dom_2of3 = (
        pd.to_numeric(
            audit_df.get(
                "trend_bull_dominant_2_of_3", pd.Series(0, index=audit_df.index)
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    bear_dom_2of3 = (
        pd.to_numeric(
            audit_df.get(
                "trend_bear_dominant_2_of_3", pd.Series(0, index=audit_df.index)
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    bull_score_gt_bear = bull_commit.gt(bear_commit)
    bear_score_gt_bull = bear_commit.gt(bull_commit)
    bull_dominance_agrees = (
        dominant_side_series.eq(1) & bull_score_gt_bear & bull_dom_2of3.eq(1)
    )
    bear_dominance_agrees = (
        dominant_side_series.eq(-1) & bear_score_gt_bull & bear_dom_2of3.eq(1)
    )
    dominance_persistence_present = bull_dom_2of3.eq(1) | bear_dom_2of3.eq(1)
    dominance_agrees = bull_dominance_agrees | bear_dominance_agrees
    medium_or_strong_environment_mask = (
        strong_environment_mask | medium_environment_mask
    )
    dominant_commit = pd.Series(
        np.maximum(
            bull_commit.to_numpy(dtype=float), bear_commit.to_numpy(dtype=float)
        ),
        index=audit_df.index,
        dtype=float,
    )
    weaker_commit = pd.Series(
        np.minimum(
            bull_commit.to_numpy(dtype=float), bear_commit.to_numpy(dtype=float)
        ),
        index=audit_df.index,
        dtype=float,
    )
    total_commit_mass = bull_commit.add(bear_commit, fill_value=np.nan)
    hh_count_series = pd.to_numeric(
        audit_df.get("hh_count", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    hl_count_series = pd.to_numeric(
        audit_df.get("hl_count", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    lh_count_series = pd.to_numeric(
        audit_df.get("lh_count", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    ll_count_series = pd.to_numeric(
        audit_df.get("ll_count", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    bull_structure_pairs = pd.Series(
        np.minimum(
            hh_count_series.to_numpy(dtype=float), hl_count_series.to_numpy(dtype=float)
        ),
        index=audit_df.index,
        dtype=float,
    )
    bear_structure_pairs = pd.Series(
        np.minimum(
            lh_count_series.to_numpy(dtype=float), ll_count_series.to_numpy(dtype=float)
        ),
        index=audit_df.index,
        dtype=float,
    )
    last_hh_idx_series = pd.to_numeric(
        audit_df.get("last_hh_idx", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    )
    last_hl_idx_series = pd.to_numeric(
        audit_df.get("last_hl_idx", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    )
    last_lh_idx_series = pd.to_numeric(
        audit_df.get("last_lh_idx", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    )
    last_ll_idx_series = pd.to_numeric(
        audit_df.get("last_ll_idx", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    )
    row_pos = pd.Series(np.arange(len(audit_df), dtype=float), index=audit_df.index)
    bull_pair_idx_stack = np.column_stack(
        [
            last_hh_idx_series.to_numpy(dtype=float),
            last_hl_idx_series.to_numpy(dtype=float),
        ]
    )
    bull_pair_idx_filled = np.where(
        np.isnan(bull_pair_idx_stack), -np.inf, bull_pair_idx_stack
    )
    bull_last_pair_idx = pd.Series(
        np.max(bull_pair_idx_filled, axis=1),
        index=audit_df.index,
        dtype=float,
    ).replace(-np.inf, np.nan)
    bear_pair_idx_stack = np.column_stack(
        [
            last_lh_idx_series.to_numpy(dtype=float),
            last_ll_idx_series.to_numpy(dtype=float),
        ]
    )
    bear_pair_idx_filled = np.where(
        np.isnan(bear_pair_idx_stack), -np.inf, bear_pair_idx_stack
    )
    bear_last_pair_idx = pd.Series(
        np.max(bear_pair_idx_filled, axis=1),
        index=audit_df.index,
        dtype=float,
    ).replace(-np.inf, np.nan)
    bull_pair_age = (row_pos - bull_last_pair_idx).clip(lower=0.0)
    bear_pair_age = (row_pos - bear_last_pair_idx).clip(lower=0.0)
    bull_pressure_raw = pd.to_numeric(
        audit_df.get("trend_pressure_bull_raw", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    bear_pressure_raw = pd.to_numeric(
        audit_df.get("trend_pressure_bear_raw", pd.Series(0.0, index=audit_df.index)),
        errors="coerce",
    ).fillna(0.0)
    commit_gap_persist_3 = pd.to_numeric(
        audit_df.get(
            "trend_commit_gap_persist_3", pd.Series(np.nan, index=audit_df.index)
        ),
        errors="coerce",
    )
    dominant_structure_pairs = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) >= bear_commit.to_numpy(dtype=float),
            bull_structure_pairs.to_numpy(dtype=float),
            bear_structure_pairs.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    weaker_structure_pairs = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) < bear_commit.to_numpy(dtype=float),
            bull_structure_pairs.to_numpy(dtype=float),
            bear_structure_pairs.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    dominant_pair_age = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) >= bear_commit.to_numpy(dtype=float),
            bull_pair_age.to_numpy(dtype=float),
            bear_pair_age.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    weaker_pair_age = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) < bear_commit.to_numpy(dtype=float),
            bull_pair_age.to_numpy(dtype=float),
            bear_pair_age.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    dominant_pressure_raw = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) >= bear_commit.to_numpy(dtype=float),
            bull_pressure_raw.to_numpy(dtype=float),
            bear_pressure_raw.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    weaker_pressure_raw = pd.Series(
        np.where(
            bull_commit.to_numpy(dtype=float) < bear_commit.to_numpy(dtype=float),
            bull_pressure_raw.to_numpy(dtype=float),
            bear_pressure_raw.to_numpy(dtype=float),
        ),
        index=audit_df.index,
        dtype=float,
    )
    bull_recent_event = (
        trend_event.gt(0)
        .rolling(window=3, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
        .eq(1)
    )
    bear_recent_event = (
        trend_event.lt(0)
        .rolling(window=3, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    stale_neutral_strong_env_mask = stale_neutral_neutral_mask & strong_environment_mask
    stale_neutral_medium_env_mask = stale_neutral_neutral_mask & medium_environment_mask
    stale_neutral_weak_or_mixed_env_mask = (
        stale_neutral_neutral_mask & ~strong_environment_mask & ~medium_environment_mask
    )

    candidate_tight_mask = (
        stale_neutral_neutral_mask
        & strong_environment_mask
        & commit_gap.ge(MEDIUM_COMMIT_GAP_MAX)
        & (bull_dominance_agrees | bear_dominance_agrees)
    )
    candidate_medium_mask = (
        stale_neutral_neutral_mask
        & strong_environment_mask
        & commit_gap.ge(LOW_COMMIT_GAP_MAX)
        & commit_gap.lt(MEDIUM_COMMIT_GAP_MAX)
        & (bull_dominance_agrees | bear_dominance_agrees)
    )
    candidate_loose_mask = (
        stale_neutral_neutral_mask
        & medium_or_strong_environment_mask
        & commit_gap.ge(LOW_COMMIT_GAP_MAX)
        & dominance_persistence_present
        & (bull_score_gt_bear | bear_score_gt_bull)
    )
    stale_neutral_promotion_candidate_mask = (
        candidate_tight_mask | candidate_medium_mask
    )

    old_neutral_strong_env_audit: dict[str, object] = {
        "row_count": int(len(old_neutral_strong_env_df)),
        "regime_distribution": (
            _state_distribution(regime.loc[old_neutral_strong_env_mask], REGIME_NAMES)
            if "regime" in audit_df.columns
            else {}
        ),
        "metrics": _numeric_summary(
            old_neutral_strong_env_df.assign(
                trend_commit_gap=commit_gap.loc[old_neutral_strong_env_mask]
            ),
            [
                "trend_commit_gap",
                "trend_directional_evidence_score",
                "trend_confidence",
                "trend_bull_commit_score",
                "trend_bear_commit_score",
            ],
        ),
        "dominance_persistence_rates_pct": {
            "bull_dominant_2_of_3_rate_pct": (
                float(bull_dom_2of3.loc[old_neutral_strong_env_mask].mean() * 100.0)
                if len(old_neutral_strong_env_df)
                else 0.0
            ),
            "bear_dominant_2_of_3_rate_pct": (
                float(bear_dom_2of3.loc[old_neutral_strong_env_mask].mean() * 100.0)
                if len(old_neutral_strong_env_df)
                else 0.0
            ),
        },
        "dominant_side_distribution": _state_distribution(
            dominant_side_series.loc[old_neutral_strong_env_mask],
            {-1: "BEAR_DOM", 0: "TIED", 1: "BULL_DOM"},
        ),
    }

    future_states_3 = _future_window(strict, 3)
    future_states_5 = _future_window(strict, 5)
    weak_or_mixed_environment_mask = ~(
        strong_environment_mask | medium_environment_mask
    )
    weak_or_mixed_env_at_3 = weak_or_mixed_environment_mask.shift(-3)
    weak_or_mixed_env_at_5 = weak_or_mixed_environment_mask.shift(-5)

    def _regime_split(mask: pd.Series) -> dict[str, int]:
        if "regime" not in audit_df.columns:
            return {}
        return _state_distribution(regime.loc[mask], REGIME_NAMES)

    def _forward_candidate_summary(mask: pd.Series) -> dict[str, object]:
        scoped = audit_df.loc[mask].copy()
        if scoped.empty:
            return {
                "row_count": 0,
                "share_of_stale_neutral_neutral_pct": 0.0,
            }
        valid3 = mask & future_states_3.notna().all(axis=1)
        valid5 = mask & future_states_5.notna().all(axis=1)
        same_side_3 = future_states_3.eq(dominant_side_series, axis=0).any(axis=1)
        same_side_5 = future_states_5.eq(dominant_side_series, axis=0).any(axis=1)
        return {
            "row_count": int(len(scoped)),
            "share_of_stale_neutral_neutral_pct": (
                float(len(scoped) / int(stale_neutral_neutral_mask.sum()) * 100.0)
                if int(stale_neutral_neutral_mask.sum()) > 0
                else 0.0
            ),
            "by_regime": _regime_split(mask),
            "by_dominant_side": _state_distribution(
                dominant_side_series.loc[mask],
                {-1: "BEAR_DOM", 0: "TIED", 1: "BULL_DOM"},
            ),
            "metrics": _numeric_summary(
                scoped.assign(trend_commit_gap=commit_gap.loc[mask]),
                [
                    "bars_in_trend_state",
                    "trend_confidence",
                    "trend_commit_gap",
                    "trend_directional_evidence_score",
                ],
            ),
            "forward_outcomes_pct": {
                "stays_neutral_next_3_pct": (
                    float(future_states_3.loc[valid3].eq(0).all(axis=1).mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "stays_neutral_next_5_pct": (
                    float(future_states_5.loc[valid5].eq(0).all(axis=1).mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
                "becomes_directional_next_3_pct": (
                    float(future_states_3.loc[valid3].ne(0).any(axis=1).mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "becomes_directional_next_5_pct": (
                    float(future_states_5.loc[valid5].ne(0).any(axis=1).mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
                "agrees_with_dominant_side_next_3_pct": (
                    float(same_side_3.loc[valid3].mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "agrees_with_dominant_side_next_5_pct": (
                    float(same_side_5.loc[valid5].mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
            },
        }

    def _comparison_summary(
        mask: pd.Series, reference_side: pd.Series
    ) -> dict[str, object]:
        scoped = audit_df.loc[mask].copy()
        if scoped.empty:
            return {"row_count": 0}
        plus3 = strict.shift(-3)
        plus5 = strict.shift(-5)
        valid3 = mask & plus3.notna()
        valid5 = mask & plus5.notna()
        return {
            "row_count": int(len(scoped)),
            "metrics": _numeric_summary(
                scoped.assign(trend_commit_gap=commit_gap.loc[mask]),
                [
                    "trend_commit_gap",
                    "trend_directional_evidence_score",
                    "adx_strength",
                    "compression_score",
                    "trend_persistence_5",
                    "trend_persistence_20",
                ],
            ),
            "forward_profile_pct": {
                "directional_at_plus_3_pct": (
                    float(plus3.loc[valid3].ne(0).mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "directional_at_plus_5_pct": (
                    float(plus5.loc[valid5].ne(0).mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
                "same_side_at_plus_3_pct": (
                    float(
                        plus3.loc[valid3].eq(reference_side.loc[valid3]).mean() * 100.0
                    )
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "same_side_at_plus_5_pct": (
                    float(
                        plus5.loc[valid5].eq(reference_side.loc[valid5]).mean() * 100.0
                    )
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
            },
        }

    def _forward_metrics(
        mask: pd.Series, side_reference: pd.Series | None = None
    ) -> dict[str, float]:
        valid3 = mask & future_states_3.notna().all(axis=1)
        valid5 = mask & future_states_5.notna().all(axis=1)
        out = {
            "directional_next_3_pct": (
                float(future_states_3.loc[valid3].ne(0).any(axis=1).mean() * 100.0)
                if int(valid3.sum()) > 0
                else 0.0
            ),
            "directional_next_5_pct": (
                float(future_states_5.loc[valid5].ne(0).any(axis=1).mean() * 100.0)
                if int(valid5.sum()) > 0
                else 0.0
            ),
        }
        if side_reference is not None:
            out["same_side_next_3_pct"] = (
                float(
                    future_states_3.loc[valid3]
                    .eq(side_reference.loc[valid3], axis=0)
                    .any(axis=1)
                    .mean()
                    * 100.0
                )
                if int(valid3.sum()) > 0
                else 0.0
            )
            out["same_side_next_5_pct"] = (
                float(
                    future_states_5.loc[valid5]
                    .eq(side_reference.loc[valid5], axis=0)
                    .any(axis=1)
                    .mean()
                    * 100.0
                )
                if int(valid5.sum()) > 0
                else 0.0
            )
        return out

    def _event_recency_counts(mask: pd.Series) -> dict[str, int]:
        scoped = stale_recent_pattern.loc[mask]
        return {
            "bull_only_recent": int(scoped.eq("bull_only_recent").sum()),
            "bear_only_recent": int(scoped.eq("bear_only_recent").sum()),
            "both_recent": int(scoped.eq("both_recent").sum()),
            "neither_recent": int(scoped.eq("neither_recent").sum()),
        }

    def _forward_weaker_commit(step: int) -> pd.Series:
        bull_future = bull_commit.shift(-step)
        bear_future = bear_commit.shift(-step)
        weaker_is_bull = bull_commit.lt(bear_commit)
        return pd.Series(
            np.where(
                weaker_is_bull.to_numpy(),
                bull_future.to_numpy(dtype=float),
                bear_future.to_numpy(dtype=float),
            ),
            index=audit_df.index,
            dtype=float,
        )

    def _next_directional_segment_lengths(mask: pd.Series) -> list[int]:
        strict_vals = strict.fillna(0).astype(int).to_numpy()
        mask_vals = mask.fillna(False).to_numpy(dtype=bool)
        n = len(strict_vals)
        lengths: list[int] = []
        for idx in np.flatnonzero(mask_vals):
            j = idx + 1
            while j < n and strict_vals[j] == 0:
                j += 1
            if j >= n:
                continue
            state = strict_vals[j]
            k = j + 1
            while k < n and strict_vals[k] == state:
                k += 1
            lengths.append(int(k - j))
        return lengths

    stale_neutral_neutral_env_split: dict[str, object] = {
        "row_count": int(stale_neutral_neutral_mask.sum()),
        "strong_env": {
            "row_count": int(stale_neutral_strong_env_mask.sum()),
            "share_pct": (
                float(
                    stale_neutral_strong_env_mask.sum()
                    / stale_neutral_neutral_mask.sum()
                    * 100.0
                )
                if int(stale_neutral_neutral_mask.sum()) > 0
                else 0.0
            ),
            "regime_split": _regime_split(stale_neutral_strong_env_mask),
        },
        "medium_env": {
            "row_count": int(stale_neutral_medium_env_mask.sum()),
            "share_pct": (
                float(
                    stale_neutral_medium_env_mask.sum()
                    / stale_neutral_neutral_mask.sum()
                    * 100.0
                )
                if int(stale_neutral_neutral_mask.sum()) > 0
                else 0.0
            ),
            "regime_split": _regime_split(stale_neutral_medium_env_mask),
        },
        "weak_or_mixed_env": {
            "row_count": int(stale_neutral_weak_or_mixed_env_mask.sum()),
            "share_pct": (
                float(
                    stale_neutral_weak_or_mixed_env_mask.sum()
                    / stale_neutral_neutral_mask.sum()
                    * 100.0
                )
                if int(stale_neutral_neutral_mask.sum()) > 0
                else 0.0
            ),
            "regime_split": _regime_split(stale_neutral_weak_or_mixed_env_mask),
        },
    }

    stale_neutral_candidate_forward_audit: dict[str, object] = {
        "candidate_tight": _forward_candidate_summary(candidate_tight_mask),
        "candidate_medium": _forward_candidate_summary(candidate_medium_mask),
        "candidate_loose": _forward_candidate_summary(candidate_loose_mask),
    }

    stale_dominant_commit_bucket = pd.cut(
        dominant_commit.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.30, 0.45, np.inf],
        labels=["low_dom", "medium_dom", "high_dom"],
    )
    stale_weaker_commit_bucket = pd.cut(
        weaker_commit.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.15, 0.30, np.inf],
        labels=["low_weak", "medium_weak", "high_weak"],
    )
    stale_total_commit_bucket = pd.cut(
        total_commit_mass.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.45, 0.70, np.inf],
        labels=["low_mass", "medium_mass", "high_mass"],
    )
    stale_commit_gap_bucket = pd.cut(
        commit_gap.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, LOW_COMMIT_GAP_MAX, MEDIUM_COMMIT_GAP_MAX, np.inf],
        labels=["low_gap", "medium_gap", "high_gap"],
    )
    stale_commit_structure_audit: dict[str, object] = {}
    for bucket in ["low_gap", "medium_gap", "high_gap"]:
        bucket_mask = stale_neutral_neutral_mask.copy()
        bucket_mask.loc[stale_neutral_neutral_mask] = (
            stale_commit_gap_bucket.astype(str).eq(bucket).to_numpy()
        )
        scoped = audit_df.loc[bucket_mask].copy()
        stale_commit_structure_audit[bucket] = {
            "row_count": int(len(scoped)),
            "dominant_commit_bucket_distribution": _bucket_counts(
                stale_dominant_commit_bucket[
                    stale_commit_gap_bucket.astype(str).eq(bucket)
                ],
                ["low_dom", "medium_dom", "high_dom"],
            ),
            "weaker_commit_bucket_distribution": _bucket_counts(
                stale_weaker_commit_bucket[
                    stale_commit_gap_bucket.astype(str).eq(bucket)
                ],
                ["low_weak", "medium_weak", "high_weak"],
            ),
            "total_commit_mass_distribution": _bucket_counts(
                stale_total_commit_bucket[
                    stale_commit_gap_bucket.astype(str).eq(bucket)
                ],
                ["low_mass", "medium_mass", "high_mass"],
            ),
            "metrics": _numeric_summary(
                scoped.assign(
                    dominant_commit=dominant_commit.loc[bucket_mask],
                    weaker_commit=weaker_commit.loc[bucket_mask],
                    total_commit_mass=total_commit_mass.loc[bucket_mask],
                    trend_commit_gap=commit_gap.loc[bucket_mask],
                ),
                [
                    "dominant_commit",
                    "weaker_commit",
                    "total_commit_mass",
                    "trend_commit_gap",
                    "trend_directional_evidence_score",
                    "trend_confidence",
                ],
            ),
            "forward_outcomes_pct": _forward_metrics(bucket_mask, dominant_side_series),
        }
    stale_neutral_commit_structure_audit = stale_commit_structure_audit

    stale_recent_pattern = pd.Series(
        "neither_recent", index=audit_df.index, dtype=object
    )
    stale_recent_pattern.loc[bull_recent_event & ~bear_recent_event] = (
        "bull_only_recent"
    )
    stale_recent_pattern.loc[~bull_recent_event & bear_recent_event] = (
        "bear_only_recent"
    )
    stale_recent_pattern.loc[bull_recent_event & bear_recent_event] = "both_recent"
    dominant_side_recent_agrees = (
        dominant_side_series.eq(1) & bull_recent_event & ~bear_recent_event
    ) | (dominant_side_series.eq(-1) & bear_recent_event & ~bull_recent_event)
    dominant_side_recent_conflicts = (
        dominant_side_series.eq(1) & bear_recent_event
    ) | (dominant_side_series.eq(-1) & bull_recent_event)
    stale_neutral_event_recency_audit: dict[str, object] = {}
    for label in [
        "bull_only_recent",
        "bear_only_recent",
        "both_recent",
        "neither_recent",
    ]:
        mask = stale_neutral_neutral_mask & stale_recent_pattern.eq(label)
        stale_neutral_event_recency_audit[label] = {
            "row_count": int(mask.sum()),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }
    stale_neutral_event_recency_audit["dominant_side_recent_alignment"] = {
        "aligned_row_count": int(
            (stale_neutral_neutral_mask & dominant_side_recent_agrees).sum()
        ),
        "conflicted_row_count": int(
            (stale_neutral_neutral_mask & dominant_side_recent_conflicts).sum()
        ),
        "aligned_forward_outcomes_pct": _forward_metrics(
            stale_neutral_neutral_mask & dominant_side_recent_agrees,
            dominant_side_series,
        ),
        "conflicted_forward_outcomes_pct": _forward_metrics(
            stale_neutral_neutral_mask & dominant_side_recent_conflicts,
            dominant_side_series,
        ),
    }

    stale_contradiction_bucket = pd.cut(
        contradiction_penalty.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.25, 0.50, np.inf],
        labels=["low_contradiction", "medium_contradiction", "high_contradiction"],
    )
    stale_neutral_contradiction_audit: dict[str, object] = {}
    for label in ["low_contradiction", "medium_contradiction", "high_contradiction"]:
        mask = stale_neutral_neutral_mask.copy()
        mask.loc[stale_neutral_neutral_mask] = (
            stale_contradiction_bucket.astype(str).eq(label).to_numpy()
        )
        stale_neutral_contradiction_audit[label] = {
            "row_count": int(mask.sum()),
            "metrics": _numeric_summary(
                audit_df.loc[mask].assign(
                    contradiction_penalty=contradiction_penalty.loc[mask],
                    dominant_commit=dominant_commit.loc[mask],
                    weaker_commit=weaker_commit.loc[mask],
                ),
                [
                    "contradiction_penalty",
                    "dominant_commit",
                    "weaker_commit",
                    "trend_confidence",
                ],
            ),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }

    stale_dom_grid = pd.cut(
        dominant_commit.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.30, 0.45, np.inf],
        labels=["low_dom", "medium_dom", "high_dom"],
    )
    stale_weak_grid = pd.cut(
        weaker_commit.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.15, 0.30, np.inf],
        labels=["low_weak", "medium_weak", "high_weak"],
    )
    stale_neutral_dual_commit_grid: dict[str, object] = {}
    for dom_label in ["low_dom", "medium_dom", "high_dom"]:
        stale_neutral_dual_commit_grid[dom_label] = {}
        for weak_label in ["low_weak", "medium_weak", "high_weak"]:
            mask = stale_neutral_neutral_mask.copy()
            mask.loc[stale_neutral_neutral_mask] = (
                stale_dom_grid.astype(str).eq(dom_label).to_numpy()
                & stale_weak_grid.astype(str).eq(weak_label).to_numpy()
            )
            stale_neutral_dual_commit_grid[dom_label][weak_label] = {
                "row_count": int(mask.sum()),
                "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
            }

    stale_neutral_commit_component_audit: dict[str, object] = {}
    for bucket in ["low_gap", "medium_gap", "high_gap"]:
        bucket_mask = stale_neutral_neutral_mask.copy()
        bucket_mask.loc[stale_neutral_neutral_mask] = (
            stale_commit_gap_bucket.astype(str).eq(bucket).to_numpy()
        )
        scoped = audit_df.loc[bucket_mask].copy()
        stale_neutral_commit_component_audit[bucket] = {
            "row_count": int(len(scoped)),
            "metrics": _numeric_summary(
                scoped.assign(
                    dominant_commit=dominant_commit.loc[bucket_mask],
                    weaker_commit=weaker_commit.loc[bucket_mask],
                    total_commit_mass=total_commit_mass.loc[bucket_mask],
                    dominant_structure_pairs=dominant_structure_pairs.loc[bucket_mask],
                    weaker_structure_pairs=weaker_structure_pairs.loc[bucket_mask],
                    dominant_pair_age=dominant_pair_age.loc[bucket_mask],
                    weaker_pair_age=weaker_pair_age.loc[bucket_mask],
                    dominant_pressure_raw=dominant_pressure_raw.loc[bucket_mask],
                    weaker_pressure_raw=weaker_pressure_raw.loc[bucket_mask],
                    contradiction_penalty=contradiction_penalty.loc[bucket_mask],
                    trend_commit_gap_persist_3=commit_gap_persist_3.loc[bucket_mask],
                ),
                [
                    "dominant_commit",
                    "weaker_commit",
                    "total_commit_mass",
                    "dominant_structure_pairs",
                    "weaker_structure_pairs",
                    "dominant_pair_age",
                    "weaker_pair_age",
                    "dominant_pressure_raw",
                    "weaker_pressure_raw",
                    "trend_conf_event_quality",
                    "trend_conf_freshness",
                    "trend_conf_persistence",
                    "trend_commit_gap_persist_3",
                    "contradiction_penalty",
                ],
            ),
            "forward_outcomes_pct": _forward_metrics(bucket_mask, dominant_side_series),
        }

    stale_total_commit_mass_bucket = pd.cut(
        total_commit_mass.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.45, 0.70, np.inf],
        labels=["low_mass", "medium_mass", "high_mass"],
    )
    stale_neutral_commit_mass_vs_resolution_audit: dict[str, object] = {}
    for mass_label in ["low_mass", "medium_mass", "high_mass"]:
        stale_neutral_commit_mass_vs_resolution_audit[mass_label] = {}
        for gap_label in ["low_gap", "medium_gap", "high_gap"]:
            mask = stale_neutral_neutral_mask.copy()
            mask.loc[stale_neutral_neutral_mask] = (
                stale_total_commit_mass_bucket.astype(str).eq(mass_label).to_numpy()
                & stale_commit_gap_bucket.astype(str).eq(gap_label).to_numpy()
            )
            stale_neutral_commit_mass_vs_resolution_audit[mass_label][gap_label] = {
                "row_count": int(mask.sum()),
                "metrics": _numeric_summary(
                    audit_df.loc[mask].assign(
                        contradiction_penalty=contradiction_penalty.loc[mask],
                        trend_directional_evidence_score=directional_evidence.loc[mask],
                        total_commit_mass=total_commit_mass.loc[mask],
                        trend_commit_gap=commit_gap.loc[mask],
                    ),
                    [
                        "contradiction_penalty",
                        "trend_directional_evidence_score",
                        "total_commit_mass",
                        "trend_commit_gap",
                    ],
                ),
                "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
            }

    stale_weak_side_survival_audit: dict[str, object] = {}
    stale_weak_bucket = pd.cut(
        weaker_commit.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.08, 0.15, 0.30, np.inf],
        labels=["very_low_weak", "low_weak", "medium_weak", "high_weak"],
    )
    for dom_label in ["low_dom", "medium_dom", "high_dom"]:
        stale_weak_side_survival_audit[dom_label] = {}
        for weak_label in ["very_low_weak", "low_weak", "medium_weak"]:
            mask = stale_neutral_neutral_mask.copy()
            mask.loc[stale_neutral_neutral_mask] = (
                stale_dom_grid.astype(str).eq(dom_label).to_numpy()
                & stale_weak_bucket.astype(str).eq(weak_label).to_numpy()
            )
            stale_weak_side_survival_audit[dom_label][weak_label] = {
                "row_count": int(mask.sum()),
                "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
            }

    abs_strength_raw = pd.to_numeric(
        audit_df.get("trend_strength_raw", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    ).abs()
    abs_strength_ema = pd.to_numeric(
        audit_df.get("trend_strength_ema", pd.Series(np.nan, index=audit_df.index)),
        errors="coerce",
    ).abs()
    stale_strength_raw_bucket = pd.cut(
        abs_strength_raw.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=["low_raw_strength", "medium_raw_strength", "high_raw_strength"],
    )
    stale_strength_ema_bucket = pd.cut(
        abs_strength_ema.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["low_ema_strength", "medium_ema_strength", "high_ema_strength"],
    )
    stale_neutral_strength_vs_resolution_audit = {
        "raw_strength_buckets": {},
        "ema_strength_buckets": {},
    }
    for label in ["low_raw_strength", "medium_raw_strength", "high_raw_strength"]:
        mask = stale_neutral_neutral_mask.copy()
        mask.loc[stale_neutral_neutral_mask] = (
            stale_strength_raw_bucket.astype(str).eq(label).to_numpy()
        )
        stale_neutral_strength_vs_resolution_audit["raw_strength_buckets"][label] = {
            "row_count": int(mask.sum()),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }
    for label in ["low_ema_strength", "medium_ema_strength", "high_ema_strength"]:
        mask = stale_neutral_neutral_mask.copy()
        mask.loc[stale_neutral_neutral_mask] = (
            stale_strength_ema_bucket.astype(str).eq(label).to_numpy()
        )
        stale_neutral_strength_vs_resolution_audit["ema_strength_buckets"][label] = {
            "row_count": int(mask.sum()),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }

    stale_neutral_conflict_signature_audit: dict[str, object] = {}
    conflict_signatures = {
        "low_dom_low_weak": (
            stale_dom_grid.astype(str).eq("low_dom").to_numpy()
            & stale_weak_grid.astype(str).eq("low_weak").to_numpy()
        ),
        "medium_dom_low_weak": (
            stale_dom_grid.astype(str).eq("medium_dom").to_numpy()
            & stale_weak_grid.astype(str).eq("low_weak").to_numpy()
        ),
        "high_dom_low_weak": (
            stale_dom_grid.astype(str).eq("high_dom").to_numpy()
            & stale_weak_grid.astype(str).eq("low_weak").to_numpy()
        ),
        "medium_dom_medium_weak": (
            stale_dom_grid.astype(str).eq("medium_dom").to_numpy()
            & stale_weak_grid.astype(str).eq("medium_weak").to_numpy()
        ),
        "high_dom_medium_weak": (
            stale_dom_grid.astype(str).eq("high_dom").to_numpy()
            & stale_weak_grid.astype(str).eq("medium_weak").to_numpy()
        ),
    }
    for label, local_mask in conflict_signatures.items():
        mask = stale_neutral_neutral_mask.copy()
        mask.loc[stale_neutral_neutral_mask] = local_mask
        stale_neutral_conflict_signature_audit[label] = {
            "row_count": int(mask.sum()),
            "regime_split": _regime_split(mask),
            "event_recency_split": _event_recency_counts(mask),
            "metrics": _numeric_summary(
                audit_df.loc[mask].assign(
                    contradiction_penalty=contradiction_penalty.loc[mask],
                    dominant_commit=dominant_commit.loc[mask],
                    weaker_commit=weaker_commit.loc[mask],
                ),
                ["contradiction_penalty", "dominant_commit", "weaker_commit"],
            ),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }

    clean_asymmetry_base_mask = (
        stale_neutral_neutral_mask
        & dominant_commit.le(0.45)
        & weaker_commit.le(0.08)
        & total_commit_mass.le(0.45)
        & contradiction_penalty.le(0.50)
    )
    clean_asymmetry_cells = {
        "low_dom_low_contradiction": (
            clean_asymmetry_base_mask
            & dominant_commit.le(0.30)
            & contradiction_penalty.le(0.25)
        ),
        "low_dom_medium_contradiction": (
            clean_asymmetry_base_mask
            & dominant_commit.le(0.30)
            & contradiction_penalty.gt(0.25)
            & contradiction_penalty.le(0.50)
        ),
        "medium_dom_low_contradiction": (
            clean_asymmetry_base_mask
            & dominant_commit.gt(0.30)
            & dominant_commit.le(0.45)
            & contradiction_penalty.le(0.25)
        ),
        "medium_dom_medium_contradiction": (
            clean_asymmetry_base_mask
            & dominant_commit.gt(0.30)
            & dominant_commit.le(0.45)
            & contradiction_penalty.gt(0.25)
            & contradiction_penalty.le(0.50)
        ),
    }
    clean_asymmetry_candidate_audit: dict[str, object] = {}
    for label, mask in clean_asymmetry_cells.items():
        clean_asymmetry_candidate_audit[label] = {
            "row_count": int(mask.sum()),
            "regime_split": _regime_split(mask),
            "environment_split": {
                "strong_env": int((mask & strong_environment_mask).sum()),
                "medium_env": int((mask & medium_environment_mask).sum()),
                "weak_or_mixed_env": int((mask & weak_or_mixed_environment_mask).sum()),
            },
            "forward_outcomes_pct": {
                **_forward_metrics(mask, dominant_side_series),
                "still_neutral_next_3_pct": (
                    float(
                        future_states_3.loc[mask & future_states_3.notna().all(axis=1)]
                        .eq(0)
                        .all(axis=1)
                        .mean()
                        * 100.0
                    )
                    if int((mask & future_states_3.notna().all(axis=1)).sum()) > 0
                    else 0.0
                ),
                "still_neutral_next_5_pct": (
                    float(
                        future_states_5.loc[mask & future_states_5.notna().all(axis=1)]
                        .eq(0)
                        .all(axis=1)
                        .mean()
                        * 100.0
                    )
                    if int((mask & future_states_5.notna().all(axis=1)).sum()) > 0
                    else 0.0
                ),
            },
        }

    clean_asymmetry_age_audit: dict[str, object] = {}
    weaker_commit_plus_3 = _forward_weaker_commit(3)
    weaker_commit_plus_5 = _forward_weaker_commit(5)
    for label, age_mask in {
        "age_6_9": bars_in_state.ge(6) & bars_in_state.le(9),
        "age_10_14": bars_in_state.ge(10) & bars_in_state.le(14),
        "age_15_plus": bars_in_state.ge(15),
    }.items():
        mask = clean_asymmetry_base_mask & age_mask
        valid3 = mask & weaker_commit_plus_3.notna()
        valid5 = mask & weaker_commit_plus_5.notna()
        clean_asymmetry_age_audit[label] = {
            "row_count": int(mask.sum()),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
            "weak_side_revival_rate_pct": {
                "plus_3": (
                    float(weaker_commit_plus_3.loc[valid3].ge(0.15).mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "plus_5": (
                    float(weaker_commit_plus_5.loc[valid5].ge(0.15).mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
            },
        }

    clean_asymmetry_env_audit: dict[str, object] = {}
    for label, env_mask in {
        "strong_env": strong_environment_mask,
        "medium_env": medium_environment_mask,
        "weak_or_mixed_env": weak_or_mixed_environment_mask,
    }.items():
        mask = clean_asymmetry_base_mask & env_mask
        clean_asymmetry_env_audit[label] = {
            "row_count": int(mask.sum()),
            "forward_outcomes_pct": _forward_metrics(mask, dominant_side_series),
        }

    clean_asymmetry_segment_lengths = _next_directional_segment_lengths(
        clean_asymmetry_base_mask
    )
    clean_asymmetry_transition_risk_proxy = {
        "row_count": int(clean_asymmetry_base_mask.sum()),
        "current_segment_age": _numeric_summary(
            audit_df.loc[clean_asymmetry_base_mask],
            ["bars_in_trend_state"],
        ),
        "next_directional_segment_length": {
            "count": int(len(clean_asymmetry_segment_lengths)),
            "mean": (
                float(np.mean(clean_asymmetry_segment_lengths))
                if clean_asymmetry_segment_lengths
                else None
            ),
            "median": (
                float(np.median(clean_asymmetry_segment_lengths))
                if clean_asymmetry_segment_lengths
                else None
            ),
        },
        "fragmentation_proxy_pct": {
            "one_bar_realized_segment_share_pct": (
                float(
                    sum(length == 1 for length in clean_asymmetry_segment_lengths)
                    / len(clean_asymmetry_segment_lengths)
                    * 100.0
                )
                if clean_asymmetry_segment_lengths
                else 0.0
            ),
            "two_bar_realized_segment_share_pct": (
                float(
                    sum(length == 2 for length in clean_asymmetry_segment_lengths)
                    / len(clean_asymmetry_segment_lengths)
                    * 100.0
                )
                if clean_asymmetry_segment_lengths
                else 0.0
            ),
        },
    }

    comparable_bias_rows_mask = (
        neutral_with_directional_bias_mask
        & dominant_commit.le(0.45)
        & weaker_commit.le(0.08)
        & total_commit_mass.le(0.45)
    )
    clean_asymmetry_vs_bias_rows_audit = {
        "clean_asymmetry_candidates": {
            "row_count": int(clean_asymmetry_base_mask.sum()),
            "forward_outcomes_pct": _forward_metrics(
                clean_asymmetry_base_mask, dominant_side_series
            ),
            "metrics": _numeric_summary(
                audit_df.loc[clean_asymmetry_base_mask].assign(
                    contradiction_penalty=contradiction_penalty.loc[
                        clean_asymmetry_base_mask
                    ]
                ),
                ["trend_confidence", "contradiction_penalty"],
            ),
        },
        "neutral_with_directional_bias_comparison": {
            "row_count": int(comparable_bias_rows_mask.sum()),
            "forward_outcomes_pct": _forward_metrics(comparable_bias_rows_mask, bias),
            "metrics": _numeric_summary(
                audit_df.loc[comparable_bias_rows_mask].assign(
                    contradiction_penalty=contradiction_penalty.loc[
                        comparable_bias_rows_mask
                    ]
                ),
                ["trend_confidence", "contradiction_penalty"],
            ),
        },
    }

    def _live_safety_entry(
        *,
        source_function: str,
        source_columns: list[str],
        live_available: bool,
        forward_dependency: bool,
        verdict: str,
        note: str,
    ) -> dict[str, object]:
        return {
            "source_function": source_function,
            "source_columns": source_columns,
            "live_available": live_available,
            "forward_dependency": forward_dependency,
            "verdict": verdict,
            "note": note,
        }

    promotion_input_live_safety_audit = {
        "trend_state": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_state"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical strict state emitted on the live row.",
        ),
        "trend_bias_state": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_bias_state"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical bias state emitted on the live row.",
        ),
        "bars_in_trend_state": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["bars_in_trend_state"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Current-state dwell counter; uses only prior state history.",
        ),
        "trend_bull_commit_score": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_bull_commit_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical side-specific commit score on the current row.",
        ),
        "trend_bear_commit_score": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_bear_commit_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical side-specific commit score on the current row.",
        ),
        "trend_commit_gap": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_commit_gap"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical current-row gap already emitted by the engine.",
        ),
        "trend_directional_evidence_score": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_directional_evidence_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical current-row directional-evidence aggregate.",
        ),
        "trend_confidence": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_confidence"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Live-safe confidence output, admissible if used as a current-row filter only.",
        ),
        "trend_conf_contradiction_penalty": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_conf_contradiction_penalty"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical contradiction term already emitted by the engine.",
        ),
        "dominant_commit": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["trend_bull_commit_score", "trend_bear_commit_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Pure current-row max of live commit scores.",
        ),
        "weaker_commit": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["trend_bull_commit_score", "trend_bear_commit_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Pure current-row min of live commit scores.",
        ),
        "total_commit_mass": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["trend_bull_commit_score", "trend_bear_commit_score"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Pure current-row sum of live commit scores.",
        ),
        "trend_conf_structure_continuity": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_conf_structure_continuity"],
            live_available=True,
            forward_dependency=False,
            verdict="needs_confirmation",
            note="Appears causal, but should be traced to confirm the upstream structure aggregation is fully live-safe.",
        ),
        "trend_persistence_5": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_persistence_5"],
            live_available=True,
            forward_dependency=False,
            verdict="needs_confirmation",
            note="Looks causal from state history, but should be explicitly traced before using it in a detector.",
        ),
        "trend_persistence_20": _live_safety_entry(
            source_function=TREND_STATE_SOURCE,
            source_columns=["trend_persistence_20"],
            live_available=True,
            forward_dependency=False,
            verdict="needs_confirmation",
            note="Looks causal from state history, but should be explicitly traced before using it in a detector.",
        ),
        "environment_bucket": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=[
                "adx_strength",
                "ema_slope_strength",
                "structure_continuity",
                "compression_score",
            ],
            live_available=True,
            forward_dependency=False,
            verdict="needs_confirmation",
            note="Current validator gate is causal if all component metrics are causal; confirm each upstream metric before engine use.",
        ),
        "regime_state": _live_safety_entry(
            source_function=REGIME_SOURCE,
            source_columns=["regime"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical live-safe regime label already frozen as causal.",
        ),
        "regime_confidence": _live_safety_entry(
            source_function=REGIME_SOURCE,
            source_columns=["regime_confidence"],
            live_available=True,
            forward_dependency=False,
            verdict="confirmed",
            note="Canonical live-safe regime confidence already available on the current row.",
        ),
        "event_recency_components": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["trend_event"],
            live_available=True,
            forward_dependency=False,
            verdict="needs_confirmation",
            note="Rolling recent-event helpers are causal, but they are validator-derived and should be formalized if promoted into engine logic.",
        ),
        "same_side_next_3_5": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["future trend_state windows"],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Forward outcome metric for research only; may not enter deployed logic.",
        ),
        "directional_next_3_5": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["future trend_state windows"],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Forward conversion metric for research only; may not enter deployed logic.",
        ),
        "still_neutral_next_3_5": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["future trend_state windows"],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Forward neutral-survival metric for research only; may not enter deployed logic.",
        ),
        "next_directional_segment_length": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["future realized trend_state runs"],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Uses realized future segments and is not admissible for live detection.",
        ),
        "fragmentation_proxy": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=["future realized trend_state runs"],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Research-only fragmentation screen derived from future realized segments.",
        ),
        "research_bucket_labels": _live_safety_entry(
            source_function=VALIDATOR_DERIVED_SOURCE,
            source_columns=[
                "clean_asymmetry_candidate_audit",
                "candidate_tight",
                "low_dom_low_contradiction",
            ],
            live_available=False,
            forward_dependency=True,
            verdict="forbidden",
            note="Research buckets may inspire thresholds, but bucket labels themselves may not be used in deployed logic.",
        ),
    }

    regime_filter_masks = {
        "any_regime": pd.Series(True, index=audit_df.index),
        "transitional_or_trending": regime.ge(1),
        "trending_only": regime.eq(2),
    }
    stale_neutral_row_count = int(stale_neutral_neutral_mask.sum())
    selected_promotion_config = {
        "min_neutral_age": 10,
        "min_gap": 0.10,
        "min_evidence": 0.22,
        "max_total_mass": 0.40,
        "max_contradiction_penalty": 0.50,
        "regime_filter": "any_regime",
    }
    regime_name = selected_promotion_config["regime_filter"]
    regime_mask = regime_filter_masks[regime_name]
    min_neutral_age = int(selected_promotion_config["min_neutral_age"])
    min_gap = float(selected_promotion_config["min_gap"])
    min_evidence = float(selected_promotion_config["min_evidence"])
    max_total_mass = float(selected_promotion_config["max_total_mass"])
    max_contradiction_penalty = float(
        selected_promotion_config["max_contradiction_penalty"]
    )
    age_mask = stale_neutral_neutral_mask & bars_in_state.ge(min_neutral_age)
    promo_candidate_flag = (
        age_mask
        & commit_gap.ge(min_gap)
        & directional_evidence.ge(min_evidence)
        & contradiction_penalty.le(max_contradiction_penalty)
        & total_commit_mass.le(max_total_mass)
        & regime_mask
    )
    promo_candidate_side = dominant_side_series.where(promo_candidate_flag, other=0)
    score_gap = ((commit_gap - min_gap) / max(0.30 - min_gap, 1e-6)).clip(0.0, 1.0)
    score_evidence = (
        (directional_evidence - min_evidence) / max(0.70 - min_evidence, 1e-6)
    ).clip(0.0, 1.0)
    score_contradiction = (
        1.0 - contradiction_penalty / max(max_contradiction_penalty, 1e-6)
    ).clip(0.0, 1.0)
    score_mass = (1.0 - total_commit_mass / max(max_total_mass, 1e-6)).clip(0.0, 1.0)
    promo_candidate_score = (
        0.35 * score_gap
        + 0.30 * score_evidence
        + 0.20 * score_contradiction
        + 0.15 * score_mass
    ).where(promo_candidate_flag)
    bucket_id = (
        f"age{min_neutral_age}"
        f"_gap{min_gap:.2f}"
        f"_ev{min_evidence:.2f}"
        f"_mass{max_total_mass:.2f}"
        f"_contra{max_contradiction_penalty:.2f}"
        f"_{regime_name}"
    )
    row_count = int(promo_candidate_flag.sum())
    valid3 = promo_candidate_flag & future_states_3.notna().all(axis=1)
    valid5 = promo_candidate_flag & future_states_5.notna().all(axis=1)
    selected_prototype_row = {
        "promo_candidate_bucket_id": bucket_id,
        "min_neutral_age": min_neutral_age,
        "min_gap": min_gap,
        "min_evidence": min_evidence,
        "max_total_mass": max_total_mass,
        "max_contradiction_penalty": max_contradiction_penalty,
        "regime_filter": regime_name,
        "candidate_row_count": row_count,
        "share_of_stale_neutral_pct": (
            float(row_count / stale_neutral_row_count * 100.0)
            if stale_neutral_row_count > 0
            else 0.0
        ),
        "mean_promo_candidate_score": (
            float(promo_candidate_score.mean()) if row_count > 0 else None
        ),
        "directional_next_3_pct": (
            float(future_states_3.loc[valid3].ne(0).any(axis=1).mean() * 100.0)
            if int(valid3.sum()) > 0
            else 0.0
        ),
        "directional_next_5_pct": (
            float(future_states_5.loc[valid5].ne(0).any(axis=1).mean() * 100.0)
            if int(valid5.sum()) > 0
            else 0.0
        ),
        "same_side_next_3_pct": (
            float(
                future_states_3.loc[valid3]
                .eq(promo_candidate_side.loc[valid3], axis=0)
                .any(axis=1)
                .mean()
                * 100.0
            )
            if int(valid3.sum()) > 0
            else 0.0
        ),
        "same_side_next_5_pct": (
            float(
                future_states_5.loc[valid5]
                .eq(promo_candidate_side.loc[valid5], axis=0)
                .any(axis=1)
                .mean()
                * 100.0
            )
            if int(valid5.sum()) > 0
            else 0.0
        ),
        "regime_ranging_rows": int((promo_candidate_flag & regime.eq(0)).sum()),
        "regime_transitional_rows": int((promo_candidate_flag & regime.eq(1)).sum()),
        "regime_trending_rows": int((promo_candidate_flag & regime.eq(2)).sum()),
        "bear_dom_rows": int(
            (promo_candidate_flag & promo_candidate_side.eq(-1)).sum()
        ),
        "bull_dom_rows": int((promo_candidate_flag & promo_candidate_side.eq(1)).sum()),
    }
    confirmed_input_promotion_prototype_sweep = {
        "config_count": 1,
        "configs_with_rows": int(row_count > 0),
        "selected_config_id": bucket_id,
        "selected_config": selected_promotion_config,
        "selected_config_evaluation": pd.DataFrame([selected_prototype_row]),
    }

    strict_neutral_anomaly_mask = (
        stale_neutral_neutral_mask
        & strong_environment_mask
        & dominant_commit.ge(0.55)
        & weaker_commit.le(0.15)
        & directional_evidence.ge(DIRECTIONAL_EVIDENCE_HIGH)
        & pd.to_numeric(
            audit_df.get(
                "trend_persistence_5", pd.Series(np.nan, index=audit_df.index)
            ),
            errors="coerce",
        ).ge(0.80)
        & dominance_agrees
    )
    strict_neutral_anomaly_bucket = {
        "row_count": int(strict_neutral_anomaly_mask.sum()),
        "share_of_stale_neutral_neutral_pct": (
            float(
                strict_neutral_anomaly_mask.sum()
                / stale_neutral_neutral_mask.sum()
                * 100.0
            )
            if int(stale_neutral_neutral_mask.sum()) > 0
            else 0.0
        ),
        "metrics": _numeric_summary(
            audit_df.loc[strict_neutral_anomaly_mask].assign(
                dominant_commit=dominant_commit.loc[strict_neutral_anomaly_mask],
                weaker_commit=weaker_commit.loc[strict_neutral_anomaly_mask],
                total_commit_mass=total_commit_mass.loc[strict_neutral_anomaly_mask],
                contradiction_penalty=contradiction_penalty.loc[
                    strict_neutral_anomaly_mask
                ],
            ),
            [
                "dominant_commit",
                "weaker_commit",
                "total_commit_mass",
                "contradiction_penalty",
                "trend_directional_evidence_score",
                "trend_confidence",
            ],
        ),
        "forward_outcomes_pct": _forward_metrics(
            strict_neutral_anomaly_mask, dominant_side_series
        ),
    }

    candidate_regime = regime.loc[stale_neutral_promotion_candidate_mask]
    candidate_gap_bucket = pd.cut(
        commit_gap.loc[stale_neutral_promotion_candidate_mask],
        bins=[LOW_COMMIT_GAP_MAX, MEDIUM_COMMIT_GAP_MAX, np.inf],
        labels=["medium_gap", "high_gap"],
        include_lowest=True,
    )
    candidate_dom_side = dominant_side_series.loc[
        stale_neutral_promotion_candidate_mask
    ]
    future_neutral_3 = strict.shift(-3).eq(0)
    future_neutral_5 = strict.shift(-5).eq(0)
    candidate_valid_3 = (
        stale_neutral_promotion_candidate_mask & strict.shift(-3).notna()
    )
    candidate_valid_5 = (
        stale_neutral_promotion_candidate_mask & strict.shift(-5).notna()
    )
    stale_neutral_promotion_candidate_df = audit_df.loc[
        stale_neutral_promotion_candidate_mask
    ].copy()
    stale_neutral_promotion_candidate_audit: dict[str, object] = {
        "row_count": int(len(stale_neutral_promotion_candidate_df)),
        "candidate_share_of_old_neutral_strong_env": (
            float(
                len(stale_neutral_promotion_candidate_df)
                / len(old_neutral_strong_env_df)
            )
            if len(old_neutral_strong_env_df)
            else 0.0
        ),
        "by_regime": (
            _state_distribution(candidate_regime, REGIME_NAMES)
            if "regime" in audit_df.columns
            else {}
        ),
        "by_dominant_side": _state_distribution(
            candidate_dom_side,
            {-1: "BEAR_DOM", 0: "TIED", 1: "BULL_DOM"},
        ),
        "by_gap_bucket": {
            bucket: int(candidate_gap_bucket.astype(str).eq(bucket).sum())
            for bucket in ["medium_gap", "high_gap"]
        },
        "neutral_age": _numeric_summary(
            stale_neutral_promotion_candidate_df,
            ["bars_in_trend_state"],
        ),
        "still_neutral_after_n_bars_pct": {
            "after_3_bars_pct": (
                float(future_neutral_3.loc[candidate_valid_3].mean() * 100.0)
                if int(candidate_valid_3.sum()) > 0
                else 0.0
            ),
            "after_5_bars_pct": (
                float(future_neutral_5.loc[candidate_valid_5].mean() * 100.0)
                if int(candidate_valid_5.sum()) > 0
                else 0.0
            ),
        },
    }

    mature_directional_in_range_mask = directional_in_range_mask & bars_in_state.ge(6)
    mature_directional_in_range_df = audit_df.loc[
        mature_directional_in_range_mask
    ].copy()
    mature_directional_in_range_decay_audit: dict[str, object] = {
        "row_count": int(len(mature_directional_in_range_df)),
        "metrics": _numeric_summary(
            mature_directional_in_range_df.assign(
                trend_commit_gap=commit_gap.loc[mature_directional_in_range_mask]
            ),
            [
                "compression_score",
                "adx_strength",
                "trend_commit_gap",
                "trend_directional_evidence_score",
            ],
        ),
        "bias_distribution": _state_distribution(
            bias.loc[mature_directional_in_range_mask], STATE_NAMES
        ),
        "dominance_persistence_rates_pct": {
            "bull_dominant_2_of_3_rate_pct": (
                float(
                    bull_dom_2of3.loc[mature_directional_in_range_mask].mean() * 100.0
                )
                if len(mature_directional_in_range_df)
                else 0.0
            ),
            "bear_dominant_2_of_3_rate_pct": (
                float(
                    bear_dom_2of3.loc[mature_directional_in_range_mask].mean() * 100.0
                )
                if len(mature_directional_in_range_df)
                else 0.0
            ),
        },
    }

    stale_gap_bucket = pd.cut(
        commit_gap.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, LOW_COMMIT_GAP_MAX, MEDIUM_COMMIT_GAP_MAX, np.inf],
        labels=["low_gap", "medium_gap", "high_gap"],
    )
    stale_age_bucket = pd.cut(
        bars_in_state.loc[stale_neutral_neutral_mask],
        bins=[5, 9, 14, np.inf],
        labels=["age_6_9", "age_10_14", "age_15_plus"],
    )
    stale_evidence_bucket = pd.cut(
        directional_evidence.loc[stale_neutral_neutral_mask],
        bins=[-np.inf, 0.35, DIRECTIONAL_EVIDENCE_HIGH, np.inf],
        labels=["low_evidence", "medium_evidence", "high_evidence"],
    )
    stale_neutral_gap_age_grid: dict[str, object] = {}
    for gap_bucket in ["low_gap", "medium_gap", "high_gap"]:
        gap_mask = stale_neutral_neutral_mask.copy()
        gap_mask.loc[stale_neutral_neutral_mask] = (
            stale_gap_bucket.astype(str).eq(gap_bucket).to_numpy()
        )
        stale_neutral_gap_age_grid[gap_bucket] = {}
        for age_bucket in ["age_6_9", "age_10_14", "age_15_plus"]:
            cell_mask = gap_mask.copy()
            cell_mask.loc[stale_neutral_neutral_mask] = (
                stale_gap_bucket.astype(str).eq(gap_bucket).to_numpy()
                & stale_age_bucket.astype(str).eq(age_bucket).to_numpy()
            )
            valid3 = cell_mask & future_states_3.notna().all(axis=1)
            valid5 = cell_mask & future_states_5.notna().all(axis=1)
            cell_evidence = stale_evidence_bucket.astype(str)[
                stale_gap_bucket.astype(str).eq(gap_bucket)
                & stale_age_bucket.astype(str).eq(age_bucket)
            ]
            stale_neutral_gap_age_grid[gap_bucket][age_bucket] = {
                "row_count": int(cell_mask.sum()),
                "directional_evidence_bucket_distribution": {
                    bucket: int(cell_evidence.eq(bucket).sum())
                    for bucket in ["low_evidence", "medium_evidence", "high_evidence"]
                },
                "next_3_directional_conversion_pct": (
                    float(future_states_3.loc[valid3].ne(0).any(axis=1).mean() * 100.0)
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "next_5_directional_conversion_pct": (
                    float(future_states_5.loc[valid5].ne(0).any(axis=1).mean() * 100.0)
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
                "weak_or_mixed_env_at_plus_3_pct": (
                    float(
                        weak_or_mixed_env_at_3.loc[valid3].fillna(False).mean() * 100.0
                    )
                    if int(valid3.sum()) > 0
                    else 0.0
                ),
                "weak_or_mixed_env_at_plus_5_pct": (
                    float(
                        weak_or_mixed_env_at_5.loc[valid5].fillna(False).mean() * 100.0
                    )
                    if int(valid5.sum()) > 0
                    else 0.0
                ),
            }

    candidate_vs_range_decay_comparison: dict[str, object] = {
        "candidate_tight": _comparison_summary(
            candidate_tight_mask, dominant_side_series
        ),
        "candidate_medium": _comparison_summary(
            candidate_medium_mask, dominant_side_series
        ),
        "candidate_loose": _comparison_summary(
            candidate_loose_mask, dominant_side_series
        ),
        "mature_directional_in_range": _comparison_summary(
            mature_directional_in_range_mask, strict
        ),
    }

    if not neutral_with_directional_bias_df.empty:
        gap_bucket = pd.cut(
            commit_gap.loc[neutral_with_directional_bias_mask],
            bins=[-np.inf, LOW_COMMIT_GAP_MAX, MEDIUM_COMMIT_GAP_MAX, np.inf],
            labels=["low_gap", "medium_gap", "high_gap"],
        )
        age_bucket = pd.cut(
            bars_in_state.loc[neutral_with_directional_bias_mask],
            bins=[0, 2, 5, np.inf],
            labels=["age_1_2", "age_3_5", "age_6_plus"],
        )
        neutral_with_directional_bias_df = neutral_with_directional_bias_df.assign(
            trend_commit_gap=commit_gap.loc[
                neutral_with_directional_bias_mask
            ].to_numpy(),
            gap_bucket=gap_bucket.astype(str).to_numpy(),
            age_bucket=age_bucket.astype(str).to_numpy(),
        )
        neutral_with_directional_bias_audit = {
            "row_count": int(len(neutral_with_directional_bias_df)),
            "by_regime": (
                _confusion_matrix(
                    bias.loc[neutral_with_directional_bias_mask],
                    regime.loc[neutral_with_directional_bias_mask],
                    row_labels=STATE_NAMES,
                    col_labels=REGIME_NAMES,
                )
                if "regime" in audit_df.columns
                else {}
            ),
            "by_age_bucket": {
                bucket: _subset_profile(
                    neutral_with_directional_bias_df,
                    neutral_with_directional_bias_df["age_bucket"].eq(bucket),
                )
                for bucket in ["age_1_2", "age_3_5", "age_6_plus"]
            },
            "by_gap_bucket": {
                bucket: _subset_profile(
                    neutral_with_directional_bias_df,
                    neutral_with_directional_bias_df["gap_bucket"].eq(bucket),
                )
                for bucket in ["low_gap", "medium_gap", "high_gap"]
            },
        }
    else:
        neutral_with_directional_bias_audit = {"row_count": 0}

    commit_gap_bucket = pd.cut(
        commit_gap,
        bins=[-np.inf, LOW_COMMIT_GAP_MAX, MEDIUM_COMMIT_GAP_MAX, np.inf],
        labels=["low_gap", "medium_gap", "high_gap"],
    )
    commit_gap_audit: dict[str, object] = {}
    for bucket in ["low_gap", "medium_gap", "high_gap"]:
        bucket_mask = commit_gap_bucket.astype(str).eq(bucket)
        scoped = audit_df.loc[bucket_mask].copy()
        if scoped.empty:
            commit_gap_audit[bucket] = {"rows": 0}
            continue
        commit_gap_audit[bucket] = {
            "rows": int(len(scoped)),
            "state_distribution": _state_distribution(
                strict.loc[bucket_mask], STATE_NAMES
            ),
            "metrics": _numeric_summary(
                scoped.assign(trend_commit_gap=commit_gap.loc[bucket_mask]),
                [
                    "trend_commit_gap",
                    "trend_directional_evidence_score",
                    "trend_confidence",
                    "adx_strength",
                    "ema_slope_strength",
                    "structure_continuity",
                    "compression_score",
                    "bars_in_trend_state",
                ],
            ),
        }

    neutral_age_bucket = pd.cut(
        bars_in_state.loc[neutral_mask],
        bins=[0, 2, 5, np.inf],
        labels=["age_1_2", "age_3_5", "age_6_plus"],
    )
    neutral_age_df = audit_df.loc[neutral_mask].copy()
    if not neutral_age_df.empty:
        neutral_age_df = neutral_age_df.assign(
            trend_commit_gap=commit_gap.loc[neutral_mask].to_numpy(),
            age_bucket=neutral_age_bucket.astype(str).to_numpy(),
        )
        neutral_age_audit = {
            bucket: {
                "rows": int(neutral_age_df["age_bucket"].eq(bucket).sum()),
                "metrics": _numeric_summary(
                    neutral_age_df.loc[neutral_age_df["age_bucket"].eq(bucket)],
                    [
                        "trend_commit_gap",
                        "trend_directional_evidence_score",
                        "trend_confidence",
                        "adx_strength",
                        "ema_slope_strength",
                        "structure_continuity",
                        "compression_score",
                        "bars_in_trend_state",
                    ],
                ),
                "trend_bias_state_distribution": _state_distribution(
                    bias.loc[
                        neutral_age_df.index[neutral_age_df["age_bucket"].eq(bucket)]
                    ],
                    STATE_NAMES,
                ),
            }
            for bucket in ["age_1_2", "age_3_5", "age_6_plus"]
        }
    else:
        neutral_age_audit = {}

    semantic_buckets = {
        "neutral_with_high_bull_commit": _subset_profile(
            audit_df, neutral_high_bull_commit_mask
        ),
        "neutral_with_high_bear_commit": _subset_profile(
            audit_df, neutral_high_bear_commit_mask
        ),
        "neutral_with_high_directional_evidence_strict": _subset_profile(
            audit_df, neutral_directional_evidence_strict_mask
        ),
        "neutral_with_high_directional_evidence_broad": _subset_profile(
            audit_df, neutral_directional_evidence_broad_mask
        ),
        "directional_with_low_commit_gap": _subset_profile(
            audit_df, directional_low_commit_gap_mask
        ),
        "directional_with_low_continuity": _subset_profile(
            audit_df, directional_low_continuity_mask
        ),
        "directional_with_low_confidence": _subset_profile(
            audit_df, directional_low_conf_mask
        ),
        "neutral_with_bias_and_strong_strength": _subset_profile(
            audit_df, neutral_bias_strength_mask
        ),
    }

    current_row = audit_df.loc[
        pd.to_numeric(audit_df["trend_state"], errors="coerce").notna()
    ].tail(1)
    current_snapshot = {}
    if not current_row.empty:
        row = current_row.iloc[0]
        current_snapshot = {
            "timestamp": str(row["timestamp"]),
            "trend_state": STATE_NAMES.get(
                int(
                    pd.to_numeric(pd.Series([row["trend_state"]]), errors="coerce")
                    .fillna(0)
                    .iloc[0]
                ),
                str(row["trend_state"]),
            ),
            "trend_bias_state": STATE_NAMES.get(
                int(
                    pd.to_numeric(
                        pd.Series([row.get("trend_bias_state", 0)]), errors="coerce"
                    )
                    .fillna(0)
                    .iloc[0]
                ),
                str(row.get("trend_bias_state", 0)),
            ),
            "trend_confidence": float(
                pd.to_numeric(
                    pd.Series([row.get("trend_confidence", np.nan)]), errors="coerce"
                ).iloc[0]
            ),
            "trend_strength_ema": float(
                pd.to_numeric(
                    pd.Series([row.get("trend_strength_ema", np.nan)]), errors="coerce"
                ).iloc[0]
            ),
            "bars_in_trend_state": int(
                pd.to_numeric(
                    pd.Series([row.get("bars_in_trend_state", 0)]), errors="coerce"
                )
                .fillna(0)
                .iloc[0]
            ),
        }

    transition_windows = _sample_transition_windows(
        plot_df,
        transition_col="trend_state_changed",
        n_windows=n_windows,
    )

    html_path = (
        plot_trend_state_validation(
            plot_df,
            outpath=outpath,
            title=title,
        )
        if outpath is not None
        else None
    )

    summary = {
        "row_count": int(len(audit_df)),
        "current_trend_snapshot": current_snapshot,
        "strict_state_counts": strict_state_counts,
        "strict_state_pct": strict_state_pct,
        "bias_state_counts": bias_state_counts,
        "bias_state_pct": bias_state_pct,
        "transition_count": transition_count,
        "avg_bias_score_by_strict_state": avg_bias_score_by_strict_state,
        "avg_age_by_strict_state": avg_age_by_strict_state,
        "avg_strength_raw_by_strict_state": avg_strength_raw_by_strict_state,
        "avg_strength_ema_by_strict_state": avg_strength_ema_by_strict_state,
        "strict_neutral_rows": strict_neutral_rows,
        "bias_carry_rows": bias_carry_rows,
        "strict_bull_not_ready_rows": strict_bull_not_ready_rows,
        "strict_bear_not_ready_rows": strict_bear_not_ready_rows,
        "confidence_distribution": confidence_distribution,
        "transitions_table": transitions_table,
        "bias_transitions_table": bias_transitions_table,
        "transition_matrix": transitions_table,
        "bias_transition_matrix": bias_transitions_table,
        "dwell_diagnostics": segment_stats,
        "duration_stats": segment_stats["duration_stats"],
        "confidence_by_state": confidence_by_state,
        "confidence_ordering_check": confidence_ordering_check,
        "confidence_separation_check": confidence_separation_check,
        "neutral_confidence_cap_check": neutral_confidence_cap_check,
        "strength_by_state": strength_by_state,
        "commitment_by_state": commitment_by_state,
        "bias_interaction": bias_interaction,
        "regime_interaction": regime_interaction,
        "neutral_overuse_audit": neutral_overuse_audit,
        "neutral_confidence_audit": neutral_confidence_audit,
        "neutral_in_trend_audit": neutral_in_trend_audit,
        "directional_in_range_audit": directional_in_range_audit,
        "old_neutral_strong_env_audit": old_neutral_strong_env_audit,
        "stale_neutral_neutral_env_split": stale_neutral_neutral_env_split,
        "stale_neutral_candidate_forward_audit": stale_neutral_candidate_forward_audit,
        "stale_neutral_gap_age_grid": stale_neutral_gap_age_grid,
        "candidate_vs_range_decay_comparison": candidate_vs_range_decay_comparison,
        "stale_neutral_commit_structure_audit": stale_neutral_commit_structure_audit,
        "stale_neutral_commit_component_audit": stale_neutral_commit_component_audit,
        "stale_neutral_commit_mass_vs_resolution_audit": stale_neutral_commit_mass_vs_resolution_audit,
        "stale_neutral_weak_side_survival_audit": stale_weak_side_survival_audit,
        "stale_neutral_strength_vs_resolution_audit": stale_neutral_strength_vs_resolution_audit,
        "stale_neutral_conflict_signature_audit": stale_neutral_conflict_signature_audit,
        "clean_asymmetry_candidate_audit": clean_asymmetry_candidate_audit,
        "clean_asymmetry_age_audit": clean_asymmetry_age_audit,
        "clean_asymmetry_env_audit": clean_asymmetry_env_audit,
        "clean_asymmetry_transition_risk_proxy": clean_asymmetry_transition_risk_proxy,
        "clean_asymmetry_vs_bias_rows_audit": clean_asymmetry_vs_bias_rows_audit,
        "promotion_input_live_safety_audit": promotion_input_live_safety_audit,
        "confirmed_input_promotion_prototype_sweep": confirmed_input_promotion_prototype_sweep,
        "stale_neutral_event_recency_audit": stale_neutral_event_recency_audit,
        "stale_neutral_contradiction_audit": stale_neutral_contradiction_audit,
        "stale_neutral_dual_commit_grid": stale_neutral_dual_commit_grid,
        "strict_neutral_anomaly_bucket": strict_neutral_anomaly_bucket,
        "stale_neutral_promotion_candidate_audit": stale_neutral_promotion_candidate_audit,
        "mature_directional_in_range_decay_audit": mature_directional_in_range_decay_audit,
        "neutral_with_directional_bias_audit": neutral_with_directional_bias_audit,
        "commit_gap_audit": commit_gap_audit,
        "neutral_age_audit": neutral_age_audit,
        "semantic_buckets": semantic_buckets,
        "structure_loss_bull_rows": structure_loss_bull_rows,
        "structure_loss_bear_rows": structure_loss_bear_rows,
        "emerging_bull_rows": emerging_bull_rows,
        "emerging_bear_rows": emerging_bear_rows,
        "regime_phase_counts": regime_phase_counts,
    }

    return {
        "summary": summary,
        "transition_windows": transition_windows,
        "html_path": html_path,
    }


def plot_trend_state_validation(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
    title: str = "Trend State Validation",
) -> Path:
    out = _ensure_datetime(df).copy()

    for col in ["trend_bias_score_live", "trend_strength_raw", "trend_strength_ema"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    outpath = Path(outpath)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.56, 0.14, 0.12, 0.18],
        subplot_titles=(
            "Price + Swings + Strict Regime",
            "Strict State vs Bias State",
            "Trend Confidence",
            "Bias / Strength Scores",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=out["timestamp"],
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            name="OHLC",
            increasing_line_color="#00cc96",
            increasing_fillcolor="#00cc96",
            decreasing_line_color="#ef553b",
            decreasing_fillcolor="#ef553b",
        ),
        row=1,
        col=1,
    )

    _add_state_background(
        fig,
        out,
        row=1,
        col=1,
        state_col="trend_state",
        opacity=0.10,
        max_runs=180,
    )

    if "swing_high" in out.columns:
        sh = out[out["swing_high"] == 1]
        fig.add_trace(
            go.Scatter(
                x=sh["timestamp"],
                y=sh["high"],
                mode="markers",
                name="Swing High (Origin)",
                marker=dict(symbol="triangle-down", size=8, color="#ef553b"),
            ),
            row=1,
            col=1,
        )

    if "swing_low" in out.columns:
        sl = out[out["swing_low"] == 1]
        fig.add_trace(
            go.Scatter(
                x=sl["timestamp"],
                y=sl["low"],
                mode="markers",
                name="Swing Low (Origin)",
                marker=dict(symbol="triangle-up", size=8, color="#00cc96"),
            ),
            row=1,
            col=1,
        )

    if {"swing_high_confirm_flag", "swing_high_confirm_price"}.issubset(out.columns):
        shc = out[out["swing_high_confirm_flag"] == 1]
        fig.add_trace(
            go.Scatter(
                x=shc["timestamp"],
                y=shc["swing_high_confirm_price"],
                mode="markers",
                name="Swing High (Confirm)",
                marker=dict(symbol="x", size=9, color="#ffa15a"),
            ),
            row=1,
            col=1,
        )

    if {"swing_low_confirm_flag", "swing_low_confirm_price"}.issubset(out.columns):
        slc = out[out["swing_low_confirm_flag"] == 1]
        fig.add_trace(
            go.Scatter(
                x=slc["timestamp"],
                y=slc["swing_low_confirm_price"],
                mode="markers",
                name="Swing Low (Confirm)",
                marker=dict(symbol="x", size=9, color="#ab63fa"),
            ),
            row=1,
            col=1,
        )

    if "last_swing_high" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_high"],
                mode="lines",
                name="Last Confirmed Swing High",
                line=dict(width=1, dash="dot", color="#ef553b"),
            ),
            row=1,
            col=1,
        )

    if "last_swing_low" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["last_swing_low"],
                mode="lines",
                name="Last Confirmed Swing Low",
                line=dict(width=1, dash="dot", color="#00cc96"),
            ),
            row=1,
            col=1,
        )

    if "trend_state_changed" in out.columns:
        tc = out[out["trend_state_changed"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tc["timestamp"],
                y=tc["close"],
                mode="markers",
                name="Strict Transition",
                marker=dict(symbol="x", size=10, color="#ffffff"),
            ),
            row=1,
            col=1,
        )

    if "trend_bias_changed" in out.columns:
        bc = out[out["trend_bias_changed"] == 1]
        fig.add_trace(
            go.Scatter(
                x=bc["timestamp"],
                y=bc["close"],
                mode="markers",
                name="Bias Transition",
                marker=dict(symbol="diamond", size=7, color="#19d3f3"),
            ),
            row=1,
            col=1,
        )

    if "trend_structure_loss_bull" in out.columns:
        tlb = out[out["trend_structure_loss_bull"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tlb["timestamp"],
                y=tlb["close"],
                mode="markers",
                name="Bull Structure Loss",
                marker=dict(symbol="circle-open", size=9, color="#00cc96"),
            ),
            row=1,
            col=1,
        )

    if "trend_structure_loss_bear" in out.columns:
        tlbr = out[out["trend_structure_loss_bear"] == 1]
        fig.add_trace(
            go.Scatter(
                x=tlbr["timestamp"],
                y=tlbr["close"],
                mode="markers",
                name="Bear Structure Loss",
                marker=dict(symbol="circle-open", size=9, color="#ef553b"),
            ),
            row=1,
            col=1,
        )

    if "trend_emerging_bull" in out.columns:
        eb = out[out["trend_emerging_bull"] == 1]
        fig.add_trace(
            go.Scatter(
                x=eb["timestamp"],
                y=eb["close"],
                mode="markers",
                name="Emerging Bull",
                marker=dict(symbol="star", size=9, color="#00cc96"),
            ),
            row=1,
            col=1,
        )

    if "trend_emerging_bear" in out.columns:
        er = out[out["trend_emerging_bear"] == 1]
        fig.add_trace(
            go.Scatter(
                x=er["timestamp"],
                y=er["close"],
                mode="markers",
                name="Emerging Bear",
                marker=dict(symbol="star", size=9, color="#ef553b"),
            ),
            row=1,
            col=1,
        )

    if "trend_state" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_state"].astype(float),
                mode="lines",
                name="Strict State",
                line=dict(width=2, color="#ffffff"),
            ),
            row=2,
            col=1,
        )

    if "trend_bias_state" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_bias_state"].astype(float),
                mode="lines",
                name="Bias State",
                line=dict(width=2, dash="dot", color="#19d3f3"),
            ),
            row=2,
            col=1,
        )

    fig.add_hline(
        y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=2, col=1
    )

    if "trend_confidence" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_confidence"].astype(float),
                mode="lines",
                name="Confidence",
                line=dict(width=2, color="#ffa15a"),
            ),
            row=3,
            col=1,
        )

    fig.add_hline(
        y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=3, col=1
    )
    fig.add_hline(
        y=0.45, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=3, col=1
    )
    fig.add_hline(
        y=0.60, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=3, col=1
    )
    fig.add_hline(
        y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=3, col=1
    )

    if "trend_bias_score_live" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_bias_score_live"],
                mode="lines",
                name="Bias Score Live",
                line=dict(width=2, color="#19d3f3"),
            ),
            row=4,
            col=1,
        )

    if "trend_strength_raw" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_strength_raw"],
                mode="lines",
                name="Trend Strength Raw",
                line=dict(width=1, color="#b6e880"),
            ),
            row=4,
            col=1,
        )

    if "trend_strength_ema" in out.columns:
        fig.add_trace(
            go.Scatter(
                x=out["timestamp"],
                y=out["trend_strength_ema"],
                mode="lines",
                name="Trend Strength EMA",
                line=dict(width=3, color="#fecb52"),
            ),
            row=4,
            col=1,
        )

    fig.add_hline(
        y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=4, col=1
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, side="right")
    fig.update_yaxes(
        title_text="State",
        row=2,
        col=1,
        tickmode="array",
        tickvals=[-1, 0, 1],
        range=[-1.25, 1.25],
    )
    fig.update_yaxes(
        title_text="Conf",
        row=3,
        col=1,
        range=[-0.05, 1.05],
    )
    fig.update_yaxes(
        title_text="Bias / Strength",
        row=4,
        col=1,
        range=[-1.05, 1.05],
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=1200,
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="white"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(outpath))
    return outpath
