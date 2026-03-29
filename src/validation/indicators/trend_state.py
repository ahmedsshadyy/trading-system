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


def validate_trend_state(
    df: pd.DataFrame,
    *,
    outpath: str | Path,
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
    strong_environment_mask = (
        adx_strength.ge(STRONG_ENV_ADX_MIN)
        & ema_slope_strength.ge(STRONG_ENV_SLOPE_MIN)
        & structure_continuity.ge(STRONG_ENV_CONTINUITY_MIN)
        & compression_score.le(STRONG_ENV_COMPRESSION_MAX)
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
    old_neutral_strong_env_mask = (
        neutral_mask & bias.eq(0) & bars_in_state.ge(6) & strong_environment_mask
    )
    stale_neutral_promotion_candidate_mask = (
        old_neutral_strong_env_mask
        & commit_gap.ge(LOW_COMMIT_GAP_MAX)
        & (
            pd.to_numeric(
                audit_df.get(
                    "trend_bull_dominant_2_of_3", pd.Series(0, index=audit_df.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
            | pd.to_numeric(
                audit_df.get(
                    "trend_bear_dominant_2_of_3", pd.Series(0, index=audit_df.index)
                ),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )
    )

    neutral_in_trend_df = audit_df.loc[neutral_in_trend_mask].copy()
    directional_in_range_df = audit_df.loc[directional_in_range_mask].copy()
    neutral_with_directional_bias_df = audit_df.loc[
        neutral_with_directional_bias_mask
    ].copy()
    old_neutral_strong_env_df = audit_df.loc[old_neutral_strong_env_mask].copy()
    stale_neutral_promotion_candidate_df = audit_df.loc[
        stale_neutral_promotion_candidate_mask
    ].copy()

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

    html_path = plot_trend_state_validation(
        plot_df,
        outpath=outpath,
        title=title,
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
