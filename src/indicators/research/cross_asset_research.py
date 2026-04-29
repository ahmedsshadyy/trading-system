from __future__ import annotations

import math
import re
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.indicators.features.cross_asset import (
    HORIZONS_BY_TIMEFRAME,
    LAG_SCAN_LAGS,
    aligned_timestamp_for_instrument,
)

_NORMAL = NormalDist()
_RET_RAW_RE = re.compile(r"^ret_raw_(.+)$")
_RET_VOL_RE = re.compile(r"^ret_vol_(.+)$")
_CORR_RE = re.compile(r"^corr_(?!sig_class_|stability_class_)(.+)__(.+)__w(\d+)$")
_LAG_RE = re.compile(r"^lagcorr_best_(.+)__(.+)__w(\d+)$")

PRIMARY_MATRIX_FAMILIES = (
    "contemporaneous",
    "lag_1",
    "lag_k",
    "best_lag",
    "stability_zscore",
)
PRIMARY_METRICS = (
    "pearson_correlation",
    "spearman_correlation",
    "fisher_z",
    "r_squared",
    "classic_t_stat",
    "classic_p_value",
    "hac_t_stat",
    "hac_p_value",
    "fdr_q_value",
    "overlap_count",
    "effective_sample_size",
    "sign_stability",
    "rolling_correlation_std",
    "stability_zscore",
    "best_lag_persistence_score",
)


def _parse(pattern: re.Pattern[str], column: str) -> tuple[str, str, int] | None:
    match = pattern.match(column)
    if match is None:
        return None
    left, right, window = match.groups()
    return left, right, int(window)


def _available_symbols(context: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    for column in context.columns:
        match = _RET_RAW_RE.match(column)
        if match is not None:
            symbols.add(match.group(1))
    return sorted(symbols)


def _pair_id(left: str, right: str) -> str:
    return f"{left}__{right}"


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _safe_int(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result


def _normal_pvalue_from_t(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return float(2.0 * (1.0 - _NORMAL.cdf(abs(float(t_stat)))))


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return _safe_float(x_rank.corr(y_rank))


def _fisher_z(corr: float) -> float:
    if not np.isfinite(corr):
        return float("nan")
    clipped = float(np.clip(corr, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def _effective_sample_size(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    x_ac1 = _safe_float(x.autocorr(lag=1))
    y_ac1 = _safe_float(y.autocorr(lag=1))
    if not np.isfinite(x_ac1) or not np.isfinite(y_ac1):
        return float(len(x))
    denom = 1.0 + (x_ac1 * y_ac1)
    if denom <= 0.0:
        return float(len(x))
    eff_n = len(x) * (1.0 - (x_ac1 * y_ac1)) / denom
    return float(np.clip(eff_n, 1.0, float(len(x))))


def _newey_west_bandwidth(n_obs: int) -> int:
    if n_obs <= 1:
        return 0
    return max(1, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def _newey_west_t_stat(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    n_obs = int(len(x_arr))
    if n_obs < 5:
        return float("nan"), float("nan")

    design = np.column_stack([np.ones(n_obs, dtype=float), x_arr])
    xtx = design.T @ design
    try:
        xtx_inv = np.linalg.pinv(xtx)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    beta = xtx_inv @ (design.T @ y_arr)
    resid = y_arr - (design @ beta)
    bandwidth = _newey_west_bandwidth(n_obs)
    omega = np.zeros((2, 2), dtype=float)
    for lag in range(bandwidth + 1):
        weight = 1.0 if lag == 0 else 1.0 - (lag / (bandwidth + 1.0))
        gamma = np.zeros((2, 2), dtype=float)
        for t in range(lag, n_obs):
            xt = design[t : t + 1].T
            ut = resid[t]
            if lag == 0:
                gamma += (ut * ut) * (xt @ xt.T)
                continue
            xlag = design[t - lag : t - lag + 1].T
            ulag = resid[t - lag]
            gamma += (ut * ulag) * (xt @ xlag.T)
        if lag == 0:
            omega += gamma
        else:
            omega += weight * (gamma + gamma.T)
    cov = xtx_inv @ omega @ xtx_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    if se <= 0.0 or not np.isfinite(se):
        return float("nan"), float("nan")
    t_stat = float(beta[1] / se)
    return t_stat, _normal_pvalue_from_t(t_stat)


def _slice_tail_overlap(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int,
    lag: int = 0,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y.shift(lag), errors="coerce"),
        }
    ).dropna()
    if aligned.empty:
        return (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )
    tail = aligned.tail(window).reset_index(drop=True)
    return tail["x"].astype(float), tail["y"].astype(float)


def _corr_stats(
    x: pd.Series,
    y: pd.Series,
) -> dict[str, float]:
    n_obs = int(min(len(x), len(y)))
    if n_obs < 3:
        return {
            "pearson_correlation": float("nan"),
            "spearman_correlation": float("nan"),
            "fisher_z": float("nan"),
            "r_squared": float("nan"),
            "classic_t_stat": float("nan"),
            "classic_p_value": float("nan"),
            "hac_t_stat": float("nan"),
            "hac_p_value": float("nan"),
            "overlap_count": float(n_obs),
            "effective_sample_size": float("nan"),
        }

    pearson = _safe_float(x.corr(y))
    if not np.isfinite(pearson):
        return {
            "pearson_correlation": float("nan"),
            "spearman_correlation": float("nan"),
            "fisher_z": float("nan"),
            "r_squared": float("nan"),
            "classic_t_stat": float("nan"),
            "classic_p_value": float("nan"),
            "hac_t_stat": float("nan"),
            "hac_p_value": float("nan"),
            "overlap_count": float(n_obs),
            "effective_sample_size": float("nan"),
        }
    denom = max(1.0 - (pearson * pearson), 1e-12)
    classic_t = pearson * math.sqrt(max(n_obs - 2, 1) / denom)
    hac_t, hac_p = _newey_west_t_stat(x, y)
    return {
        "pearson_correlation": pearson,
        "spearman_correlation": _spearman_corr(x, y),
        "fisher_z": _fisher_z(pearson),
        "r_squared": float(pearson * pearson),
        "classic_t_stat": float(classic_t),
        "classic_p_value": _normal_pvalue_from_t(classic_t),
        "hac_t_stat": _safe_float(hac_t),
        "hac_p_value": _safe_float(hac_p),
        "overlap_count": float(n_obs),
        "effective_sample_size": _effective_sample_size(x, y),
    }


def _rolling_series_stats(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int,
) -> dict[str, float]:
    corr = (
        pd.to_numeric(x, errors="coerce")
        .rolling(window)
        .corr(pd.to_numeric(y, errors="coerce"))
    )
    valid = corr.dropna()
    if valid.empty:
        return {
            "latest_corr": float("nan"),
            "rolling_correlation_mean": float("nan"),
            "rolling_correlation_std": float("nan"),
            "stability_zscore": float("nan"),
            "sign_flip_rate": float("nan"),
            "sign_stability": float("nan"),
        }
    latest = float(valid.iloc[-1])
    std = float(valid.std(ddof=0))
    sign_series = np.sign(valid.to_numpy(dtype=float))
    if len(sign_series) <= 1:
        sign_flip_rate = 0.0
    else:
        sign_flip_rate = float(np.mean(sign_series[1:] != sign_series[:-1]))
    mean = float(valid.mean())
    zscore = (latest - mean) / std if std > 0.0 else float("nan")
    return {
        "latest_corr": latest,
        "rolling_correlation_mean": mean,
        "rolling_correlation_std": std,
        "stability_zscore": _safe_float(zscore),
        "sign_flip_rate": sign_flip_rate,
        "sign_stability": float(np.clip(1.0 - sign_flip_rate, 0.0, 1.0)),
    }


def _rolling_best_lag_series(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int,
    lags: tuple[int, ...],
) -> tuple[pd.Series, pd.Series]:
    corr_by_lag = {
        lag: pd.to_numeric(x, errors="coerce")
        .rolling(window)
        .corr(pd.to_numeric(y.shift(lag), errors="coerce"))
        for lag in lags
    }
    corr_frame = pd.DataFrame(corr_by_lag, index=x.index, dtype=float)
    valid_mask = corr_frame.notna().any(axis=1)
    abs_corr = corr_frame.abs().fillna(-np.inf)
    best_lag = abs_corr.idxmax(axis=1).astype(float).where(valid_mask, np.nan)
    best_score = pd.Series(np.nan, index=x.index, dtype=float)
    for lag in lags:
        best_score = best_score.where(best_lag != float(lag), corr_frame[lag])
    return best_score, best_lag


def _apply_bh_qvalues(stats_df: pd.DataFrame) -> pd.DataFrame:
    if stats_df.empty or "hac_p_value" not in stats_df.columns:
        return stats_df
    out = stats_df.copy()
    out["fdr_q_value"] = np.nan
    group_cols = [
        "timestamp",
        "timeframe",
        "context_type",
        "context_value",
        "window",
        "matrix_family",
        "return_mode",
    ]
    for _, group in out.groupby(group_cols, dropna=False):
        pvals = pd.to_numeric(group["hac_p_value"], errors="coerce")
        valid = pvals.dropna()
        if valid.empty:
            continue
        ranks = valid.rank(method="first")
        adjusted = (valid * len(valid) / ranks).clip(upper=1.0)
        adjusted = adjusted.sort_index(ascending=False).cummin().sort_index()
        out.loc[adjusted.index, "fdr_q_value"] = adjusted.to_numpy(dtype=float)
    return out


def _classification(
    *,
    abs_effect: float,
    hac_p_value: float,
    fdr_q_value: float,
    overlap_count: float,
    effective_sample_size: float,
    min_obs: float,
    sign_stability: float,
) -> str:
    if (
        not np.isfinite(overlap_count)
        or overlap_count < min_obs
        or not np.isfinite(effective_sample_size)
        or effective_sample_size < min_obs * 0.75
    ):
        return "noise"
    if not np.isfinite(hac_p_value) or not np.isfinite(fdr_q_value):
        return "noise"
    if (
        hac_p_value <= 0.05
        and fdr_q_value <= 0.10
        and sign_stability >= 0.75
        and abs_effect >= 0.20
    ):
        return "tradable_research_candidate"
    if hac_p_value <= 0.05 and fdr_q_value <= 0.10 and sign_stability >= 0.60:
        return "stable"
    if hac_p_value <= 0.10 or fdr_q_value <= 0.20:
        return "weak"
    return "noise"


def _context_rows(
    frame: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp": aligned_timestamp_for_instrument(
                frame["timestamp"],
                instrument=instrument,
                timeframe=timeframe,
            ),
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for column in ("session_name", "regime", "regime_context_caution"):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    return out.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")


def _iter_context_slices(context: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    slices: list[tuple[str, str, pd.DataFrame]] = [("global", "all_rows", context)]
    if "session_name" in context.columns:
        for session_name, subset in context.groupby("session_name", dropna=True):
            if subset.empty:
                continue
            slices.append(("session", str(session_name), subset))
    if "regime" in context.columns:
        for regime, subset in context.groupby("regime", dropna=True):
            if subset.empty:
                continue
            slices.append(("regime", str(regime), subset))
    if "regime" in context.columns and "regime_context_caution" in context.columns:
        stable = context[
            pd.to_numeric(context["regime_context_caution"], errors="coerce").fillna(1)
            == 0
        ]
        if not stable.empty:
            for regime, subset in stable.groupby("regime", dropna=True):
                if subset.empty:
                    continue
                slices.append(("regime_stable", str(regime), subset))
    return slices


def _matrix_diagnostics(
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if stats_df.empty:
        return pd.DataFrame()
    diag_base = stats_df[
        (stats_df["matrix_family"] == "contemporaneous")
        & (stats_df["metric"] == "pearson_correlation")
    ].copy()
    if diag_base.empty:
        return pd.DataFrame()
    group_cols = [
        "timestamp",
        "timeframe",
        "context_type",
        "context_value",
        "window",
        "return_mode",
    ]
    for keys, group in diag_base.groupby(group_cols, dropna=False):
        matrix = group.pivot(index="left", columns="right", values="value")
        labels = sorted(set(matrix.index).union(matrix.columns))
        if not labels:
            continue
        square = pd.DataFrame(
            np.eye(len(labels), dtype=float), index=labels, columns=labels
        )
        for _, row in group.iterrows():
            left = str(row["left"])
            right = str(row["right"])
            value = _safe_float(row["value"])
            square.loc[left, right] = value
            square.loc[right, left] = value
        eigvals = np.linalg.eigvalsh(square.fillna(0.0).to_numpy(dtype=float))
        eigvals = np.sort(np.real(eigvals))[::-1]
        positive = eigvals[eigvals > 1e-12]
        total = float(np.sum(np.abs(eigvals)))
        first_pc = (
            float(positive[0] / positive.sum()) if len(positive) > 0 else float("nan")
        )
        if len(positive) > 0:
            probs = positive / positive.sum()
            entropy = float(-(probs * np.log(probs)).sum())
            effective_rank = float(np.exp(entropy))
            condition_number = (
                float(positive.max() / positive.min())
                if positive.min() > 0
                else float("inf")
            )
        else:
            entropy = float("nan")
            effective_rank = float("nan")
            condition_number = float("nan")
        rows.extend(
            [
                {
                    "timestamp": keys[0],
                    "timeframe": keys[1],
                    "context_type": keys[2],
                    "context_value": keys[3],
                    "window": keys[4],
                    "return_mode": keys[5],
                    "metric": "first_principal_component_explained_variance",
                    "value": first_pc,
                },
                {
                    "timestamp": keys[0],
                    "timeframe": keys[1],
                    "context_type": keys[2],
                    "context_value": keys[3],
                    "window": keys[4],
                    "return_mode": keys[5],
                    "metric": "effective_rank",
                    "value": effective_rank,
                },
                {
                    "timestamp": keys[0],
                    "timeframe": keys[1],
                    "context_type": keys[2],
                    "context_value": keys[3],
                    "window": keys[4],
                    "return_mode": keys[5],
                    "metric": "matrix_entropy",
                    "value": entropy,
                },
                {
                    "timestamp": keys[0],
                    "timeframe": keys[1],
                    "context_type": keys[2],
                    "context_value": keys[3],
                    "window": keys[4],
                    "return_mode": keys[5],
                    "metric": "condition_number",
                    "value": condition_number,
                },
                {
                    "timestamp": keys[0],
                    "timeframe": keys[1],
                    "context_type": keys[2],
                    "context_value": keys[3],
                    "window": keys[4],
                    "return_mode": keys[5],
                    "metric": "matrix_trace_abs_sum",
                    "value": total,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_cross_asset_correlation_audit(
    frame: pd.DataFrame,
    market_context: pd.DataFrame,
    *,
    instrument: str = "XAU_USD",
    timeframe: str = "H4",
) -> dict[str, pd.DataFrame]:
    if market_context.empty:
        empty = pd.DataFrame()
        return {
            "return_panel": empty,
            "matrix_long": empty,
            "pair_stability_summary": empty,
            "lead_lag_candidate_ranking": empty,
            "significance_acceptance": empty,
            "coverage_audit": empty,
            "matrix_diagnostics": empty,
            "pairwise_snapshot": empty,
            "stability_summary": empty,
            "lag_summary": empty,
            "session_stratification": empty,
            "regime_stratification": empty,
        }

    # Suppress expected numpy RuntimeWarnings from per-session/per-regime
    # correlation groups that have too few observations.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return _build_audit_tables(
            frame, market_context, instrument=instrument, timeframe=timeframe
        )


def _build_audit_tables(
    frame: pd.DataFrame,
    market_context: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> dict[str, pd.DataFrame]:
    context = market_context.copy()
    context["timestamp"] = pd.to_datetime(
        context["timestamp"], utc=True, errors="coerce"
    )
    context = context.sort_values("timestamp").reset_index(drop=True)
    conditioning = _context_rows(frame, instrument=instrument, timeframe=timeframe)
    context = context.merge(conditioning, on="timestamp", how="left")

    symbols = _available_symbols(context)
    if not symbols:
        empty = pd.DataFrame()
        return {
            "return_panel": empty,
            "matrix_long": empty,
            "pair_stability_summary": empty,
            "lead_lag_candidate_ranking": empty,
            "significance_acceptance": empty,
            "coverage_audit": empty,
            "matrix_diagnostics": empty,
            "pairwise_snapshot": empty,
            "stability_summary": empty,
            "lag_summary": empty,
            "session_stratification": empty,
            "regime_stratification": empty,
        }

    return_panel_rows: list[dict[str, object]] = []
    for symbol in symbols:
        for return_mode, prefix in (
            ("raw", "ret_raw_"),
            ("vol_norm", "ret_vol_"),
            ("vol_norm_winsorized", "ret_vol_winsor_"),
        ):
            column = f"{prefix}{symbol}"
            if column not in context.columns:
                continue
            values = pd.to_numeric(context[column], errors="coerce")
            return_panel_rows.extend(
                {
                    "timestamp": ts,
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "return_mode": return_mode,
                    "value": val,
                }
                for ts, val in zip(context["timestamp"], values, strict=False)
            )

    stats_rows: list[dict[str, object]] = []
    pair_stability_rows: list[dict[str, object]] = []
    ranking_seed_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    latest_ts = (
        context["timestamp"].dropna().iloc[-1]
        if context["timestamp"].notna().any()
        else pd.NaT
    )

    for context_type, context_value, subset in _iter_context_slices(context):
        subset = subset.sort_values("timestamp").reset_index(drop=True)
        for return_mode, prefix in (("raw", "ret_raw_"), ("vol_norm", "ret_vol_")):
            available = [
                symbol for symbol in symbols if f"{prefix}{symbol}" in subset.columns
            ]
            if len(available) < 2:
                continue
            for window in HORIZONS_BY_TIMEFRAME[timeframe]:
                min_obs = max(30.0, 1.5 * float(window))
                for column in context.columns:
                    parsed = _parse(_CORR_RE, column)
                    if parsed is None:
                        continue
                    left, right, parsed_window = parsed
                    if parsed_window != window:
                        continue
                    left_col = f"{prefix}{left}"
                    right_col = f"{prefix}{right}"
                    if (
                        left_col not in subset.columns
                        or right_col not in subset.columns
                    ):
                        continue
                    x_series = pd.to_numeric(subset[left_col], errors="coerce")
                    y_series = pd.to_numeric(subset[right_col], errors="coerce")
                    rolling_stats = _rolling_series_stats(
                        x_series, y_series, window=window
                    )
                    lag_metrics: dict[int, dict[str, float]] = {}
                    for lag in LAG_SCAN_LAGS[timeframe]:
                        lag_x, lag_y = _slice_tail_overlap(
                            x_series, y_series, window=window, lag=lag
                        )
                        lag_metrics[lag] = _corr_stats(lag_x, lag_y)
                    best_lag = max(
                        lag_metrics,
                        key=lambda lag: (
                            abs(_safe_float(lag_metrics[lag]["pearson_correlation"]))
                            if np.isfinite(
                                _safe_float(lag_metrics[lag]["pearson_correlation"])
                            )
                            else -np.inf
                        ),
                    )
                    best_lag_stats = lag_metrics[best_lag]
                    lag0_stats = lag_metrics.get(0, {})
                    lag1_stats = lag_metrics.get(1, {})
                    _, best_lag_series = _rolling_best_lag_series(
                        x_series, y_series, window=window, lags=LAG_SCAN_LAGS[timeframe]
                    )
                    valid_best_lags = best_lag_series.dropna()
                    if valid_best_lags.empty:
                        best_lag_persistence = float("nan")
                        lag_turnover = float("nan")
                    else:
                        counts = valid_best_lags.value_counts(normalize=True)
                        best_lag_persistence = float(counts.iloc[0])
                        if len(valid_best_lags) <= 1:
                            lag_turnover = 0.0
                        else:
                            lag_turnover = float(
                                np.mean(
                                    valid_best_lags.to_numpy(dtype=float)[1:]
                                    != valid_best_lags.to_numpy(dtype=float)[:-1]
                                )
                            )
                    valid_pair = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
                    first_half = valid_pair.iloc[: len(valid_pair) // 2]
                    second_half = valid_pair.iloc[len(valid_pair) // 2 :]
                    first_half_corr = _safe_float(first_half["x"].corr(first_half["y"]))
                    second_half_corr = _safe_float(
                        second_half["x"].corr(second_half["y"])
                    )
                    session_sign_consistency = float("nan")
                    regime_sign_consistency = float("nan")
                    if "session_name" in subset.columns:
                        session_corrs = (
                            subset[["session_name", left_col, right_col]]
                            .dropna()
                            .groupby("session_name", dropna=True)
                            .apply(
                                lambda g: _safe_float(
                                    pd.to_numeric(g[left_col], errors="coerce").corr(
                                        pd.to_numeric(g[right_col], errors="coerce")
                                    )
                                )
                            )
                        )
                        if not session_corrs.empty:
                            session_sign_consistency = float(
                                np.mean(
                                    np.sign(session_corrs.to_numpy(dtype=float))
                                    == np.sign(rolling_stats["latest_corr"])
                                )
                            )
                    if "regime" in subset.columns:
                        regime_corrs = (
                            subset[["regime", left_col, right_col]]
                            .dropna()
                            .groupby("regime", dropna=True)
                            .apply(
                                lambda g: _safe_float(
                                    pd.to_numeric(g[left_col], errors="coerce").corr(
                                        pd.to_numeric(g[right_col], errors="coerce")
                                    )
                                )
                            )
                        )
                        if not regime_corrs.empty:
                            regime_sign_consistency = float(
                                np.mean(
                                    np.sign(regime_corrs.to_numpy(dtype=float))
                                    == np.sign(rolling_stats["latest_corr"])
                                )
                            )
                    family_payloads = {
                        ("contemporaneous", 0): lag0_stats,
                        ("lag_1", 1): lag1_stats,
                        ("best_lag", int(best_lag)): best_lag_stats,
                    }
                    for lag, lag_stats in lag_metrics.items():
                        family_payloads[("lag_k", int(lag))] = lag_stats
                    family_payloads[("stability_zscore", 0)] = {
                        "pearson_correlation": rolling_stats["latest_corr"],
                        "spearman_correlation": float("nan"),
                        "fisher_z": _fisher_z(rolling_stats["latest_corr"]),
                        "r_squared": (
                            rolling_stats["latest_corr"] ** 2
                            if np.isfinite(rolling_stats["latest_corr"])
                            else float("nan")
                        ),
                        "classic_t_stat": lag0_stats.get(
                            "classic_t_stat", float("nan")
                        ),
                        "classic_p_value": lag0_stats.get(
                            "classic_p_value", float("nan")
                        ),
                        "hac_t_stat": lag0_stats.get("hac_t_stat", float("nan")),
                        "hac_p_value": lag0_stats.get("hac_p_value", float("nan")),
                        "overlap_count": lag0_stats.get("overlap_count", float("nan")),
                        "effective_sample_size": lag0_stats.get(
                            "effective_sample_size", float("nan")
                        ),
                        "sign_stability": rolling_stats["sign_stability"],
                        "rolling_correlation_std": rolling_stats[
                            "rolling_correlation_std"
                        ],
                        "stability_zscore": rolling_stats["stability_zscore"],
                        "best_lag_persistence_score": best_lag_persistence,
                    }
                    for (matrix_family, lag), metrics in family_payloads.items():
                        metric_payload = dict(metrics)
                        metric_payload.setdefault(
                            "sign_stability", rolling_stats["sign_stability"]
                        )
                        metric_payload.setdefault(
                            "rolling_correlation_std",
                            rolling_stats["rolling_correlation_std"],
                        )
                        metric_payload.setdefault(
                            "stability_zscore", rolling_stats["stability_zscore"]
                        )
                        metric_payload.setdefault(
                            "best_lag_persistence_score", best_lag_persistence
                        )
                        for metric_name, value in metric_payload.items():
                            stats_rows.append(
                                {
                                    "timestamp": latest_ts,
                                    "timeframe": timeframe,
                                    "context_type": context_type,
                                    "context_value": context_value,
                                    "window": window,
                                    "matrix_family": matrix_family,
                                    "return_mode": return_mode,
                                    "left": left,
                                    "right": right,
                                    "lag": lag,
                                    "metric": metric_name,
                                    "value": value,
                                }
                            )
                    pair_stability_rows.append(
                        {
                            "timestamp": latest_ts,
                            "timeframe": timeframe,
                            "context_type": context_type,
                            "context_value": context_value,
                            "window": window,
                            "return_mode": return_mode,
                            "pair": _pair_id(left, right),
                            "left": left,
                            "right": right,
                            "latest_corr": rolling_stats["latest_corr"],
                            "rolling_correlation_mean": rolling_stats[
                                "rolling_correlation_mean"
                            ],
                            "rolling_correlation_std": rolling_stats[
                                "rolling_correlation_std"
                            ],
                            "stability_zscore": rolling_stats["stability_zscore"],
                            "sign_flip_rate": rolling_stats["sign_flip_rate"],
                            "sign_stability": rolling_stats["sign_stability"],
                            "first_half_corr": first_half_corr,
                            "second_half_corr": second_half_corr,
                            "session_sign_consistency": session_sign_consistency,
                            "regime_sign_consistency": regime_sign_consistency,
                            "best_lag": float(best_lag),
                            "best_lag_effect": best_lag_stats.get(
                                "pearson_correlation", float("nan")
                            ),
                            "best_lag_persistence_score": best_lag_persistence,
                            "lag_turnover": lag_turnover,
                        }
                    )
                    coverage_rows.append(
                        {
                            "timestamp": latest_ts,
                            "timeframe": timeframe,
                            "context_type": context_type,
                            "context_value": context_value,
                            "window": window,
                            "return_mode": return_mode,
                            "pair": _pair_id(left, right),
                            "left": left,
                            "right": right,
                            "overlap_count": lag0_stats.get(
                                "overlap_count", float("nan")
                            ),
                            "effective_sample_size": lag0_stats.get(
                                "effective_sample_size", float("nan")
                            ),
                            "min_observations_required": min_obs,
                        }
                    )
                    ranking_seed_rows.append(
                        {
                            "timestamp": latest_ts,
                            "timeframe": timeframe,
                            "context_type": context_type,
                            "context_value": context_value,
                            "window": window,
                            "return_mode": return_mode,
                            "pair": _pair_id(left, right),
                            "left": left,
                            "right": right,
                            "contemporaneous_effect": lag0_stats.get(
                                "pearson_correlation", float("nan")
                            ),
                            "best_lag_effect": best_lag_stats.get(
                                "pearson_correlation", float("nan")
                            ),
                            "best_lag": float(best_lag),
                            "hac_t_stat": best_lag_stats.get(
                                "hac_t_stat", float("nan")
                            ),
                            "hac_p_value": best_lag_stats.get(
                                "hac_p_value", float("nan")
                            ),
                            "overlap_count": lag0_stats.get(
                                "overlap_count", float("nan")
                            ),
                            "effective_sample_size": lag0_stats.get(
                                "effective_sample_size", float("nan")
                            ),
                            "sign_stability": rolling_stats["sign_stability"],
                            "best_lag_persistence_score": best_lag_persistence,
                        }
                    )

    stats_wide = pd.DataFrame(stats_rows)
    if stats_wide.empty:
        empty = pd.DataFrame()
        return {
            "return_panel": pd.DataFrame(return_panel_rows),
            "matrix_long": empty,
            "pair_stability_summary": empty,
            "lead_lag_candidate_ranking": empty,
            "significance_acceptance": empty,
            "coverage_audit": empty,
            "matrix_diagnostics": empty,
            "pairwise_snapshot": empty,
            "stability_summary": empty,
            "lag_summary": empty,
            "session_stratification": empty,
            "regime_stratification": empty,
        }

    metrics_wide = stats_wide.pivot_table(
        index=[
            "timestamp",
            "timeframe",
            "context_type",
            "context_value",
            "window",
            "matrix_family",
            "return_mode",
            "left",
            "right",
            "lag",
        ],
        columns="metric",
        values="value",
        aggfunc="last",
    ).reset_index()
    metrics_wide.columns.name = None
    metrics_wide = _apply_bh_qvalues(metrics_wide)
    metrics_wide["min_observations_required"] = np.maximum(
        30.0, 1.5 * pd.to_numeric(metrics_wide["window"], errors="coerce").fillna(0.0)
    )
    metrics_wide["acceptance_classification"] = [
        _classification(
            abs_effect=abs(_safe_float(effect)),
            hac_p_value=_safe_float(hac_p),
            fdr_q_value=_safe_float(q_value),
            overlap_count=_safe_float(overlap),
            effective_sample_size=_safe_float(eff_n),
            min_obs=_safe_float(min_obs),
            sign_stability=_safe_float(sign_stability),
        )
        for effect, hac_p, q_value, overlap, eff_n, min_obs, sign_stability in zip(
            metrics_wide.get("pearson_correlation", pd.Series(dtype=float)),
            metrics_wide.get("hac_p_value", pd.Series(dtype=float)),
            metrics_wide.get("fdr_q_value", pd.Series(dtype=float)),
            metrics_wide.get("overlap_count", pd.Series(dtype=float)),
            metrics_wide.get("effective_sample_size", pd.Series(dtype=float)),
            metrics_wide.get("min_observations_required", pd.Series(dtype=float)),
            metrics_wide.get("sign_stability", pd.Series(dtype=float)),
            strict=False,
        )
    ]
    metrics_wide["acceptance_flag"] = (
        metrics_wide["acceptance_classification"].eq("tradable_research_candidate")
        | metrics_wide["acceptance_classification"].eq("stable")
    ).astype(np.int8)

    matrix_value_columns = [
        column
        for column in [
            *PRIMARY_METRICS,
            "min_observations_required",
            "acceptance_flag",
        ]
        if column in metrics_wide.columns
    ]
    matrix_long = (
        metrics_wide.melt(
            id_vars=[
                "timestamp",
                "timeframe",
                "context_type",
                "context_value",
                "window",
                "matrix_family",
                "return_mode",
                "left",
                "right",
                "lag",
            ],
            value_vars=matrix_value_columns,
            var_name="metric",
            value_name="value",
        )
        .sort_values(
            [
                "timestamp",
                "context_type",
                "context_value",
                "window",
                "matrix_family",
                "return_mode",
                "left",
                "right",
                "lag",
                "metric",
            ]
        )
        .reset_index(drop=True)
    )

    acceptance = metrics_wide[
        [
            "timestamp",
            "timeframe",
            "context_type",
            "context_value",
            "window",
            "matrix_family",
            "return_mode",
            "left",
            "right",
            "lag",
            "overlap_count",
            "effective_sample_size",
            "hac_p_value",
            "fdr_q_value",
            "sign_stability",
            "min_observations_required",
            "acceptance_classification",
            "acceptance_flag",
        ]
    ].copy()
    acceptance["pair"] = (
        acceptance["left"].astype(str) + "__" + acceptance["right"].astype(str)
    )

    best_lag_acceptance = acceptance[acceptance["matrix_family"] == "best_lag"].copy()
    contemporaneous_acceptance = acceptance[
        acceptance["matrix_family"] == "contemporaneous"
    ].copy()

    coverage = pd.DataFrame(coverage_rows).merge(
        contemporaneous_acceptance[
            [
                "pair",
                "context_type",
                "context_value",
                "window",
                "return_mode",
                "overlap_count",
                "effective_sample_size",
                "acceptance_classification",
                "acceptance_flag",
            ]
        ],
        on=[
            "pair",
            "context_type",
            "context_value",
            "window",
            "return_mode",
            "overlap_count",
            "effective_sample_size",
        ],
        how="left",
    )

    pair_stability = pd.DataFrame(pair_stability_rows)
    if not pair_stability.empty and not acceptance.empty:
        pair_stability = pair_stability.merge(
            best_lag_acceptance[
                [
                    "pair",
                    "context_type",
                    "context_value",
                    "window",
                    "return_mode",
                    "acceptance_classification",
                    "acceptance_flag",
                ]
            ],
            on=["pair", "context_type", "context_value", "window", "return_mode"],
            how="left",
        )

    ranking = pd.DataFrame(ranking_seed_rows)
    if not ranking.empty:
        global_best = acceptance[
            (acceptance["context_type"] == "global")
            & (acceptance["matrix_family"] == "best_lag")
        ][
            [
                "pair",
                "window",
                "return_mode",
                "hac_p_value",
                "fdr_q_value",
                "acceptance_classification",
                "acceptance_flag",
            ]
        ]
        ranking = ranking.merge(
            global_best,
            on=["pair", "window", "return_mode"],
            how="left",
            suffixes=("", "__global_best"),
        )
        session_robustness = (
            acceptance[
                (acceptance["context_type"] == "session")
                & (acceptance["matrix_family"] == "best_lag")
            ]
            .groupby(["pair", "window", "return_mode"], dropna=False)["acceptance_flag"]
            .mean()
            .rename("session_robustness")
            .reset_index()
        )
        regime_robustness = (
            acceptance[
                (acceptance["context_type"].isin(["regime", "regime_stable"]))
                & (acceptance["matrix_family"] == "best_lag")
            ]
            .groupby(["pair", "window", "return_mode"], dropna=False)["acceptance_flag"]
            .mean()
            .rename("regime_robustness")
            .reset_index()
        )
        ranking = ranking.merge(
            session_robustness,
            on=["pair", "window", "return_mode"],
            how="left",
        ).merge(
            regime_robustness,
            on=["pair", "window", "return_mode"],
            how="left",
        )
        ranking["candidate_classification"] = [
            _classification(
                abs_effect=abs(_safe_float(best_effect)),
                hac_p_value=_safe_float(hac_p),
                fdr_q_value=_safe_float(q_value),
                overlap_count=_safe_float(overlap),
                effective_sample_size=_safe_float(eff_n),
                min_obs=max(30.0, 1.5 * float(window)),
                sign_stability=_safe_float(sign_stability),
            )
            for best_effect, hac_p, q_value, overlap, eff_n, window, sign_stability in zip(
                ranking["best_lag_effect"],
                ranking["hac_p_value__global_best"],
                ranking["fdr_q_value"],
                ranking["overlap_count"],
                ranking["effective_sample_size"],
                ranking["window"],
                ranking["sign_stability"],
                strict=False,
            )
        ]
        ranking = ranking.sort_values(
            ["candidate_classification", "window", "best_lag_effect"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    matrix_diagnostics = _matrix_diagnostics(matrix_long)

    pairwise_snapshot = (
        metrics_wide[
            (metrics_wide["context_type"] == "global")
            & (metrics_wide["matrix_family"] == "contemporaneous")
        ][
            [
                "timestamp",
                "left",
                "right",
                "window",
                "return_mode",
                "pearson_correlation",
                "fdr_q_value",
                "acceptance_classification",
            ]
        ]
        .rename(
            columns={
                "pearson_correlation": "correlation",
                "fdr_q_value": "fdr_q_value",
            }
        )
        .reset_index(drop=True)
    )
    pairwise_snapshot["pair"] = (
        pairwise_snapshot["left"] + "__" + pairwise_snapshot["right"]
    )

    lag_summary = (
        metrics_wide[
            (metrics_wide["context_type"] == "global")
            & (metrics_wide["matrix_family"] == "best_lag")
        ][
            [
                "left",
                "right",
                "window",
                "return_mode",
                "pearson_correlation",
                "lag",
                "best_lag_persistence_score",
                "hac_t_stat",
                "fdr_q_value",
            ]
        ]
        .rename(
            columns={
                "pearson_correlation": "latest_best_corr",
                "lag": "latest_best_lag",
            }
        )
        .reset_index(drop=True)
    )
    lag_summary["pair"] = lag_summary["left"] + "__" + lag_summary["right"]

    session_stratification = pair_stability[
        pair_stability["context_type"] == "session"
    ].copy()
    regime_stratification = pair_stability[
        pair_stability["context_type"].isin(["regime", "regime_stable"])
    ].copy()

    return {
        "return_panel": pd.DataFrame(return_panel_rows),
        "matrix_long": matrix_long,
        "pair_stability_summary": pair_stability,
        "lead_lag_candidate_ranking": ranking,
        "significance_acceptance": acceptance,
        "coverage_audit": coverage,
        "matrix_diagnostics": matrix_diagnostics,
        "pairwise_snapshot": pairwise_snapshot,
        "stability_summary": pair_stability,
        "lag_summary": lag_summary,
        "session_stratification": session_stratification,
        "regime_stratification": regime_stratification,
    }


def summarize_cross_asset_correlation_audit(
    audit_tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    matrix_long = audit_tables.get("matrix_long", pd.DataFrame())
    ranking = audit_tables.get("lead_lag_candidate_ranking", pd.DataFrame())
    acceptance = audit_tables.get("significance_acceptance", pd.DataFrame())
    diagnostics = audit_tables.get("matrix_diagnostics", pd.DataFrame())
    pairwise_snapshot = audit_tables.get("pairwise_snapshot", pd.DataFrame())
    lag_summary = audit_tables.get("lag_summary", pd.DataFrame())
    if matrix_long.empty:
        return {
            "matrix_rows": 0,
            "pair_count": 0,
            "context_types": [],
            "return_modes": [],
            "matrix_families": [],
            "tradable_candidate_count": 0,
            "accepted_relation_count": 0,
            "diagnostic_metric_count": 0,
            "pairs": [],
            "lag_pairs": [],
        }
    pair_series = (
        pairwise_snapshot.get("pair", pd.Series(dtype=object))
        if not pairwise_snapshot.empty
        else matrix_long["left"].astype(str) + "__" + matrix_long["right"].astype(str)
    )
    ranking_classes = ranking.get("candidate_classification", pd.Series(dtype=object))
    acceptance_classes = acceptance.get(
        "acceptance_classification", pd.Series(dtype=object)
    )
    return {
        "matrix_rows": int(len(matrix_long)),
        "pair_count": int(
            matrix_long[["left", "right"]].drop_duplicates().shape[0]
            if {"left", "right"}.issubset(matrix_long.columns)
            else 0
        ),
        "context_types": sorted(
            matrix_long.get("context_type", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "return_modes": sorted(
            matrix_long.get("return_mode", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "matrix_families": sorted(
            matrix_long.get("matrix_family", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "tradable_candidate_count": int(
            (ranking_classes == "tradable_research_candidate").sum()
        ),
        "accepted_relation_count": int(
            acceptance_classes.isin(["stable", "tradable_research_candidate"]).sum()
        ),
        "diagnostic_metric_count": int(len(diagnostics)),
        "pairs": sorted(pair_series.dropna().astype(str).unique().tolist()),
        "lag_pairs": sorted(
            lag_summary.get("pair", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }
