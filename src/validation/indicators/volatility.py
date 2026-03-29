"""Validation helpers for the volatility feature layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.validation.common.chart_core import save_figure_html

REQUIRED_LIVE_COLUMNS = {
    # ATR family
    "atr_14",
    "atr_pct_rank_100",
    "atr_zscore_100",
    "atr_to_close",
    "atr_slope_5",
    "atr_slope_20",
    "atr_expanding_flag",
    "atr_contracting_flag",
    # BB width family
    "bb_width",
    "bb_width_pct_rank_100",
    "bb_width_zscore_100",
    "bb_width_below_30",
    "bb_width_below_40",
    "bb_width_above_70",
    "bb_width_expanding_flag",
    "bb_width_contracting_flag",
    # Realized vol
    "rv_close_10",
    "rv_close_20",
    "rv_close_50",
    "rv_close_100",
    "rv_close_zscore_100",
    "rv_close_pct_rank_100",
    # Range estimators
    "parkinson_vol_20",
    "gk_vol_20",
    "rs_vol_20",
    # Spread
    "atr_vs_rv20_ratio",
    "rv20_vs_bbwidth_ratio",
    "parkinson_vs_rv20_ratio",
    # Pullback
    "atr_ratio_rolling",
    "rv_ratio_rolling",
    "bb_width_ratio_rolling",
    # Candle spread
    "body_ratio",
    "body_ratio_above_60",
    "bar_range_atr",
    "wide_range_bar_flag",
    # GARCH
    "garch_vol_1_1",
    "garch_var_1_1",
    "garch_vol_pct_rank_100",
    "garch_vol_zscore_100",
    "garch_to_atr_ratio",
    "garch_to_rv20_ratio",
    "vol_shock_std_resid",
    "garch_vol_expanding_flag",
    "garch_high_vol_flag",
    # SV
    "sv_vol_filtered",
    "sv_logvar_filtered",
    "sv_to_atr_ratio",
    "sv_to_garch_ratio",
    "sv_vol_pct_rank_100",
}

RESEARCH_ONLY_COLUMNS = {
    "r_garch_refit_vol_1_1",
    "r_garch_t_vol_1_1",
    "r_garch_t_std_resid",
    "r_garch_t_nu_estimated",
    "r_sv_vol_smoothed",
    "r_sv_logvar_smoothed",
    "r_sv_filter_smooth_gap",
}

FLAG_COLUMNS = [
    "atr_expanding_flag",
    "atr_contracting_flag",
    "bb_width_below_30",
    "bb_width_below_40",
    "bb_width_above_70",
    "bb_width_expanding_flag",
    "bb_width_contracting_flag",
    "body_ratio_above_60",
    "wide_range_bar_flag",
    "garch_vol_expanding_flag",
    "garch_high_vol_flag",
]

SUMMARY_COLUMNS = [
    "atr_14",
    "atr_pct_rank_100",
    "bb_width",
    "rv_close_20",
    "parkinson_vol_20",
    "garch_vol_1_1",
    "vol_shock_std_resid",
]

_OVERLAP_FLAGS = [
    "atr_expanding_flag",
    "atr_contracting_flag",
    "bb_width_expanding_flag",
    "bb_width_contracting_flag",
    "garch_vol_expanding_flag",
    "garch_high_vol_flag",
]

_BRIDGE_RATIO_COLS = [
    "atr_to_close",
    "garch_to_atr_ratio",
    "garch_to_rv20_ratio",
    "sv_to_atr_ratio",
    "sv_to_garch_ratio",
    "atr_vs_rv20_ratio",
    "rv20_vs_bbwidth_ratio",
    "parkinson_vs_rv20_ratio",
]


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------


def _continuous_stats(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _no_inf_ok(df: pd.DataFrame) -> bool:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return True
    values = numeric.to_numpy(dtype=float)
    finite_mask = ~np.isnan(values)
    return bool(np.isfinite(values[finite_mask]).all())


def _first_valid_row(series: pd.Series) -> int | None:
    valid = pd.to_numeric(series, errors="coerce").notna().to_numpy()
    positions = np.flatnonzero(valid)
    return int(positions[0]) if len(positions) else None


def _flag_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        col: {
            "zeros": int(pd.to_numeric(df[col], errors="coerce").fillna(0).eq(0).sum()),
            "ones": int(pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1).sum()),
        }
        for col in FLAG_COLUMNS
        if col in df.columns
    }


def _correlations(df: pd.DataFrame, include_sv: bool) -> dict[str, float | None]:
    pairs = [
        ("atr_14", "rv_close_20"),
        ("atr_14", "garch_vol_1_1"),
        ("garch_vol_1_1", "rv_close_20"),
    ]
    if include_sv:
        pairs.append(("sv_vol_filtered", "garch_vol_1_1"))

    result: dict[str, float | None] = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        if a in df.columns and b in df.columns:
            s_a = pd.to_numeric(df[a], errors="coerce")
            s_b = pd.to_numeric(df[b], errors="coerce")
            corr = s_a.corr(s_b)
            result[key] = float(corr) if pd.notna(corr) else None
        else:
            result[key] = None
    return result


# ---------------------------------------------------------------------------
# New diagnostic helpers
# ---------------------------------------------------------------------------


def _garch_readiness_audit(df: pd.DataFrame) -> dict:
    """Audit GARCH initialization, warmup completion, and live-safe readiness."""
    col = "garch_vol_1_1"
    if col not in df.columns:
        return {"available": False}

    s = pd.to_numeric(df[col], errors="coerce")
    warmup_rows = int(s.isna().sum())
    first_valid = _first_valid_row(s)
    coverage = float(s.notna().mean())

    result: dict = {
        "warmup_rows_nan": warmup_rows,
        "first_live_safe_row": first_valid,
        "total_rows": len(df),
        "coverage_pct": round(100.0 * coverage, 2),
        "estimation_window": "first_500_valid_returns",
        "initialization_method": "unconditional_variance_from_warmup_window",
        "recursion_scope": "all_returns_post_estimation",
    }

    # Flag if first valid row is suspiciously early (< 200 bars)
    result["readiness_ok"] = (first_valid is None) or (first_valid >= 200)
    if not result["readiness_ok"]:
        result["readiness_warning"] = (
            f"GARCH valid at row {first_valid} — expected >= 200 (warmup=500). "
            "Check that warmup rows are properly masked to NaN."
        )

    # Sensitivity proxy: compare early vs late vol levels
    if first_valid is not None and len(s.dropna()) >= 200:
        valid = s.dropna()
        early_mean = float(valid.iloc[:100].mean())
        later_mean = float(valid.iloc[100:300].mean()) if len(valid) >= 300 else None
        result["early_100_mean_vol"] = round(early_mean, 8)
        result["rows_101_to_300_mean_vol"] = (
            round(later_mean, 8) if later_mean else None
        )

    return result


def _residual_diagnostics(df: pd.DataFrame) -> dict:
    """Full distribution and serial-dependence diagnostics for GARCH residuals."""
    col = "vol_shock_std_resid"
    if col not in df.columns:
        return {"available": False}

    r = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(r) < 10:
        return {"available": False, "count": len(r)}

    abs_r = r.abs()
    r_arr = r.to_numpy()
    sq_arr = r_arr**2

    # Tail exceedances
    tail: dict = {}
    for thresh in (2, 3, 4, 5, 8):
        n_exc = int((abs_r > thresh).sum())
        tail[f"gt_{thresh}sigma"] = {
            "count": n_exc,
            "pct": round(100.0 * n_exc / len(r), 3),
        }

    # Distribution stats
    quants = {f"p{p}": round(float(r.quantile(p / 100)), 4) for p in (1, 5, 50, 95, 99)}

    # Autocorrelation of residuals and squared residuals
    def _safe_autocorr(arr: np.ndarray, lag: int = 1) -> float | None:
        if len(arr) <= lag + 1:
            return None
        s = pd.Series(arr)
        v = s.autocorr(lag=lag)
        return round(float(v), 4) if pd.notna(v) else None

    ac1_resid = _safe_autocorr(r_arr, lag=1)
    ac1_sq = _safe_autocorr(sq_arr, lag=1)
    ac5_sq = _safe_autocorr(sq_arr, lag=5)

    # Clustering: are extreme events consecutive?
    extreme_mask = (abs_r > 3).to_numpy()
    extreme_ac1 = _safe_autocorr(extreme_mask.astype(float), lag=1)

    return {
        "count": len(r),
        "mean": round(float(r.mean()), 6),
        "std": round(float(r.std(ddof=0)), 6),
        "skewness": round(float(r.skew()), 4),
        "excess_kurtosis": round(float(r.kurtosis()), 4),
        "quantiles": quants,
        "tail_exceedances": tail,
        "serial_dependence": {
            "autocorr_lag1_resid": ac1_resid,
            "autocorr_lag1_sq_resid": ac1_sq,
            "autocorr_lag5_sq_resid": ac5_sq,
            "clustering_extreme_gt3sigma_ac1": extreme_ac1,
            "vol_clustering_present": (ac1_sq is not None and abs(ac1_sq) > 0.05),
        },
    }


def _sv_audit(df: pd.DataFrame) -> dict:
    """Filtered SV usefulness audit: smoothness, shock response, disagreement with GARCH."""
    sv_col = "sv_vol_filtered"
    garch_col = "garch_vol_1_1"

    if sv_col not in df.columns:
        return {"available": False}

    sv = pd.to_numeric(df[sv_col], errors="coerce")
    valid = sv.dropna()

    if valid.empty:
        return {"available": False, "count": 0}

    # Summary stats
    summary = {
        "count": len(valid),
        "mean": round(float(valid.mean()), 8),
        "std": round(float(valid.std(ddof=0)), 8),
        "p5": round(float(valid.quantile(0.05)), 8),
        "p50": round(float(valid.median()), 8),
        "p95": round(float(valid.quantile(0.95)), 8),
    }

    # Smoothness
    sv_diff = sv.diff().abs()
    sv_ac1 = round(float(valid.autocorr(lag=1)), 4) if len(valid) > 2 else None
    smoothness = {
        "lag1_autocorr": sv_ac1,
        "mean_abs_first_diff": round(float(sv_diff.mean()), 8),
    }

    vs_garch: dict = {}
    disagreement: dict = {}

    if garch_col in df.columns:
        garch = pd.to_numeric(df[garch_col], errors="coerce")
        g_valid = garch.dropna()

        if not g_valid.empty:
            g_diff = garch.diff().abs()
            g_ac1 = round(float(g_valid.autocorr(lag=1)), 4)
            vs_garch = {
                "garch_lag1_autocorr": g_ac1,
                "garch_mean_abs_first_diff": round(float(g_diff.mean()), 8),
                "sv_smoother_than_garch": float(sv_diff.mean()) < float(g_diff.mean()),
            }

        # High-vol disagreement (p75 threshold)
        both_valid = sv.notna() & garch.notna()
        n_both = int(both_valid.sum())
        if n_both > 0:
            sv_p75 = sv[both_valid].quantile(0.75)
            garch_p75 = garch[both_valid].quantile(0.75)
            sv_hi = (sv > sv_p75) & both_valid
            garch_hi = (garch > garch_p75) & both_valid
            agree = int(((sv_hi == garch_hi) & both_valid).sum())
            disagree = int(((sv_hi != garch_hi) & both_valid).sum())
            disagreement = {
                "n_compared": n_both,
                "agreement_rate": round(agree / n_both, 4),
                "disagreement_rate": round(disagree / n_both, 4),
                "sv_high_garch_not": int(((sv_hi & ~garch_hi) & both_valid).sum()),
                "garch_high_sv_not": int(((~sv_hi & garch_hi) & both_valid).sum()),
                "interpretation": (
                    "SV and GARCH largely agree on high-vol regime"
                    if agree / n_both >= 0.80
                    else "SV and GARCH disagree materially — review which is correct"
                ),
            }

    return {
        "available": True,
        "summary": summary,
        "smoothness": smoothness,
        "vs_garch": vs_garch,
        "disagreement_with_garch": disagreement,
    }


def _flag_overlap_matrix(df: pd.DataFrame) -> dict:
    """Pairwise overlap matrix and dwell statistics for vol-state flags."""
    present = [c for c in _OVERLAP_FLAGS if c in df.columns]
    if not present:
        return {}

    dwell: dict = {}
    for col in present:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        ones_pct = round(100.0 * float(s.mean()), 2)
        # Compute run lengths for state=1
        runs_id = (s != s.shift()).cumsum()
        run_firsts = s.groupby(runs_id).first()
        run_lens = s.groupby(runs_id).transform("count")
        active_lens = run_lens[s == 1]
        dwell[col] = {
            "ones_pct": ones_pct,
            "mean_dwell_bars": (
                round(float(active_lens.mean()), 2) if len(active_lens) > 0 else None
            ),
            "max_dwell_bars": int(active_lens.max()) if len(active_lens) > 0 else None,
        }

    overlap: dict = {}
    n = len(df)
    for i, c1 in enumerate(present):
        for c2 in present[i + 1 :]:
            s1 = pd.to_numeric(df[c1], errors="coerce").fillna(0).astype(bool)
            s2 = pd.to_numeric(df[c2], errors="coerce").fillna(0).astype(bool)
            both = int((s1 & s2).sum())
            either = int((s1 | s2).sum())
            jaccard = round(both / either, 4) if either > 0 else 0.0
            agree = int((s1 == s2).sum())
            overlap[f"{c1}_x_{c2}"] = {
                "both_true": both,
                "both_true_pct": round(100.0 * both / n, 2),
                "jaccard": jaccard,
                "agreement_rate": round(agree / n, 4),
            }

    return {"dwell_stats": dwell, "pairwise_overlap": overlap}


def _bridge_ratio_audit(df: pd.DataFrame) -> dict:
    """Stats and anomaly check for all cross-family bridge ratios."""
    result: dict = {}
    for col in _BRIDGE_RATIO_COLS:
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        inf_count = int(np.isinf(raw.values).sum())
        valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
        neg_count = int((valid < 0).sum())
        if valid.empty:
            result[col] = {"count": 0, "inf_count": inf_count}
            continue
        result[col] = {
            "count": len(valid),
            "mean": round(float(valid.mean()), 6),
            "std": round(float(valid.std(ddof=0)), 6),
            "p1": round(float(valid.quantile(0.01)), 6),
            "p5": round(float(valid.quantile(0.05)), 6),
            "p95": round(float(valid.quantile(0.95)), 6),
            "p99": round(float(valid.quantile(0.99)), 6),
            "inf_count": inf_count,
            "negative_count": neg_count,
        }
    return result


def _warmup_chain(df: pd.DataFrame) -> dict:
    """First-valid-row for each layer in the warmup dependency chain."""
    cols = [
        "atr_14",
        "atr_pct_rank_100",
        "rv_close_20",
        "rv_close_100",
        "bb_width",
        "bb_width_pct_rank_100",
        "garch_vol_1_1",
        "sv_vol_filtered",
    ]
    chain: dict = {}
    for col in cols:
        if col in df.columns:
            chain[col] = _first_valid_row(pd.to_numeric(df[col], errors="coerce"))
    # Add explanation for atr_pct_rank_100
    atr_first = chain.get("atr_14")
    rank_first = chain.get("atr_pct_rank_100")
    chain["atr_pct_rank_100_explanation"] = (
        f"atr_14 first valid at row {atr_first}, "
        f"then 100 rows needed for rolling rank → expected first valid at row "
        f"{(atr_first or 0) + 99}.  Actual: {rank_first}."
    )
    return chain


def _vol_by_regime(df: pd.DataFrame) -> dict:
    """Median vol measures grouped by canonical regime, if regime column present."""
    regime_col = next(
        (c for c in ("regime", "trend_regime", "combined_regime") if c in df.columns),
        None,
    )
    if regime_col is None:
        return {"available": False, "note": "no regime column found in df"}

    vol_cols = [
        "atr_pct_rank_100",
        "bb_width_pct_rank_100",
        "rv_close_20",
        "garch_vol_1_1",
        "sv_vol_filtered",
    ]
    labels = {1: "trending", 0: "ranging", -1: "transitional"}
    result: dict = {}
    for val, label in labels.items():
        mask = df[regime_col] == val
        if not mask.any():
            continue
        sub = df[mask]
        result[label] = {
            col: round(float(pd.to_numeric(sub[col], errors="coerce").median()), 6)
            for col in vol_cols
            if col in df.columns
        }
    return result


def _garch_t_comparison(df: pd.DataFrame) -> dict:
    """Compare Gaussian vs Student-t GARCH residuals."""
    g_col = "vol_shock_std_resid"
    t_col = "r_garch_t_std_resid"
    nu_col = "r_garch_t_nu_estimated"

    if g_col not in df.columns:
        return {"available": False}

    g = pd.to_numeric(df[g_col], errors="coerce").dropna()
    result: dict = {
        "gaussian": {
            "std": round(float(g.std(ddof=0)), 4),
            "excess_kurtosis": round(float(g.kurtosis()), 4),
            "gt_3sigma_pct": round(100.0 * float((g.abs() > 3).mean()), 3),
            "gt_5sigma_pct": round(100.0 * float((g.abs() > 5).mean()), 3),
        }
    }

    if nu_col in df.columns:
        nu_val = pd.to_numeric(df[nu_col], errors="coerce").dropna()
        result["estimated_nu"] = (
            round(float(nu_val.iloc[0]), 2) if not nu_val.empty else None
        )

    if t_col in df.columns:
        t = pd.to_numeric(df[t_col], errors="coerce").dropna()
        if not t.empty:
            result["student_t"] = {
                "std": round(float(t.std(ddof=0)), 4),
                "excess_kurtosis": round(float(t.kurtosis()), 4),
                "gt_3sigma_pct": round(100.0 * float((t.abs() > 3).mean()), 3),
                "gt_5sigma_pct": round(100.0 * float((t.abs() > 5).mean()), 3),
            }
            result["improvement"] = {
                "kurtosis_reduction": round(
                    float(g.kurtosis()) - float(t.kurtosis()), 4
                ),
                "gt3sigma_pct_reduction": round(
                    100.0 * float((g.abs() > 3).mean())
                    - 100.0 * float((t.abs() > 3).mean()),
                    3,
                ),
            }
        else:
            result["student_t"] = {"available": False, "count": 0}
    else:
        result["student_t"] = {
            "available": False,
            "note": "r_garch_t_std_resid not in df",
        }

    return result


def _shock_response_audit(df: pd.DataFrame, *, top_n: int = 50) -> dict:
    """For the top_n largest absolute-return bars, compare how ATR/GARCH/SV respond."""
    if "close" not in df.columns:
        return {"available": False}

    close = pd.to_numeric(df["close"], errors="coerce")
    log_ret = np.log(close / close.shift(1)).abs()
    if log_ret.dropna().empty:
        return {"available": False}

    shock_idx = log_ret.nlargest(top_n).index
    n = len(df)

    vol_series: dict[str, pd.Series] = {}
    for col in ("atr_14", "garch_vol_1_1", "sv_vol_filtered", "rv_close_20"):
        if col in df.columns:
            vol_series[col] = pd.to_numeric(df[col], errors="coerce")

    if not vol_series:
        return {"available": False}

    offsets = (-1, 0, 1, 3, 5)
    results: dict[str, dict] = {}

    for col, series in vol_series.items():
        arr = series.to_numpy(dtype=float)
        idx_arr = df.index
        offset_means: dict[int, float] = {}

        for offset in offsets:
            vals = []
            for si in shock_idx:
                pos = df.index.get_loc(si) if si in df.index else -1
                if pos < 0:
                    continue
                tgt = pos + offset
                if 0 <= tgt < n and np.isfinite(arr[tgt]):
                    vals.append(arr[tgt])
            offset_means[offset] = round(float(np.mean(vals)), 8) if vals else None

        # Compute jump (t vs t-1) and persistence (t+3 vs t)
        t_minus1 = offset_means.get(-1)
        t0 = offset_means.get(0)
        t3 = offset_means.get(3)
        t5 = offset_means.get(5)

        jump = (
            round((t0 - t_minus1) / t_minus1, 4)
            if (t0 and t_minus1 and t_minus1 > 0)
            else None
        )
        persistence = round((t3 - t0) / t0, 4) if (t3 and t0 and t0 > 0) else None

        results[col] = {
            "avg_at_offsets": {str(k): v for k, v in offset_means.items()},
            "jump_t_vs_tminus1": jump,
            "persistence_t3_vs_t0": persistence,
        }

    return {"top_n_shocks": top_n, "series": results}


def _sv_garch_disagreement_profile(df: pd.DataFrame) -> dict:
    """Profile rows where SV and GARCH disagree on high-vol state."""
    sv_col = "sv_vol_filtered"
    garch_col = "garch_vol_1_1"

    if sv_col not in df.columns or garch_col not in df.columns:
        return {"available": False}

    sv = pd.to_numeric(df[sv_col], errors="coerce")
    garch = pd.to_numeric(df[garch_col], errors="coerce")
    both = sv.notna() & garch.notna()

    sv_p75 = sv[both].quantile(0.75)
    garch_p75 = garch[both].quantile(0.75)
    sv_hi = (sv > sv_p75) & both
    garch_hi = (garch > garch_p75) & both
    disagree = (sv_hi != garch_hi) & both

    if not disagree.any():
        return {"available": True, "disagreement_count": 0}

    sub = df[disagree].copy()
    result: dict = {"disagreement_count": int(disagree.sum())}

    # ATR and BB percentile distribution
    for col in ("atr_pct_rank_100", "bb_width_pct_rank_100"):
        if col in sub.columns:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if not s.empty:
                result[f"{col}_in_disagreement"] = {
                    "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4),
                }

    # Return magnitude on disagreement bars
    if "close" in sub.columns:
        c = pd.to_numeric(df["close"], errors="coerce")
        lr = np.log(c / c.shift(1)).abs()
        d_returns = lr[disagree].dropna()
        if not d_returns.empty:
            result["abs_log_return_in_disagreement"] = {
                "mean": round(float(d_returns.mean()), 6),
                "median": round(float(d_returns.median()), 6),
                "p95": round(float(d_returns.quantile(0.95)), 6),
            }

    # Regime distribution if present
    regime_col = next(
        (c for c in ("regime", "trend_regime", "combined_regime") if c in sub.columns),
        None,
    )
    if regime_col:
        result["regime_distribution"] = sub[regime_col].value_counts().to_dict()

    return result


def _tv_comparison_audit(df: pd.DataFrame) -> dict:
    """
    Compare canonical GARCH and RV20 against TradingView benchmark series.

    DOCTRINAL NOTE
    --------------
    The TradingView 'GARCH' proxy is NOT standard return-based GARCH(1,1).
    It is a recursive variance of (close − EMA(close)) with fixed alpha/beta.
    Mismatch is therefore expected and NOT an implementation error.
    Focus on rank-correlation and high-vol episode overlap, not raw equality.
    """
    from src.indicators.foundation.volatility import (
        compute_tv_garch_proxy,
        compute_tv_hist_vol,
    )

    if "close" not in df.columns:
        return {"available": False}

    # Compute TV series
    tv_proxy = compute_tv_garch_proxy(df)
    tv_hv = compute_tv_hist_vol(df)  # annualized %, per=1 (TV default)

    # TV-scaled RV for fair comparison (annualized %)
    rv20_raw = (
        pd.to_numeric(df["rv_close_20"], errors="coerce")
        if "rv_close_20" in df.columns
        else None
    )
    rv20_tv_scaled = (
        rv20_raw * 100.0 * float(np.sqrt(365)) if rv20_raw is not None else None
    )

    garch = (
        pd.to_numeric(df["garch_vol_1_1"], errors="coerce")
        if "garch_vol_1_1" in df.columns
        else None
    )

    def _pct_rank(s: pd.Series, window: int = 100) -> pd.Series:
        return s.rolling(window, min_periods=window).apply(
            lambda w: float(np.count_nonzero(w <= w[-1]) / len(w)), raw=True
        )

    def _compare_pair(a: pd.Series, b: pd.Series, label_a: str, label_b: str) -> dict:
        """Compute comparison metrics between two series."""
        both = a.notna() & b.notna()
        if both.sum() < 10:
            return {"available": False, "n": int(both.sum())}
        av = a[both].to_numpy(dtype=float)
        bv = b[both].to_numpy(dtype=float)

        from scipy.stats import spearmanr

        pearson = float(np.corrcoef(av, bv)[0, 1])
        spearman = float(spearmanr(av, bv).correlation)
        mae = float(np.mean(np.abs(av - bv)))
        rmse = float(np.sqrt(np.mean((av - bv) ** 2)))
        ratio = av / np.where(bv > 0, bv, np.nan)
        mean_ratio = float(np.nanmean(ratio))
        median_ratio = float(np.nanmedian(ratio))

        # Decile overlap
        a_p90 = np.percentile(av, 90)
        b_p90 = np.percentile(bv, 90)
        a_p10 = np.percentile(av, 10)
        b_p10 = np.percentile(bv, 10)
        top_overlap = float(np.mean((av >= a_p90) & (bv >= b_p90)))
        bottom_overlap = float(np.mean((av <= a_p10) & (bv <= b_p10)))

        return {
            "n": int(both.sum()),
            "pearson_corr": round(pearson, 4),
            "spearman_corr": round(spearman, 4),
            "mae": round(mae, 8),
            "rmse": round(rmse, 8),
            "mean_ratio": round(mean_ratio, 4),
            "median_ratio": round(median_ratio, 4),
            "top_decile_overlap": round(top_overlap, 4),
            "bottom_decile_overlap": round(bottom_overlap, 4),
        }

    result: dict = {
        "doctrinal_note": (
            "TV GARCH proxy uses recursive (close - EMA)^2 — NOT return-based GARCH(1,1). "
            "Raw mismatch is expected. Focus on rank correlation and episode overlap."
        )
    }

    # Family A: TV proxy vs canonical GARCH
    if garch is not None:
        result["tv_proxy_vs_garch"] = _compare_pair(
            garch, tv_proxy, "garch_vol_1_1", "tv_garch_proxy"
        )

    # Family B: TV hist vol vs TV-scaled RV20
    if rv20_tv_scaled is not None:
        result["tv_hist_vol_vs_rv20_tv_scaled"] = _compare_pair(
            tv_hv, rv20_tv_scaled, "tv_hist_vol", "rv20_tv_scaled"
        )

    # Percentile-rank comparison
    rank_result: dict = {}
    for name, series in (
        ("garch", garch),
        ("tv_proxy", tv_proxy),
        ("tv_hist_vol", tv_hv),
        ("rv20_tv_scaled", rv20_tv_scaled),
    ):
        if series is not None and series.notna().sum() >= 100:
            rank_result[name] = _pct_rank(series)

    if len(rank_result) >= 2:
        pairs = [
            ("garch", "tv_proxy"),
            ("tv_hist_vol", "rv20_tv_scaled"),
        ]
        result["pct_rank_comparison"] = {}
        for a_key, b_key in pairs:
            if a_key in rank_result and b_key in rank_result:
                result["pct_rank_comparison"][f"{a_key}_vs_{b_key}"] = _compare_pair(
                    rank_result[a_key], rank_result[b_key], a_key, b_key
                )

    # Mismatch interpretation
    tv_vs_garch = result.get("tv_proxy_vs_garch", {})
    spearman = tv_vs_garch.get("spearman_corr")
    top_ol = tv_vs_garch.get("top_decile_overlap")
    if spearman is not None:
        if spearman >= 0.80:
            verdict = "Strong rank agreement — behavioral consistency confirmed."
        elif spearman >= 0.60:
            verdict = "Moderate rank agreement — different objects, similar episodes."
        else:
            verdict = "Weak rank agreement — meaningfully different series; review."
        result["interpretation"] = verdict

    return result


# ---------------------------------------------------------------------------
# Chart builder (unchanged)
# ---------------------------------------------------------------------------


def _build_volatility_figure(df: pd.DataFrame, *, title: str) -> go.Figure:
    """Create a 5-panel plotly validation figure for volatility features."""
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.15, 0.15, 0.20, 0.15],
    )

    x = df.get("timestamp", df.index)

    # Row 1 — Candlestick
    if all(c in df.columns for c in ("open", "high", "low", "close")):
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color="rgba(34, 139, 34, 0.85)",
                decreasing_line_color="rgba(200, 0, 0, 0.85)",
            ),
            row=1,
            col=1,
        )

    # Row 2 — ATR + BB width
    for col, name in (("atr_14", "ATR 14"), ("bb_width", "BB Width")):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], mode="lines", name=name),
                row=2,
                col=1,
            )

    # Row 3 — Realized vol measures
    for col, name in (
        ("rv_close_20", "RV Close 20"),
        ("parkinson_vol_20", "Parkinson 20"),
    ):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], mode="lines", name=name),
                row=3,
                col=1,
            )

    # Row 4 — GARCH + SV filtered
    for col, name in (
        ("garch_vol_1_1", "GARCH Vol 1,1"),
        ("sv_vol_filtered", "SV Filtered"),
    ):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], mode="lines", name=name),
                row=4,
                col=1,
            )

    # Row 5 — Standardized residuals
    if "vol_shock_std_resid" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["vol_shock_std_resid"],
                name="Vol Shock Std Resid",
                marker_color="rgba(100, 100, 200, 0.65)",
            ),
            row=5,
            col=1,
        )
        for level in (2.0, -2.0):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="rgba(200, 0, 0, 0.6)",
                row=5,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1200,
        xaxis_rangeslider_visible=False,
    )
    for ax in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5"):
        fig.update_layout(**{ax: {"rangeslider": {"visible": False}}})

    return fig


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_volatility(
    df: pd.DataFrame,
    *,
    live_df: pd.DataFrame | None = None,
    research_df: pd.DataFrame | None = None,
    outpath: str | Path | None = None,
    title: str = "Volatility Validation",
    include_sv: bool = True,
) -> dict[str, object]:
    """Validate volatility feature columns and produce an HTML chart."""
    audit_df = (
        research_df
        if research_df is not None
        else (live_df if live_df is not None else df)
    )

    # --- Column presence ---
    missing = sorted(REQUIRED_LIVE_COLUMNS - set(audit_df.columns))
    columns_present = len(missing) == 0

    # --- Inf check ---
    no_inf = _no_inf_ok(audit_df)

    # --- Warmup counts ---
    warmup_key_cols = [
        "atr_pct_rank_100",
        "rv_close_20",
        "rv_close_100",
        "garch_vol_1_1",
        "sv_vol_filtered",
    ]
    warmup_counts = {
        col: int(pd.to_numeric(audit_df[col], errors="coerce").isna().sum())
        for col in warmup_key_cols
        if col in audit_df.columns
    }

    first_valid_rows = {
        col: _first_valid_row(audit_df[col])
        for col in warmup_key_cols
        if col in audit_df.columns
    }

    flag_counts = _flag_counts(audit_df)

    stats = {
        col: _continuous_stats(audit_df[col])
        for col in SUMMARY_COLUMNS
        if col in audit_df.columns
    }

    # --- Research column checks ---
    research_columns_present: bool | None = None
    research_contamination_in_live: bool | None = None
    parity_ok: bool | None = None

    if research_df is not None:
        research_columns_present = all(
            c in research_df.columns for c in RESEARCH_ONLY_COLUMNS
        )

    if live_df is not None:
        research_contamination_in_live = any(
            c in live_df.columns for c in RESEARCH_ONLY_COLUMNS
        )

    if live_df is not None and research_df is not None:
        live_live_cols = [c for c in live_df.columns if not c.startswith("r_")]
        research_live_cols = [c for c in research_df.columns if not c.startswith("r_")]
        shared_cols = [c for c in live_live_cols if c in research_live_cols]
        if shared_cols:
            try:
                parity_ok = bool(
                    np.allclose(
                        live_df[shared_cols]
                        .select_dtypes(include=[np.number])
                        .fillna(0)
                        .to_numpy(dtype=float),
                        research_df[shared_cols]
                        .select_dtypes(include=[np.number])
                        .fillna(0)
                        .to_numpy(dtype=float),
                        atol=1e-9,
                        equal_nan=True,
                    )
                )
            except Exception:
                parity_ok = None

    corrs = _correlations(audit_df, include_sv=include_sv)

    summary: dict[str, object] = {
        "columns_present": columns_present,
        "missing_columns": missing,
        "no_inf_ok": no_inf,
        "warmup_counts": warmup_counts,
        "first_valid_rows": first_valid_rows,
        "flag_counts": flag_counts,
        "stats": stats,
        "garch_mode": "fixed_params_recursive",
        "sv_mode": "filtered_only" if include_sv else "disabled",
        "research_columns_present": research_columns_present,
        "research_contamination_in_live": research_contamination_in_live,
        "parity_ok": parity_ok,
        "correlations": corrs,
        # --- New diagnostic sections ---
        "garch_readiness": _garch_readiness_audit(audit_df),
        "residual_diagnostics": _residual_diagnostics(audit_df),
        "sv_audit": _sv_audit(audit_df),
        "flag_overlap": _flag_overlap_matrix(audit_df),
        "bridge_ratios": _bridge_ratio_audit(audit_df),
        "warmup_chain": _warmup_chain(audit_df),
        "vol_by_regime": _vol_by_regime(audit_df),
        "garch_t_comparison": _garch_t_comparison(audit_df),
        "shock_response": _shock_response_audit(audit_df),
        "sv_garch_disagreement_profile": _sv_garch_disagreement_profile(audit_df),
        "tv_comparison": _tv_comparison_audit(audit_df),
    }

    html_path: Path | None = None
    fig = _build_volatility_figure(df, title=title)
    if outpath is not None:
        html_path = save_figure_html(fig, outpath)

    return {"summary": summary, "html_path": html_path}
