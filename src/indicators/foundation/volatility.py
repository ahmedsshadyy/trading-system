"""
Volatility indicators.

Canonical volatility feature layer for live trading and research.

Doctrine
--------
- ATR remains the canonical live-safe baseline volatility metric.
- GARCH(1,1) is the canonical model-based conditional volatility layer.
- Stochastic volatility is research-first (not default live-safe).
- All live-safe outputs at row t depend only on past data through row t.

All functions are pure: input DataFrame is never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.indicators import ta_core as ta

EPS = 1e-12


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rolling_percent_rank(window: np.ndarray) -> float:
    current = window[-1]
    return float(np.count_nonzero(window <= current) / len(window))


def _ols_slope(window: np.ndarray) -> float:
    n = len(window)
    x = np.arange(n, dtype=float)
    x_c = x - x.mean()
    denom = np.dot(x_c, x_c)
    if denom == 0:
        return 0.0
    return float(np.dot(x_c, window) / denom)


def _safe_divide(num: pd.Series, denom: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=num.index, dtype=float)
    valid = denom.notna() & (denom.abs() > EPS)
    result.loc[valid] = num.loc[valid] / denom.loc[valid]
    return result


# ---------------------------------------------------------------------------
# Family A — Realized / range-based volatility
# ---------------------------------------------------------------------------


def add_atr_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR-based volatility features.

    Columns
    -------
    atr_14, atr_pct_rank_100, atr_zscore_100, atr_to_close,
    atr_slope_5, atr_slope_20, atr_expanding_flag, atr_contracting_flag
    """
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    atr = ta.atr(high, low, close, length=period)
    out[f"atr_{period}"] = atr

    out["atr_pct_rank_100"] = atr.rolling(100, min_periods=100).apply(
        _rolling_percent_rank, raw=True
    )

    roll_mean_100 = atr.rolling(100, min_periods=100).mean()
    roll_std_100 = atr.rolling(100, min_periods=100).std(ddof=0)
    out["atr_zscore_100"] = _safe_divide(atr - roll_mean_100, roll_std_100)

    out["atr_to_close"] = _safe_divide(atr, close)

    out["atr_slope_5"] = atr.rolling(5, min_periods=5).apply(_ols_slope, raw=True)
    out["atr_slope_20"] = atr.rolling(20, min_periods=20).apply(_ols_slope, raw=True)

    slope5 = out["atr_slope_5"]
    pct_rank = out["atr_pct_rank_100"]

    expanding = (slope5 > 0) & (pct_rank > 0.6)
    contracting = (slope5 < 0) & (pct_rank < 0.4)
    out["atr_expanding_flag"] = expanding.fillna(False).astype(int)
    out["atr_contracting_flag"] = contracting.fillna(False).astype(int)

    return out


def add_bb_width_features(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Band width volatility features.

    Columns
    -------
    bb_width, bb_width_pct_rank_100, bb_width_zscore_100,
    bb_width_below_30, bb_width_below_40, bb_width_above_70,
    bb_width_expanding_flag, bb_width_contracting_flag
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    bb = ta.bbands(close, length=period, std=std)
    upper = bb[f"BBU_{period}_{std}"]
    lower = bb[f"BBL_{period}_{std}"]
    mid = bb[f"BBM_{period}_{std}"]

    width = _safe_divide(upper - lower, mid)
    out["bb_width"] = width

    out["bb_width_pct_rank_100"] = width.rolling(100, min_periods=100).apply(
        _rolling_percent_rank, raw=True
    )

    roll_mean_100 = width.rolling(100, min_periods=100).mean()
    roll_std_100 = width.rolling(100, min_periods=100).std(ddof=0)
    out["bb_width_zscore_100"] = _safe_divide(width - roll_mean_100, roll_std_100)

    pct_rank = out["bb_width_pct_rank_100"]
    zscore = out["bb_width_zscore_100"]

    out["bb_width_below_30"] = (pct_rank < 0.30).fillna(False).astype(int)
    out["bb_width_below_40"] = (pct_rank < 0.40).fillna(False).astype(int)
    out["bb_width_above_70"] = (pct_rank > 0.70).fillna(False).astype(int)

    expanding = (zscore > 0) & (pct_rank > 0.6)
    contracting = (zscore < 0) & (pct_rank < 0.4)
    out["bb_width_expanding_flag"] = expanding.fillna(False).astype(int)
    out["bb_width_contracting_flag"] = contracting.fillna(False).astype(int)

    return out


def add_realized_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based and range-based realized volatility features.

    Columns
    -------
    rv_close_10, rv_close_20, rv_close_50, rv_close_100,
    rv_close_zscore_100, rv_close_pct_rank_100,
    parkinson_vol_20, gk_vol_20, rs_vol_20,
    atr_vs_rv20_ratio, rv20_vs_bbwidth_ratio, parkinson_vs_rv20_ratio
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")

    # Close-to-close log returns
    log_ret = np.log(close / close.shift(1))
    r2 = log_ret**2

    for n in (10, 20, 50, 100):
        out[f"rv_close_{n}"] = r2.rolling(n, min_periods=n).mean().pipe(np.sqrt)

    rv20 = out["rv_close_20"]
    rv20_mean_100 = rv20.rolling(100, min_periods=100).mean()
    rv20_std_100 = rv20.rolling(100, min_periods=100).std(ddof=0)
    out["rv_close_zscore_100"] = _safe_divide(rv20 - rv20_mean_100, rv20_std_100)
    out["rv_close_pct_rank_100"] = rv20.rolling(100, min_periods=100).apply(
        _rolling_percent_rank, raw=True
    )

    # Safe log ratios for range estimators
    log_hl = np.log(np.maximum(high / np.maximum(low, EPS), EPS))
    log_co = np.log(np.maximum(close / np.maximum(open_, EPS), EPS))
    log_hc = np.log(np.maximum(high / np.maximum(close, EPS), EPS))
    log_ho = np.log(np.maximum(high / np.maximum(open_, EPS), EPS))
    log_lc = np.log(np.maximum(low / np.maximum(close, EPS), EPS))
    log_lo = np.log(np.maximum(low / np.maximum(open_, EPS), EPS))

    # Parkinson estimator
    pk_component = (log_hl**2) / (4.0 * np.log(2))
    out["parkinson_vol_20"] = (
        pk_component.rolling(20, min_periods=20).mean().clip(lower=0).pipe(np.sqrt)
    )

    # Garman-Klass estimator
    gk_component = 0.5 * (log_hl**2) - (2.0 * np.log(2) - 1.0) * (log_co**2)
    out["gk_vol_20"] = (
        gk_component.rolling(20, min_periods=20).mean().clip(lower=0).pipe(np.sqrt)
    )

    # Rogers-Satchell estimator
    rs_component = log_hc * log_ho + log_lc * log_lo
    out["rs_vol_20"] = (
        rs_component.rolling(20, min_periods=20).mean().clip(lower=0).pipe(np.sqrt)
    )

    # Volatility spread family
    atr_to_close = out.get("atr_to_close")
    if atr_to_close is None:
        if "atr_14" in out.columns:
            atr_to_close = _safe_divide(out["atr_14"], close)
        else:
            atr_to_close = pd.Series(np.nan, index=out.index, dtype=float)

    rv20_safe = rv20.where(rv20.abs() > EPS, np.nan)
    out["atr_vs_rv20_ratio"] = _safe_divide(atr_to_close, rv20_safe)

    bb_width = out.get("bb_width")
    if bb_width is None:
        out["rv20_vs_bbwidth_ratio"] = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        bb_width_safe = bb_width.where(bb_width.abs() > EPS, np.nan)
        out["rv20_vs_bbwidth_ratio"] = _safe_divide(rv20, bb_width_safe)

    parkinson = out["parkinson_vol_20"]
    out["parkinson_vs_rv20_ratio"] = _safe_divide(parkinson, rv20_safe)

    return out


def add_pullback_vol_features(
    df: pd.DataFrame, impulse_len: int = 5, pullback_len: int = 5
) -> pd.DataFrame:
    """Add rolling pullback / impulse volatility ratios and candle spread features.

    Columns
    -------
    atr_ratio_rolling, rv_ratio_rolling, bb_width_ratio_rolling,
    body_ratio, body_ratio_above_60, bar_range_atr, wide_range_bar_flag
    """
    out = df.copy()

    # Ensure prerequisite columns
    if "atr_14" not in out.columns:
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        close = pd.to_numeric(out["close"], errors="coerce")
        out["atr_14"] = ta.atr(high, low, close, length=14)

    if "rv_close_20" not in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        log_ret = np.log(close / close.shift(1))
        out["rv_close_20"] = (
            (log_ret**2).rolling(20, min_periods=20).mean().pipe(np.sqrt)
        )

    if "bb_width" not in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        bb = ta.bbands(close, length=20, std=2.0)
        upper = bb["BBU_20_2.0"]
        lower = bb["BBL_20_2.0"]
        mid = bb["BBM_20_2.0"]
        out["bb_width"] = _safe_divide(upper - lower, mid)

    total = impulse_len + pullback_len

    def _rolling_ratio(series: pd.Series) -> list[float]:
        vals = series.to_numpy(dtype=float)
        n = len(vals)
        ratios: list[float] = []
        for i in range(n):
            if i < total - 1:
                ratios.append(np.nan)
                continue
            impulse_slice = vals[i - total + 1 : i - pullback_len + 1]
            pullback_slice = vals[i - pullback_len + 1 : i + 1]
            valid_imp = impulse_slice[~np.isnan(impulse_slice)]
            valid_pb = pullback_slice[~np.isnan(pullback_slice)]
            if len(valid_imp) == 0 or len(valid_pb) == 0:
                ratios.append(np.nan)
                continue
            impulse_mean = float(valid_imp.mean())
            pullback_mean = float(valid_pb.mean())
            if np.isnan(impulse_mean) or abs(impulse_mean) < EPS:
                ratios.append(np.nan)
            else:
                ratios.append(pullback_mean / impulse_mean)
        return ratios

    out["atr_ratio_rolling"] = _rolling_ratio(out["atr_14"])
    out["rv_ratio_rolling"] = _rolling_ratio(out["rv_close_20"])
    out["bb_width_ratio_rolling"] = _rolling_ratio(out["bb_width"])

    # Candle spread / breakout features
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    body = (close - open_).abs()
    rng = high - low
    out["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
    out["body_ratio_above_60"] = (out["body_ratio"] > 0.6).astype(int)

    atr14 = out["atr_14"]
    out["bar_range_atr"] = _safe_divide(rng, atr14)
    out["wide_range_bar_flag"] = (out["bar_range_atr"] > 1.5).fillna(False).astype(int)

    return out


# ---------------------------------------------------------------------------
# Per-signal exact helpers (not rolling — called by scanner)
# ---------------------------------------------------------------------------


def compute_atr_ratio(
    df: pd.DataFrame,
    impulse_start: int,
    impulse_end: int,
    pullback_start: int,
    pullback_end: int,
    atr_col: str = "atr_14",
) -> float:
    """Compute pullback ATR / impulse ATR ratio for a specific range.

    This is a *per-signal* helper — the scanner calls it when evaluating
    a specific setup.  It is NOT applied as a rolling column.

    Returns
    -------
    float
        Ratio < 0.8 -> quality pullback.  > 0.8 -> too volatile.
    """
    impulse_atr = df[atr_col].iloc[impulse_start:impulse_end].mean()
    pullback_atr = df[atr_col].iloc[pullback_start:pullback_end].mean()
    if impulse_atr == 0 or np.isnan(impulse_atr):
        return np.nan
    return float(pullback_atr / impulse_atr)


def compute_rv_ratio(
    df: pd.DataFrame,
    impulse_start: int,
    impulse_end: int,
    pullback_start: int,
    pullback_end: int,
) -> float:
    """Compute pullback realized vol / impulse realized vol ratio."""
    col = "rv_close_20"
    impulse_rv = df[col].iloc[impulse_start:impulse_end].mean()
    pullback_rv = df[col].iloc[pullback_start:pullback_end].mean()
    if np.isnan(impulse_rv) or abs(impulse_rv) < EPS:
        return np.nan
    return float(pullback_rv / impulse_rv)


def compute_bbwidth_ratio(
    df: pd.DataFrame,
    impulse_start: int,
    impulse_end: int,
    pullback_start: int,
    pullback_end: int,
) -> float:
    """Compute pullback BB width / impulse BB width ratio."""
    col = "bb_width"
    impulse_bb = df[col].iloc[impulse_start:impulse_end].mean()
    pullback_bb = df[col].iloc[pullback_start:pullback_end].mean()
    if np.isnan(impulse_bb) or abs(impulse_bb) < EPS:
        return np.nan
    return float(pullback_bb / impulse_bb)


# ---------------------------------------------------------------------------
# Family B — GARCH(1,1)
# ---------------------------------------------------------------------------


def _garch_nll(params: np.ndarray, returns: np.ndarray, sigma2_0: float) -> float:
    """Negative log-likelihood for GARCH(1,1) with Gaussian innovations."""
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = sigma2_0
    nll = 0.0
    for t in range(n):
        if t > 0:
            sigma2 = omega + alpha * returns[t - 1] ** 2 + beta * sigma2
        sigma2 = max(sigma2, EPS)
        nll += np.log(sigma2) + returns[t] ** 2 / sigma2
    return nll


def _garch_recurse(
    returns: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: float
) -> np.ndarray:
    """Recurse GARCH(1,1) variance forward over all returns."""
    n = len(returns)
    sigma2_arr = np.empty(n, dtype=float)
    sigma2 = sigma2_0
    for t in range(n):
        if t > 0:
            sigma2 = omega + alpha * returns[t - 1] ** 2 + beta * sigma2
        sigma2 = max(sigma2, EPS)
        sigma2_arr[t] = sigma2
    return sigma2_arr


def add_garch_features(
    df: pd.DataFrame,
    *,
    warmup: int = 500,
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add GARCH(1,1) conditional volatility features.

    Parameters estimated once on warmup rows, then recursed forward with
    FIXED params — no future data leakage.

    Columns (live-safe)
    -------------------
    garch_var_1_1, garch_vol_1_1, garch_vol_pct_rank_100, garch_vol_zscore_100,
    garch_to_atr_ratio, garch_to_rv20_ratio, vol_shock_std_resid,
    garch_vol_expanding_flag, garch_high_vol_flag

    Columns (research-only, prefix r_)
    ------------------------------------
    r_garch_refit_vol_1_1, r_garch_t_vol_1_1
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    log_ret_full = np.log(close / close.shift(1))
    valid_mask = log_ret_full.notna().to_numpy()
    first_valid_idx = int(np.argmax(valid_mask)) if valid_mask.any() else len(out)

    returns_full = log_ret_full.to_numpy()

    # Slice valid returns for estimation
    valid_returns = returns_full[valid_mask]
    n_valid = len(valid_returns)

    # Initialize all GARCH columns to NaN
    garch_var = np.full(len(out), np.nan)
    garch_vol = np.full(len(out), np.nan)

    if n_valid >= warmup:
        warmup_returns = valid_returns[:warmup]
        sigma2_0 = float(np.var(warmup_returns))

        # Fit GARCH(1,1) on warmup data
        x0 = np.array([sigma2_0 * (1.0 - 0.95), 0.10, 0.85])
        bounds = [(1e-8, 1.0), (1e-8, 1.0 - 1e-8), (1e-8, 1.0 - 1e-8)]
        constraints = [{"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}]

        try:
            result = minimize(
                _garch_nll,
                x0,
                args=(warmup_returns, sigma2_0),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if result.success and np.all(np.isfinite(result.x)):
                omega, alpha, beta = result.x
            else:
                raise RuntimeError("GARCH optimization did not converge")
        except Exception:
            unconditional_var = float(np.var(warmup_returns))
            alpha = 0.10
            beta = 0.85
            omega = unconditional_var * (1.0 - 0.95)

        # Unconditional variance for initialization
        ab_sum = alpha + beta
        if ab_sum < 1.0:
            sigma2_init = omega / (1.0 - ab_sum)
        else:
            sigma2_init = float(np.var(warmup_returns))

        # Recurse forward over all valid returns
        sigma2_valid = _garch_recurse(valid_returns, omega, alpha, beta, sigma2_init)

        # Map back to full index.
        # Rows within the estimation warmup window remain NaN — they are not
        # live-safe because the parameters that produced them required seeing
        # all warmup bars first.
        valid_positions = np.where(valid_mask)[0]
        for pos_idx, orig_idx in enumerate(valid_positions):
            if pos_idx >= warmup:
                garch_var[orig_idx] = sigma2_valid[pos_idx]
                garch_vol[orig_idx] = np.sqrt(sigma2_valid[pos_idx])

    out["garch_var_1_1"] = garch_var
    out["garch_vol_1_1"] = garch_vol

    garch_vol_s = out["garch_vol_1_1"]

    out["garch_vol_pct_rank_100"] = garch_vol_s.rolling(100, min_periods=100).apply(
        _rolling_percent_rank, raw=True
    )

    gv_mean_100 = garch_vol_s.rolling(100, min_periods=100).mean()
    gv_std_100 = garch_vol_s.rolling(100, min_periods=100).std(ddof=0)
    out["garch_vol_zscore_100"] = _safe_divide(garch_vol_s - gv_mean_100, gv_std_100)

    # Ratio to atr_to_close
    if "atr_to_close" in out.columns:
        atr_to_close = pd.to_numeric(out["atr_to_close"], errors="coerce")
    else:
        close_ = pd.to_numeric(out["close"], errors="coerce")
        atr14 = (
            pd.to_numeric(out["atr_14"], errors="coerce")
            if "atr_14" in out.columns
            else ta.atr(
                pd.to_numeric(out["high"], errors="coerce"),
                pd.to_numeric(out["low"], errors="coerce"),
                close_,
                length=14,
            )
        )
        atr_to_close = _safe_divide(atr14, close_)

    out["garch_to_atr_ratio"] = _safe_divide(garch_vol_s, atr_to_close)

    # Ratio to rv_close_20
    if "rv_close_20" in out.columns:
        rv20 = pd.to_numeric(out["rv_close_20"], errors="coerce")
    else:
        close_ = pd.to_numeric(out["close"], errors="coerce")
        log_r = np.log(close_ / close_.shift(1))
        rv20 = (log_r**2).rolling(20, min_periods=20).mean().pipe(np.sqrt)
    rv20_safe = rv20.where(rv20.abs() > EPS, np.nan)
    out["garch_to_rv20_ratio"] = _safe_divide(garch_vol_s, rv20_safe)

    # Standardized residuals
    garch_var_s = out["garch_var_1_1"]
    valid_var = garch_var_s > 0
    std_resid = pd.Series(np.nan, index=out.index, dtype=float)
    std_resid.loc[valid_var] = log_ret_full.loc[valid_var] / np.sqrt(
        garch_var_s.loc[valid_var]
    )
    out["vol_shock_std_resid"] = std_resid

    gv_mean_20 = garch_vol_s.rolling(20, min_periods=20).mean()
    pct_rank = out["garch_vol_pct_rank_100"]
    expanding_flag = (garch_vol_s > gv_mean_20) & (pct_rank > 0.6)
    high_vol_flag = pct_rank > 0.8
    out["garch_vol_expanding_flag"] = expanding_flag.fillna(False).astype(int)
    out["garch_high_vol_flag"] = high_vol_flag.fillna(False).astype(int)
    out["vol_shock_gt_3sigma"] = (std_resid.abs() > 3.0).fillna(False).astype(int)
    out["vol_shock_gt_5sigma"] = (std_resid.abs() > 5.0).fillna(False).astype(int)

    if include_research_only:
        # Rolling refit every 100 bars with expanding window
        refit_vol = np.full(len(out), np.nan)
        refit_interval = 100
        cached_omega, cached_alpha, cached_beta = None, None, None
        for i in range(warmup, len(out)):
            if (i - warmup) % refit_interval == 0:
                # Refit on expanding window up to current bar
                fit_returns = valid_returns[: min(i + 1, n_valid)]
                if len(fit_returns) < warmup:
                    continue
                s0 = float(np.var(fit_returns[:warmup]))
                x0_ = np.array([s0 * 0.05, 0.10, 0.85])
                try:
                    res_ = minimize(
                        _garch_nll,
                        x0_,
                        args=(fit_returns, s0),
                        method="SLSQP",
                        bounds=[(1e-8, 1.0), (1e-8, 1.0), (1e-8, 1.0)],
                        constraints=[
                            {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}
                        ],
                        options={"maxiter": 300, "ftol": 1e-8},
                    )
                    if res_.success and np.all(np.isfinite(res_.x)):
                        cached_omega, cached_alpha, cached_beta = res_.x
                except Exception:
                    pass
            if cached_omega is not None and i < len(garch_var):
                refit_vol[i] = (
                    np.sqrt(max(garch_var[i], EPS))
                    if not np.isnan(garch_var[i])
                    else np.nan
                )
        out["r_garch_refit_vol_1_1"] = refit_vol

        # Student-t adjusted GARCH: estimate nu from Gaussian residual kurtosis,
        # then re-scale so t(nu) innovations have unit variance.
        # IMPORTANT: nu estimation uses the full residual series — research-only.
        kurt = float(std_resid.dropna().kurtosis())  # excess kurtosis
        nu = float(np.clip(6.0 / max(kurt, 0.01) + 4.0, 4.1, 100.0))
        # Under t(nu): Var(epsilon) = nu/(nu-2).  Re-scale vol so innovations ~ t(nu) with unit variance.
        t_scale = float(np.sqrt((nu - 2.0) / nu))
        t_vol = garch_vol_s * t_scale
        out["r_garch_t_vol_1_1"] = t_vol
        # t-adjusted standardized residuals
        t_var = t_vol**2
        t_var_safe = t_var.where(t_var > 0, np.nan)
        out["r_garch_t_std_resid"] = log_ret_full / t_vol.where(t_vol > 0, np.nan)
        out["r_garch_t_nu_estimated"] = (
            nu  # scalar broadcast — constant column for reference
        )

    return out


# ---------------------------------------------------------------------------
# Family C — Stochastic Volatility (research-first)
# ---------------------------------------------------------------------------


def add_stochastic_vol_features(
    df: pd.DataFrame,
    *,
    warmup: int = 500,
    include_research_only: bool = False,
    mode: str = "filtered_only",
) -> pd.DataFrame:
    """Add stochastic volatility features via EWMA log-variance proxy.

    Live-safe filtered SV: log_var_t = alpha_sv * log_var_{t-1} + (1 - alpha_sv) * log(r_t^2 + eps)
    This is entirely causal — depends only on past data.

    Columns (live-safe)
    -------------------
    sv_vol_filtered, sv_logvar_filtered, sv_to_atr_ratio, sv_to_garch_ratio, sv_vol_pct_rank_100

    Columns (research-only, prefix r_)
    ------------------------------------
    r_sv_vol_smoothed, r_sv_logvar_smoothed, r_sv_filter_smooth_gap
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    log_ret = np.log(close / close.shift(1))
    log_ret_arr = log_ret.to_numpy(dtype=float)

    alpha_sv = 0.97
    n = len(log_ret_arr)

    log_var_arr = np.full(n, np.nan)
    sv_vol_arr = np.full(n, np.nan)

    # Find first valid return
    first_valid = -1
    for i in range(n):
        if np.isfinite(log_ret_arr[i]):
            first_valid = i
            break

    if first_valid >= 0 and n > first_valid:
        # Warm up: use the empirical log variance from first warmup valid returns
        valid_rets = log_ret_arr[first_valid:]
        warmup_rets = valid_rets[: min(warmup, len(valid_rets))]
        var0 = (
            float(np.var(warmup_rets[np.isfinite(warmup_rets)]))
            if len(warmup_rets) > 0
            else EPS
        )
        log_var_0 = np.log(max(var0, EPS))

        # Forward pass (causal EWMA log-variance)
        log_var_t = log_var_0
        all_log_var = np.full(n, np.nan)
        for i in range(first_valid, n):
            r = log_ret_arr[i]
            if np.isfinite(r):
                innovation = np.log(r**2 + EPS)
                log_var_t = alpha_sv * log_var_t + (1.0 - alpha_sv) * innovation
                all_log_var[i] = log_var_t

        # Output starts after warmup
        warmup_end_idx = first_valid + warmup
        for i in range(n):
            if i >= warmup_end_idx and np.isfinite(all_log_var[i]):
                log_var_arr[i] = all_log_var[i]
                sv_vol_arr[i] = np.exp(all_log_var[i] / 2.0)

    out["sv_vol_filtered"] = sv_vol_arr
    out["sv_logvar_filtered"] = log_var_arr

    sv_vol_s = out["sv_vol_filtered"]

    # Ratio to atr_to_close
    if "atr_to_close" in out.columns:
        atr_to_close = pd.to_numeric(out["atr_to_close"], errors="coerce")
    else:
        close_ = pd.to_numeric(out["close"], errors="coerce")
        atr14 = (
            pd.to_numeric(out["atr_14"], errors="coerce")
            if "atr_14" in out.columns
            else ta.atr(
                pd.to_numeric(out["high"], errors="coerce"),
                pd.to_numeric(out["low"], errors="coerce"),
                close_,
                length=14,
            )
        )
        atr_to_close = _safe_divide(atr14, close_)

    out["sv_to_atr_ratio"] = _safe_divide(sv_vol_s, atr_to_close)

    if "garch_vol_1_1" in out.columns:
        garch_vol = pd.to_numeric(out["garch_vol_1_1"], errors="coerce")
        garch_safe = garch_vol.where(garch_vol.abs() > EPS, np.nan)
        out["sv_to_garch_ratio"] = _safe_divide(sv_vol_s, garch_safe)
    else:
        out["sv_to_garch_ratio"] = pd.Series(np.nan, index=out.index, dtype=float)

    out["sv_vol_pct_rank_100"] = sv_vol_s.rolling(100, min_periods=100).apply(
        _rolling_percent_rank, raw=True
    )

    if include_research_only:
        # Bidirectional smoother: forward + backward pass average (NOT live-safe)
        log_var_all = np.full(n, np.nan)
        if first_valid >= 0:
            warmup_rets_2 = log_ret_arr[
                first_valid : first_valid + min(warmup, n - first_valid)
            ]
            var0_2 = (
                float(np.var(warmup_rets_2[np.isfinite(warmup_rets_2)]))
                if len(warmup_rets_2) > 0
                else EPS
            )
            log_var_0_2 = np.log(max(var0_2, EPS))

            # Forward
            fwd_lv = np.full(n, np.nan)
            lv = log_var_0_2
            for i in range(first_valid, n):
                r = log_ret_arr[i]
                if np.isfinite(r):
                    lv = alpha_sv * lv + (1.0 - alpha_sv) * np.log(r**2 + EPS)
                    fwd_lv[i] = lv

            # Backward
            bwd_lv = np.full(n, np.nan)
            lv_bwd = log_var_0_2
            last_valid = n - 1
            while last_valid >= 0 and not np.isfinite(log_ret_arr[last_valid]):
                last_valid -= 1
            if last_valid >= 0:
                for i in range(last_valid, first_valid - 1, -1):
                    r = log_ret_arr[i]
                    if np.isfinite(r):
                        lv_bwd = alpha_sv * lv_bwd + (1.0 - alpha_sv) * np.log(
                            r**2 + EPS
                        )
                        bwd_lv[i] = lv_bwd

            for i in range(n):
                f = fwd_lv[i]
                b = bwd_lv[i]
                if np.isfinite(f) and np.isfinite(b):
                    log_var_all[i] = 0.5 * (f + b)
                elif np.isfinite(f):
                    log_var_all[i] = f

        out["r_sv_logvar_smoothed"] = log_var_all
        sv_smoothed = np.where(
            np.isfinite(log_var_all), np.exp(log_var_all / 2.0), np.nan
        )
        out["r_sv_vol_smoothed"] = sv_smoothed

        filt = sv_vol_arr
        gap = np.where(
            np.isfinite(sv_smoothed) & np.isfinite(filt),
            sv_smoothed - filt,
            np.nan,
        )
        out["r_sv_filter_smooth_gap"] = gap

    return out


# ---------------------------------------------------------------------------
# TradingView GARCH proxy (comparison / benchmark only — NOT canonical)
# ---------------------------------------------------------------------------


def compute_tv_garch_proxy(
    df: pd.DataFrame,
    alpha: float = 0.10,
    beta: float = 0.80,
    ema_length: int = 20,
) -> pd.Series:
    """Reproduce the TradingView 'GARCH Volatility Estimation - The Quant Science' proxy.

    IMPORTANT — this is NOT standard return-based GARCH(1,1).
    It is a recursive variance of (close − EMA(close, ema_length)) with fixed
    alpha/beta parameters.  Formula:

        variance_t = alpha * (close_t − ema_t)^2 + beta * variance_{t-1}
        tv_garch_proxy_t = sqrt(variance_t)

    The TV script computes auxiliary alpha/beta in a loop but does NOT feed
    them back into the variance recursion — the effective recursion always uses
    the fixed ``alpha`` and ``beta`` inputs.

    Use only for behavioral benchmark comparisons against our canonical
    return-based GARCH.  Not live-safe canonical output.
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    ema = close.ewm(span=ema_length, adjust=False).mean()
    dev_sq = (close - ema) ** 2
    dev_arr = dev_sq.to_numpy(dtype=float)
    n = len(dev_arr)
    var_arr = np.full(n, np.nan)

    # Find first finite deviation
    first = next((i for i in range(n) if np.isfinite(dev_arr[i])), None)
    if first is None:
        return pd.Series(np.nan, index=df.index)

    var_arr[first] = dev_arr[first]
    for i in range(first + 1, n):
        if np.isfinite(dev_arr[i]) and np.isfinite(var_arr[i - 1]):
            var_arr[i] = alpha * dev_arr[i] + beta * var_arr[i - 1]
        elif np.isfinite(dev_arr[i]):
            var_arr[i] = dev_arr[i]

    return pd.Series(np.sqrt(np.maximum(var_arr, 0.0)), index=df.index)


def compute_tv_hist_vol(
    df: pd.DataFrame,
    ema_length: int = 20,
    annual: int = 365,
    per: int = 1,
) -> pd.Series:
    """Reproduce TradingView historical volatility companion series.

    Formula: tv_hist_vol = 100 * stdev(log(close/close[1]), ema_length) * sqrt(annual/per)

    For H4 bars set per=1 (the TV script default).
    Returns values in the same % annualized space as the TV script.

    NOTE: our canonical rv_close_20 is NOT annualized.  Use
    ``compute_tv_hist_vol`` only for direct TV-comparison plots.
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(ema_length, min_periods=ema_length).std(ddof=1)
    return 100.0 * rv * float(np.sqrt(annual / per))


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (used by __init__.py and scanner code)
# ---------------------------------------------------------------------------


# add_atr(df, period=14) -> df with atr_{period} and atr_pct_50 (legacy compat)
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Backward-compatible alias: returns add_atr_features output.

    Legacy callers expecting ``atr_pct_50`` also get that column for continuity.
    """
    out = add_atr_features(df, period=period)
    # Legacy column: atr_pct_50 (percentile rank over 50 bars, 0–100 scale)
    atr = pd.to_numeric(out[f"atr_{period}"], errors="coerce")
    out["atr_pct_50"] = (
        atr.rolling(50, min_periods=50).apply(_rolling_percent_rank, raw=True) * 100.0
    )
    return out


def add_bb_width(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Backward-compatible alias for add_bb_width_features."""
    return add_bb_width_features(df, period=period, std=std)


def add_rolling_atr_ratio(
    df: pd.DataFrame, impulse_len: int = 5, pullback_len: int = 5
) -> pd.DataFrame:
    """Backward-compatible alias — adds only atr_ratio_rolling."""
    out = df.copy()
    if "atr_14" not in out.columns:
        high = pd.to_numeric(out["high"], errors="coerce")
        low = pd.to_numeric(out["low"], errors="coerce")
        close = pd.to_numeric(out["close"], errors="coerce")
        out["atr_14"] = ta.atr(high, low, close, length=14)

    atr = out["atr_14"]
    total = impulse_len + pullback_len
    ratios: list[float] = []
    for i in range(len(out)):
        if i < total - 1:
            ratios.append(np.nan)
            continue
        impulse_mean = atr.iloc[i - total + 1 : i - pullback_len + 1].mean()
        pullback_mean = atr.iloc[i - pullback_len + 1 : i + 1].mean()
        if impulse_mean == 0 or np.isnan(impulse_mean):
            ratios.append(np.nan)
        else:
            ratios.append(float(pullback_mean / impulse_mean))
    out["atr_ratio_rolling"] = ratios
    return out


def add_body_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias — adds body_ratio and body_ratio_above_60."""
    out = df.copy()
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    body = (close - open_).abs()
    rng = high - low
    out["body_ratio"] = np.where(rng > 0, body / rng, 0.0)
    out["body_ratio_above_60"] = (out["body_ratio"] > 0.6).astype(int)
    return out


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def add_volatility_features(
    df: pd.DataFrame,
    *,
    include_research_only: bool = False,
) -> pd.DataFrame:
    """Add all volatility features in canonical order.

    Builds the full volatility feature set:
    Family A (realized/range): ATR, BB width, realized vol, pullback ratios.
    Family B (model-based): GARCH(1,1) conditional variance.
    Family C (research-first): Stochastic volatility filtered proxy.

    Parameters
    ----------
    df:
        Schema-normalized OHLCV DataFrame.
    include_research_only:
        When True, also adds ``r_`` prefixed research-only columns that
        may use future data or are otherwise NOT safe for live deployment.
    """
    out = df.copy()
    out = add_atr_features(out)
    out = add_bb_width_features(out)
    out = add_realized_vol_features(out)
    out = add_pullback_vol_features(out)
    out = add_garch_features(out, include_research_only=include_research_only)
    out = add_stochastic_vol_features(out, include_research_only=include_research_only)
    return out
