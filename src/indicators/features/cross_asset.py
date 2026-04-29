from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.indicators._helpers.schema import normalize_candle_schema
from src.indicators.smt import add_smt_divergence
from src.pipeline_runtime import (
    ArtifactWriteResult,
    canonical_dataset_root,
    fingerprint_mapping,
    merge_recomputed_frontier,
    persist_partitioned_dataset,
    write_json_atomic,
)

SUPPORTED_CROSS_ASSET_TIMEFRAMES = frozenset({"H1", "H4"})
FX_SYMBOLS = (
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
    "USD_SEK",
)
CONTEXT_SYMBOLS = FX_SYMBOLS + ("XAU_USD", "USOIL", "DXY")
HORIZONS_BY_TIMEFRAME = {
    "H1": (24, 72, 168),
    "H4": (6, 18, 42),
}
LAG_SCAN_LAGS = {
    "H1": tuple(range(-12, 13)),
    "H4": tuple(range(-6, 7)),
}
LAG_SCAN_PAIRS = (
    ("XAU_USD", "USOIL"),
    ("USOIL", "DXY"),
)
SMT_PARTNERS: dict[str, tuple[tuple[str, int], ...]] = {
    "XAU_USD": (("DXY", -1), ("USD_JPY", 1)),
    "USOIL": (("USD_CAD", -1),),
}
GLOBAL_CONTEXT_SYMBOL = "GLOBAL"
GOLD_OIL_ALIAS = "gold_oil"
OIL_DXY_ALIAS = "oil_dxy"
ALIGNMENT_KEY = "__xasset_align_ts"
PAIR_ALIAS_OVERRIDES = {
    ("USOIL", "DXY"): OIL_DXY_ALIAS,
    ("DXY", "USOIL"): OIL_DXY_ALIAS,
    ("XAU_USD", "USOIL"): GOLD_OIL_ALIAS,
    ("USOIL", "XAU_USD"): GOLD_OIL_ALIAS,
}
VOL_NORM_HALFLIFE = {
    "H1": 24,
    "H4": 12,
}
VOL_FLOOR = 1e-6
_NORMAL = NormalDist()


@dataclass(slots=True)
class RawFrameCacheEntry:
    frame: pd.DataFrame
    bytes_read: int


def cross_asset_runtime_config_hash(
    *,
    timeframe: str,
    relevant_pairs: frozenset[tuple[str, str]] | None,
) -> str:
    payload = {
        "timeframe": timeframe,
        "supported_timeframes": sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES),
        "context_symbols": list(CONTEXT_SYMBOLS),
        "horizons_by_timeframe": {
            key: list(values) for key, values in HORIZONS_BY_TIMEFRAME.items()
        },
        "lag_scan_lags": {key: list(values) for key, values in LAG_SCAN_LAGS.items()},
        "lag_scan_pairs": [list(pair) for pair in LAG_SCAN_PAIRS],
        "smt_partners": {
            key: [list(item) for item in values] for key, values in SMT_PARTNERS.items()
        },
        "vol_norm_halflife": dict(VOL_NORM_HALFLIFE),
        "vol_floor": VOL_FLOOR,
        "relevant_pairs": (
            None
            if relevant_pairs is None
            else [list(pair) for pair in sorted(relevant_pairs)]
        ),
    }
    return fingerprint_mapping(payload)


def market_context_summary_path(
    *,
    features_root: str | Path,
    timeframe: str,
    variant: str,
) -> Path:
    return (
        canonical_dataset_root(
            features_root,
            dataset=f"market_context_{variant}",
            symbol=GLOBAL_CONTEXT_SYMBOL,
            timeframe=timeframe,
        )
        / "summary.json"
    )


def read_market_context_summary(
    *,
    features_root: str | Path,
    timeframe: str,
    variant: str,
) -> dict[str, object] | None:
    path = market_context_summary_path(
        features_root=features_root, timeframe=timeframe, variant=variant
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def market_context_cache_is_current(
    *,
    features_root: str | Path,
    timeframe: str,
    variant: str,
    relevant_pairs: frozenset[tuple[str, str]] | None,
) -> bool:
    summary = read_market_context_summary(
        features_root=features_root,
        timeframe=timeframe,
        variant=variant,
    )
    if not summary:
        return False
    expected = cross_asset_runtime_config_hash(
        timeframe=timeframe, relevant_pairs=relevant_pairs
    )
    return summary.get("config_hash") == expected


def _is_supported_timeframe(timeframe: str) -> bool:
    return timeframe in SUPPORTED_CROSS_ASSET_TIMEFRAMES


def _is_commodity_symbol(symbol: str) -> bool:
    return symbol in {"XAU_USD", "USOIL"}


def _partner_token(symbol: str) -> str:
    return symbol.lower()


def _pair_alias(a: str, b: str) -> str:
    return PAIR_ALIAS_OVERRIDES.get((a, b), f"{a.lower()}_{b.lower()}")


def aligned_timestamp_for_instrument(
    timestamp: pd.Series | pd.Index,
    *,
    instrument: str,
    timeframe: str,
) -> pd.Series:
    ts = pd.Series(pd.to_datetime(timestamp, utc=True, errors="coerce"), copy=False)
    if timeframe == "H4" and _is_commodity_symbol(instrument):
        return ts - pd.Timedelta(hours=1)
    return ts


def add_alignment_key(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    key: str = ALIGNMENT_KEY,
) -> pd.DataFrame:
    out = df.copy()
    out[key] = aligned_timestamp_for_instrument(
        out["timestamp"],
        instrument=instrument,
        timeframe=timeframe,
    )
    return out


def _prepare_close_frame(
    frame: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame:
    working = add_alignment_key(frame, instrument=instrument, timeframe=timeframe)
    out = working[[ALIGNMENT_KEY, "close"]].copy()
    out = out.rename(columns={ALIGNMENT_KEY: "timestamp", "close": instrument})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", instrument])
    out = out.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    out[instrument] = out[instrument].astype(float)
    return out.reset_index(drop=True)


def _iter_correlation_pairs() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for idx, left in enumerate(FX_SYMBOLS):
        for right in FX_SYMBOLS[idx + 1 :]:
            pairs.append((left, right))
    for fx in FX_SYMBOLS:
        pairs.append((fx, "XAU_USD"))
        pairs.append((fx, "USOIL"))
    pairs.append(("XAU_USD", "USOIL"))
    pairs.append(("DXY", "XAU_USD"))
    pairs.append(("DXY", "USOIL"))
    return tuple(pairs)


def _canonical_pair(a: str, b: str) -> tuple[str, str] | None:
    """Return the canonical (left, right) ordering used by ``_iter_correlation_pairs``.

    Returns None if the pair is not in the canonical set (e.g. DXY×FX pairs
    are not computed since DXY contains those FX components by construction).
    """
    pairs = _iter_correlation_pairs()
    if (a, b) in pairs:
        return (a, b)
    if (b, a) in pairs:
        return (b, a)
    return None


def relevant_correlation_pairs(primary: str) -> frozenset[tuple[str, str]]:
    """Return correlation pairs whose values are consumed by the primary instrument.

    ``_attach_named_columns`` only reads correlations for a specific subset
    of partners depending on the primary instrument:

    - FX primary: correlations vs every other FX + XAU_USD + USOIL.
    - XAU_USD: correlations vs every FX + USOIL + DXY.
    - USOIL: correlations vs every FX + XAU_USD + DXY.
    - DXY: correlations vs XAU_USD + USOIL only.

    For multi-instrument research / audit use cases, pass ``relevant_pairs=None``
    to ``build_global_market_context`` to compute the full N² matrix.
    """
    if primary in FX_SYMBOLS:
        partners = set(FX_SYMBOLS) - {primary}
        partners |= {"XAU_USD", "USOIL"}
    elif primary == "XAU_USD":
        partners = set(FX_SYMBOLS) | {"USOIL", "DXY"}
    elif primary == "USOIL":
        partners = set(FX_SYMBOLS) | {"XAU_USD", "DXY"}
    elif primary == "DXY":
        partners = {"XAU_USD", "USOIL"}
    else:
        return frozenset()
    pairs = set()
    for partner in partners:
        canonical = _canonical_pair(primary, partner)
        if canonical is not None:
            pairs.add(canonical)
    return frozenset(pairs)


def _corr_min_periods(window: int) -> int:
    """Minimum valid observations for a rolling correlation window.

    Cross-asset instruments trade different session hours (e.g. gold ~22h/day,
    FX ~24h/day).  When their timestamps are outer-merged, each 24-bar window
    has ~2 NaN holes from session gaps.  Using ``min_periods == window``
    (the pandas default) causes ALL correlations to be NaN whenever the two
    symbols have different trading calendars.  A 75 % fill requirement
    tolerates the expected session gaps while remaining statistically sound.
    """
    return max(2, int(window * 0.75))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    min_p = _corr_min_periods(window)
    mean = series.rolling(window, min_periods=min_p).mean()
    std = series.rolling(window, min_periods=min_p).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _vol_normalize_returns(
    returns: pd.Series,
    *,
    timeframe: str,
) -> tuple[pd.Series, pd.Series]:
    half_life = VOL_NORM_HALFLIFE[timeframe]
    ewma_vol = returns.ewm(
        halflife=half_life,
        adjust=False,
        min_periods=max(5, half_life // 2),
    ).std(bias=False)
    baseline = pd.to_numeric(ewma_vol, errors="coerce").dropna()
    if baseline.empty:
        floor = VOL_FLOOR
    else:
        floor = max(VOL_FLOOR, float(baseline.median()) * 0.10)
    safe_vol = ewma_vol.clip(lower=floor)
    standardized = returns / safe_vol
    winsorized = standardized.clip(lower=-5.0, upper=5.0)
    return standardized.astype(float), winsorized.astype(float)


def _normal_pvalue(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return float("nan")
    return float(2.0 * (1.0 - _NORMAL.cdf(abs(float(t_stat)))))


def _newey_west_bandwidth(n_obs: int) -> int:
    if n_obs <= 1:
        return 0
    return max(1, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))


def _newey_west_t_stat(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    n_obs = int(len(x_arr))
    if n_obs < 5:
        return float("nan")
    design = np.column_stack([np.ones(n_obs, dtype=float), x_arr])
    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
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
        return float("nan")
    return float(beta[1] / se)


def _latest_tail_overlap(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
    lag: int = 0,
) -> tuple[pd.Series, pd.Series]:
    aligned = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right.shift(lag), errors="coerce"),
        }
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    tail = aligned.tail(window).reset_index(drop=True)
    return tail["left"].astype(float), tail["right"].astype(float)


def _rolling_sign_stability(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
) -> float:
    min_p = _corr_min_periods(window)
    rolling = (
        pd.to_numeric(left, errors="coerce")
        .rolling(window, min_periods=min_p)
        .corr(pd.to_numeric(right, errors="coerce"))
    )
    valid = rolling.dropna().to_numpy(dtype=float)
    if len(valid) <= 1:
        return float("nan")
    flips = np.mean(np.sign(valid[1:]) != np.sign(valid[:-1]))
    return float(np.clip(1.0 - flips, 0.0, 1.0))


def _live_significance_class(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
) -> str:
    x_tail, y_tail = _latest_tail_overlap(left, right, window=window, lag=0)
    n_obs = len(x_tail)
    min_obs = max(30.0, 1.5 * float(window))
    if n_obs < min_obs:
        return "noise"
    corr = float(x_tail.corr(y_tail))
    if not np.isfinite(corr):
        return "noise"
    denom = max(1.0 - (corr * corr), 1e-12)
    classic_t = corr * np.sqrt(max(n_obs - 2, 1) / denom)
    classic_p = _normal_pvalue(float(classic_t))
    hac_t = _newey_west_t_stat(x_tail, y_tail)
    hac_p = _normal_pvalue(hac_t)
    stability = _rolling_sign_stability(left, right, window=window)
    if (
        np.isfinite(hac_p)
        and hac_p <= 0.05
        and np.isfinite(stability)
        and stability >= 0.75
        and abs(corr) >= 0.20
    ):
        return "research_grade"
    if (
        np.isfinite(hac_p)
        and hac_p <= 0.05
        and np.isfinite(stability)
        and stability >= 0.60
    ):
        return "significant"
    if np.isfinite(classic_p) and classic_p <= 0.10:
        return "watch"
    return "noise"


def _live_stability_class(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
) -> str:
    min_p = _corr_min_periods(window)
    rolling = (
        pd.to_numeric(left, errors="coerce")
        .rolling(window, min_periods=min_p)
        .corr(pd.to_numeric(right, errors="coerce"))
    )
    valid = rolling.dropna()
    if valid.empty:
        return "unknown"
    latest = float(valid.iloc[-1])
    std = float(valid.std(ddof=0))
    mean = float(valid.mean())
    zscore = (latest - mean) / std if std > 0.0 else float("nan")
    sign_stability = _rolling_sign_stability(left, right, window=window)
    if (
        np.isfinite(sign_stability)
        and sign_stability >= 0.80
        and np.isfinite(zscore)
        and abs(zscore) >= 1.0
    ):
        return "stable"
    if np.isfinite(sign_stability) and sign_stability >= 0.60:
        return "mixed"
    return "unstable"


def _rolling_best_lag(
    left: pd.Series,
    right: pd.Series,
    *,
    window: int,
    lags: tuple[int, ...],
) -> tuple[pd.Series, pd.Series]:
    min_p = _corr_min_periods(window)
    corr_by_lag = {
        lag: left.rolling(window, min_periods=min_p).corr(right.shift(lag))
        for lag in lags
    }
    corr_frame = pd.DataFrame(corr_by_lag, index=left.index, dtype=float)
    valid_mask = corr_frame.notna().any(axis=1)
    abs_corr = corr_frame.abs().fillna(-np.inf)
    best_lag = abs_corr.idxmax(axis=1).astype(float)
    best_lag = best_lag.where(valid_mask, np.nan)
    best_score = pd.Series(np.nan, index=left.index, dtype=float)
    for lag in lags:
        best_score = best_score.where(best_lag != float(lag), corr_frame[lag])
    return best_score, best_lag


def build_global_market_context(
    processed_frames: Mapping[str, pd.DataFrame],
    *,
    timeframe: str,
    relevant_pairs: frozenset[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build the global cross-asset market context frame.

    Parameters
    ----------
    processed_frames
        Mapping ``symbol -> raw OHLC frame`` for the symbols to include.
    timeframe
        ``H1`` or ``H4``.
    relevant_pairs
        Optional subset of ``(left, right)`` pairs to compute correlations
        for. When ``None`` (default), all 55 canonical pairs from
        ``_iter_correlation_pairs()`` are computed — needed for the
        cross-asset audit / multi-instrument research path. When provided,
        only those pairs' rolling correlations, z-scores, significance and
        stability classifications are computed; non-pair columns (returns,
        lag-scan) are unaffected.

    See ``relevant_correlation_pairs(primary)`` for the per-instrument
    relevance filter used by single-instrument pipeline runs.
    """
    if not _is_supported_timeframe(timeframe):
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    prepared: list[pd.DataFrame] = []
    available_symbols: list[str] = []
    for instrument in CONTEXT_SYMBOLS:
        frame = processed_frames.get(instrument)
        if frame is None or frame.empty or "timestamp" not in frame.columns:
            continue
        prepared.append(
            _prepare_close_frame(frame, instrument=instrument, timeframe=timeframe)
        )
        available_symbols.append(instrument)

    if not prepared:
        return pd.DataFrame(columns=["timestamp", "timeframe"])

    context = prepared[0]
    for frame in prepared[1:]:
        context = context.merge(frame, on="timestamp", how="outer")
    context = context.sort_values("timestamp").reset_index(drop=True)
    context["timeframe"] = timeframe

    pairs_to_compute = (
        _iter_correlation_pairs()
        if relevant_pairs is None
        else tuple(p for p in _iter_correlation_pairs() if p in relevant_pairs)
    )

    # Suppress expected numpy warnings from sparse outer-merged data.
    # Rolling windows with fewer than 2 finite observations produce NaN
    # correctly — the RuntimeWarnings (degrees of freedom, divide by zero,
    # invalid value) are expected noise, not bugs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for instrument in available_symbols:
            price = pd.to_numeric(context[instrument], errors="coerce").astype(float)
            raw_returns = np.log(price).diff().astype(float)
            vol_norm_returns, winsorized_returns = _vol_normalize_returns(
                raw_returns,
                timeframe=timeframe,
            )
            context[f"ret_raw_{instrument}"] = raw_returns
            context[f"ret_vol_{instrument}"] = vol_norm_returns
            context[f"ret_vol_winsor_{instrument}"] = winsorized_returns

        for left, right in pairs_to_compute:
            if left not in available_symbols or right not in available_symbols:
                continue
            left_ret = context[f"ret_vol_{left}"]
            right_ret = context[f"ret_vol_{right}"]
            for window in HORIZONS_BY_TIMEFRAME[timeframe]:
                corr_col = f"corr_{left}__{right}__w{window}"
                z_col = f"corr_z_{left}__{right}__w{window}"
                sig_col = f"corr_sig_class_{left}__{right}__w{window}"
                stability_col = f"corr_stability_class_{left}__{right}__w{window}"
                min_p = _corr_min_periods(window)
                corr = left_ret.rolling(window, min_periods=min_p).corr(right_ret)
                context[corr_col] = corr
                context[z_col] = _rolling_zscore(corr, window)
                context[sig_col] = _live_significance_class(
                    left_ret,
                    right_ret,
                    window=window,
                )
                context[stability_col] = _live_stability_class(
                    left_ret,
                    right_ret,
                    window=window,
                )

        for left, right in LAG_SCAN_PAIRS:
            if left not in available_symbols or right not in available_symbols:
                continue
            left_ret = context[f"ret_vol_{left}"]
            right_ret = context[f"ret_vol_{right}"]
            for window in HORIZONS_BY_TIMEFRAME[timeframe]:
                best_score, best_lag = _rolling_best_lag(
                    left_ret,
                    right_ret,
                    window=window,
                    lags=LAG_SCAN_LAGS[timeframe],
                )
                score_col = f"lagcorr_best_{left}__{right}__w{window}"
                lag_col = f"lagcorr_best_lag_{left}__{right}__w{window}"
                context[score_col] = best_score
                context[lag_col] = best_lag

    keep_columns = [
        "timestamp",
        "timeframe",
        *[
            column
            for column in context.columns
            if column.startswith("corr_")
            or column.startswith("corr_z_")
            or column.startswith("lagcorr_best_")
            or column.startswith("ret_raw_")
            or column.startswith("ret_vol_")
        ],
    ]
    return context[keep_columns].copy()


def build_global_market_context_incremental(
    processed_frames: Mapping[str, pd.DataFrame],
    *,
    timeframe: str,
    prior_context: pd.DataFrame,
    frontier_from_ts: pd.Timestamp,
    relevant_pairs: frozenset[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Recompute the market context only for ``[frontier_from_ts, now]``.

    The frontier window is extended backward by ``_warmup_buffer(timeframe)``
    so rolling correlations have enough history to warm up to the same
    values as a full rebuild.

    The recomputed frontier is then stitched onto ``prior_context`` via
    ``merge_recomputed_frontier``: rows strictly before ``frontier_from_ts``
    are taken from ``prior_context`` (frozen), rows from
    ``frontier_from_ts`` onward come from the freshly computed slice.

    Parameters
    ----------
    processed_frames
        Raw OHLC frames keyed by symbol — same shape as ``build_global_market_context``.
        Frames must extend back to at least ``frontier_from_ts - _warmup_buffer``
        for the frontier rolling windows to warm up correctly.
    prior_context
        The market context as produced by a previous pipeline run. Loaded
        via ``load_partitioned_dataset(... dataset='market_context_research', ...)``.
    frontier_from_ts
        Earliest timestamp to overwrite from prior_context. Typically the
        max timestamp in ``prior_context`` minus ``_warmup_buffer``.
    relevant_pairs
        Forwarded to the underlying full builder. ``None`` = full matrix.

    Notes
    -----
    Correctness is guaranteed by recomputing far enough back that the
    rolling window at ``frontier_from_ts`` has the same observation count
    as a full rebuild. Tested in
    ``tests/test_incremental_market_context.py``.
    """
    if prior_context is None or prior_context.empty:
        return build_global_market_context(
            processed_frames,
            timeframe=timeframe,
            relevant_pairs=relevant_pairs,
        )

    if frontier_from_ts.tzinfo is None:
        frontier_from_ts = frontier_from_ts.tz_localize("UTC")

    warmup = _warmup_buffer(timeframe)
    compute_from = frontier_from_ts - warmup

    trimmed: dict[str, pd.DataFrame] = {}
    for symbol, frame in processed_frames.items():
        if frame is None or frame.empty:
            continue
        trimmed[symbol] = _trim_frame_to_range(frame, min_ts=compute_from)

    frontier_context = build_global_market_context(
        trimmed,
        timeframe=timeframe,
        relevant_pairs=relevant_pairs,
    )

    if frontier_context.empty:
        return prior_context.copy()

    return merge_recomputed_frontier(
        prior_context,
        frontier_context,
        frontier_from_ts=frontier_from_ts,
    )


def load_raw_context_frames(
    *,
    raw_data_root: str | Path,
    timeframe: str,
    instruments: tuple[str, ...] = CONTEXT_SYMBOLS,
    frame_cache: dict[tuple[str, int, int], RawFrameCacheEntry] | None = None,
    runtime_details: dict[str, object] | None = None,
) -> dict[str, pd.DataFrame]:
    root = Path(raw_data_root)
    out: dict[str, pd.DataFrame] = {}
    details = runtime_details if runtime_details is not None else {}
    details.setdefault("raw_frame_cache_hits", 0)
    details.setdefault("raw_frame_disk_reads", 0)
    details.setdefault("raw_frame_read_bytes", 0)
    details.setdefault("raw_frame_symbols_loaded", [])
    for instrument in instruments:
        path = root / f"{instrument}_{timeframe}.parquet"
        if not path.exists():
            continue
        stat = path.stat()
        cache_key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
        entry = frame_cache.get(cache_key) if frame_cache is not None else None
        if entry is not None:
            details["raw_frame_cache_hits"] = int(details["raw_frame_cache_hits"]) + 1
            frame = entry.frame
        else:
            frame = normalize_candle_schema(pd.read_parquet(path), require_volume=True)
            if frame_cache is not None:
                frame_cache[cache_key] = RawFrameCacheEntry(
                    frame=frame.copy(),
                    bytes_read=int(stat.st_size),
                )
            details["raw_frame_disk_reads"] = int(details["raw_frame_disk_reads"]) + 1
            details["raw_frame_read_bytes"] = int(
                details["raw_frame_read_bytes"]
            ) + int(stat.st_size)
            symbols_loaded = list(details["raw_frame_symbols_loaded"])
            symbols_loaded.append(instrument)
            details["raw_frame_symbols_loaded"] = symbols_loaded
        out[instrument] = frame.reset_index(drop=True)
    return out


def build_processed_context_frames(
    *,
    primary_raw: pd.DataFrame,
    instrument: str,
    timeframe: str,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None,
    raw_data_root: str | Path | None,
    frame_builder: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    universe = dict(peer_raw_frames or {})
    if raw_data_root is not None:
        loaded = load_raw_context_frames(
            raw_data_root=raw_data_root, timeframe=timeframe
        )
        for key, value in loaded.items():
            universe.setdefault(key, value)
    universe[instrument] = primary_raw

    processed: dict[str, pd.DataFrame] = {}
    for symbol, raw in universe.items():
        if raw is None or raw.empty:
            continue
        processed[symbol] = frame_builder(raw.copy(), symbol)
    return processed


def _warmup_buffer(timeframe: str) -> pd.Timedelta:
    """Extra bars needed before the primary range for rolling window warmup.

    Sized to satisfy two warmup needs:

    1. Rolling correlation windows up to ``max(HORIZONS_BY_TIMEFRAME)``.
    2. EWMA volatility normalization. EWMA is path-dependent; an
       incremental rebuild that starts ``k`` half-lives before the cutoff
       converges to within ``2^-k`` of the full-history value. We use 5
       half-lives so residual drift on z-scored correlations is well
       below the noise floor of any downstream signal.
    """
    max_horizon = max(HORIZONS_BY_TIMEFRAME[timeframe])
    vol_warmup = VOL_NORM_HALFLIFE[timeframe] * 5
    total_bars = max_horizon + vol_warmup
    bar_seconds = {"H1": 3600, "H4": 14400}.get(timeframe, 3600)
    return pd.Timedelta(seconds=total_bars * bar_seconds)


def _trim_frame_to_range(
    frame: pd.DataFrame,
    *,
    min_ts: pd.Timestamp,
    max_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Trim a raw frame to ``[min_ts, max_ts]`` inclusive.

    ``max_ts=None`` keeps everything from ``min_ts`` onward.
    """
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = ts >= min_ts
    if max_ts is not None:
        mask &= ts <= max_ts
    return frame.loc[mask].copy()


def resolve_cross_asset_inputs(
    primary_raw: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    market_context: pd.DataFrame | None = None,
    processed_frames: Mapping[str, pd.DataFrame] | None = None,
    peer_raw_frames: Mapping[str, pd.DataFrame] | None = None,
    raw_data_root: str | Path | None = "data/raw",
    partner_builder: Callable[[pd.DataFrame, str], pd.DataFrame] | None = None,
    full_pair_matrix: bool = False,
    frame_cache: dict[tuple[str, int, int], RawFrameCacheEntry] | None = None,
    runtime_details: dict[str, object] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
    if not _is_supported_timeframe(timeframe):
        raise ValueError(
            f"Cross-asset context only supports {sorted(SUPPORTED_CROSS_ASSET_TIMEFRAMES)}"
        )

    details = runtime_details if runtime_details is not None else {}
    details.setdefault("market_context_source", "provided")
    details.setdefault("partner_build_count", 0)
    details.setdefault("partner_symbols_built", [])
    details.setdefault(
        "relevant_pairs_mode", "full" if full_pair_matrix else "filtered"
    )
    details.setdefault("trimmed_symbols", [])

    # Determine the primary time range so peer data can be trimmed to it.
    primary_ts = pd.to_datetime(
        primary_raw["timestamp"], utc=True, errors="coerce"
    ).dropna()
    primary_max = primary_ts.max()
    primary_min = primary_ts.min() - _warmup_buffer(timeframe)

    raw_universe: dict[str, pd.DataFrame] = {}
    for symbol, frame in (peer_raw_frames or {}).items():
        if frame is None or frame.empty:
            continue
        raw_universe[symbol] = normalize_candle_schema(frame, require_volume=True)
    if raw_data_root is not None:
        loaded = load_raw_context_frames(
            raw_data_root=raw_data_root,
            timeframe=timeframe,
            frame_cache=frame_cache,
            runtime_details=details,
        )
        for symbol, frame in loaded.items():
            raw_universe.setdefault(symbol, frame)

    trimmed_cache: dict[str, pd.DataFrame] = {}

    def _trimmed(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
        cached = trimmed_cache.get(symbol)
        if cached is not None:
            return cached
        trimmed = _trim_frame_to_range(frame, min_ts=primary_min, max_ts=primary_max)
        trimmed_cache[symbol] = trimmed
        trimmed_symbols = list(details["trimmed_symbols"])
        if symbol not in trimmed_symbols:
            trimmed_symbols.append(symbol)
        details["trimmed_symbols"] = trimmed_symbols
        return trimmed

    resolved_market_context = market_context
    if resolved_market_context is None:
        raw_context_frames = {
            symbol: _trimmed(symbol, frame)
            for symbol, frame in raw_universe.items()
            if frame is not None and not frame.empty
        }
        if instrument in CONTEXT_SYMBOLS:
            raw_context_frames[instrument] = normalize_candle_schema(
                primary_raw,
                require_volume=True,
            )
        relevant = None if full_pair_matrix else relevant_correlation_pairs(instrument)
        details["market_context_source"] = "build"
        details["market_context_config_hash"] = cross_asset_runtime_config_hash(
            timeframe=timeframe,
            relevant_pairs=relevant,
        )
        resolved_market_context = build_global_market_context(
            raw_context_frames,
            timeframe=timeframe,
            relevant_pairs=relevant,
        )

    resolved_processed = dict(processed_frames or {})
    missing_partners = [
        partner
        for partner, _ in SMT_PARTNERS.get(instrument, ())
        if partner not in resolved_processed
    ]
    if missing_partners and partner_builder is not None:
        for partner in missing_partners:
            raw_partner = raw_universe.get(partner)
            if raw_partner is None or raw_partner.empty:
                continue
            trimmed_partner = _trimmed(partner, raw_partner)
            resolved_processed[partner] = partner_builder(
                trimmed_partner.copy(), partner
            )
            details["partner_build_count"] = int(details["partner_build_count"]) + 1
            built = list(details["partner_symbols_built"])
            built.append(partner)
            details["partner_symbols_built"] = built

    return resolved_market_context, resolved_processed


def _pair_column_candidates(
    prefix: str, a: str, b: str, window: int
) -> tuple[str, str]:
    return (
        f"{prefix}_{a}__{b}__w{window}",
        f"{prefix}_{b}__{a}__w{window}",
    )


def _attach_named_columns(
    out: pd.DataFrame,
    *,
    market_context: pd.DataFrame,
    instrument: str,
    timeframe: str,
) -> pd.DataFrame:
    working = add_alignment_key(out, instrument=instrument, timeframe=timeframe)
    context = market_context.copy()
    context["timestamp"] = pd.to_datetime(
        context["timestamp"], utc=True, errors="coerce"
    )
    merged = working.merge(
        context,
        left_on=ALIGNMENT_KEY,
        right_on="timestamp",
        how="left",
        suffixes=("", "__context"),
    )
    merged = merged.drop(columns=[ALIGNMENT_KEY, "timestamp__context"], errors="ignore")

    if instrument in FX_SYMBOLS:
        partners = [symbol for symbol in FX_SYMBOLS if symbol != instrument] + [
            "XAU_USD",
            "USOIL",
        ]
        lag_pairs: list[tuple[str, str]] = []
    elif instrument == "XAU_USD":
        partners = list(FX_SYMBOLS) + ["USOIL", "DXY"]
        lag_pairs = [("XAU_USD", "USOIL")]
    elif instrument == "USOIL":
        partners = list(FX_SYMBOLS) + ["XAU_USD", "DXY"]
        lag_pairs = [("XAU_USD", "USOIL"), ("USOIL", "DXY")]
    elif instrument == "DXY":
        partners = ["XAU_USD", "USOIL"]
        lag_pairs = [("USOIL", "DXY")]
    else:
        partners = []
        lag_pairs = []

    for partner in partners:
        partner_token = _partner_token(partner)
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            for candidate in _pair_column_candidates(
                "corr", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_{partner_token}_w{window}"] = merged[candidate]
                    break
            for candidate in _pair_column_candidates(
                "corr_z", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_z_{partner_token}_w{window}"] = merged[
                        candidate
                    ]
                    break
            for candidate in _pair_column_candidates(
                "corr_sig_class", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_sig_class_{partner_token}_w{window}"] = merged[
                        candidate
                    ]
                    break
            for candidate in _pair_column_candidates(
                "corr_stability_class", instrument, partner, window
            ):
                if candidate in merged.columns:
                    merged[f"xasset_corr_stability_class_{partner_token}_w{window}"] = (
                        merged[candidate]
                    )
                    break

    for left, right in lag_pairs:
        pair_alias = _pair_alias(left, right)
        for window in HORIZONS_BY_TIMEFRAME[timeframe]:
            score_col = f"lagcorr_best_{left}__{right}__w{window}"
            lag_col = f"lagcorr_best_lag_{left}__{right}__w{window}"
            if score_col in merged.columns:
                merged[f"xasset_lagcorr_{pair_alias}_best_w{window}"] = merged[
                    score_col
                ]
            if lag_col in merged.columns:
                merged[f"xasset_lagcorr_{pair_alias}_best_lag_w{window}"] = merged[
                    lag_col
                ]

    drop_candidates = [
        column
        for column in merged.columns
        if column.startswith("corr_")
        or column.startswith("corr_z_")
        or column.startswith("lagcorr_best_")
        or column.startswith("ret_raw_")
        or column.startswith("ret_vol_")
        or column in CONTEXT_SYMBOLS
        or column == "timeframe"
    ]
    return merged.drop(columns=drop_candidates, errors="ignore")


def _align_partner_for_primary(
    primary_frame: pd.DataFrame,
    partner_frame: pd.DataFrame,
    *,
    primary_instrument: str,
    partner_instrument: str,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = add_alignment_key(
        primary_frame,
        instrument=primary_instrument,
        timeframe=timeframe,
    )
    right = add_alignment_key(
        partner_frame,
        instrument=partner_instrument,
        timeframe=timeframe,
    )
    partner_columns = [
        "timestamp",
        ALIGNMENT_KEY,
        *[
            column
            for column in partner_frame.columns
            if column in {"close", "atr_14", *SMT_REQUIRED_COLUMNS_FOR_ATTACH()}
            or column.startswith("swing_")
        ],
    ]
    partner_aligned = right[partner_columns].copy()
    merged = left[["timestamp", ALIGNMENT_KEY]].merge(
        partner_aligned,
        on=ALIGNMENT_KEY,
        how="left",
        suffixes=("", "__partner"),
    )
    partner_out = merged.drop(columns=[ALIGNMENT_KEY]).copy()
    primary_out = left.drop(columns=[ALIGNMENT_KEY]).copy()
    return primary_out, partner_out


def SMT_REQUIRED_COLUMNS_FOR_ATTACH() -> tuple[str, ...]:
    return (
        "swing_high_confirm_flag",
        "swing_low_confirm_flag",
        "swing_high_confirm_origin_idx",
        "swing_low_confirm_origin_idx",
        "swing_high_confirm_price",
        "swing_low_confirm_price",
    )


def attach_cross_asset_context(
    frame: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    market_context: pd.DataFrame | None,
    processed_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    if not _is_supported_timeframe(timeframe):
        return out

    if market_context is not None and not market_context.empty:
        out = _attach_named_columns(
            out,
            market_context=market_context,
            instrument=instrument,
            timeframe=timeframe,
        )

    partner_specs = SMT_PARTNERS.get(instrument, ())
    best_partner = np.full(len(out), None, dtype=object)
    best_score = np.full(len(out), np.nan, dtype=float)
    any_flag = np.zeros(len(out), dtype=np.int8)

    for partner, relation_sign in partner_specs:
        if processed_frames is None or partner not in processed_frames:
            continue
        primary_aligned, partner_aligned = _align_partner_for_primary(
            out,
            processed_frames[partner],
            primary_instrument=instrument,
            partner_instrument=partner,
            timeframe=timeframe,
        )
        smt_frame = add_smt_divergence(
            primary_aligned,
            partner_aligned,
            inverse=relation_sign == -1,
            partner_name=_partner_token(partner),
        )
        keep = [
            f"xasset_smt_{_partner_token(partner)}_bull_flag",
            f"xasset_smt_{_partner_token(partner)}_bear_flag",
            f"xasset_smt_{_partner_token(partner)}_dir",
            f"xasset_smt_{_partner_token(partner)}_score",
            f"xasset_smt_{_partner_token(partner)}_expected_relation",
        ]
        for column in keep:
            out[column] = smt_frame[column].to_numpy()
        score_col = f"xasset_smt_{_partner_token(partner)}_score"
        dir_col = f"xasset_smt_{_partner_token(partner)}_dir"
        partner_score = pd.to_numeric(out[score_col], errors="coerce").to_numpy(
            dtype=float
        )
        partner_dir = (
            pd.to_numeric(out[dir_col], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int8)
        )
        partner_active = np.isfinite(partner_score) & (partner_dir != 0)
        replace_mask = partner_active & (
            ~np.isfinite(best_score) | (partner_score > best_score)
        )
        best_score = np.where(replace_mask, partner_score, best_score)
        best_partner = np.where(replace_mask, partner, best_partner)
        any_flag = np.where(partner_active, 1, any_flag).astype(np.int8)

    out["xasset_smt_any_flag"] = any_flag
    out["xasset_smt_best_partner"] = pd.Series(best_partner, dtype="object").where(
        pd.Series(any_flag == 1),
        None,
    )
    out["xasset_smt_best_score"] = pd.Series(best_score).where(any_flag == 1, np.nan)
    return out


def persist_market_context(
    market_context: pd.DataFrame,
    *,
    features_root: str | Path,
    timeframe: str,
    variant: str,
    frontier_from_ts: pd.Timestamp | None,
    full_rebuild: bool,
    relevant_pairs: frozenset[tuple[str, str]] | None = None,
) -> list[ArtifactWriteResult]:
    if market_context.empty:
        return []
    dataset = f"market_context_{variant}"
    artifacts = persist_partitioned_dataset(
        market_context,
        base_dir=features_root,
        dataset=dataset,
        symbol=GLOBAL_CONTEXT_SYMBOL,
        timeframe=timeframe,
        frontier_from_ts=frontier_from_ts,
        full_rebuild=full_rebuild,
    )
    summary_artifact = write_json_atomic(
        {
            "variant": variant,
            "timeframe": timeframe,
            "row_count": int(len(market_context)),
            "column_count": int(len(market_context.columns)),
            "config_hash": cross_asset_runtime_config_hash(
                timeframe=timeframe,
                relevant_pairs=relevant_pairs,
            ),
        },
        market_context_summary_path(
            features_root=features_root,
            timeframe=timeframe,
            variant=variant,
        ),
    )
    artifacts.append(summary_artifact)
    return artifacts
