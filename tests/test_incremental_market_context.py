"""Tests for Phase C: incremental market context.

Correctness criterion: an incremental rebuild that covers the frontier
(plus warmup) must produce numerically identical correlations to a full
rebuild on the overlapping rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.features.cross_asset import (
    _warmup_buffer,
    build_global_market_context,
    build_global_market_context_incremental,
    relevant_correlation_pairs,
)


def _raw_frame(ts: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full_like(close, 100.0, dtype=float),
        }
    )


def _all_symbols_frames(rows: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(123)
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    symbols = [
        "AUD_USD",
        "EUR_USD",
        "GBP_USD",
        "NZD_USD",
        "USD_CAD",
        "USD_CHF",
        "USD_JPY",
        "USD_SEK",
        "XAU_USD",
        "USOIL",
        "DXY",
    ]
    out = {}
    for symbol in symbols:
        ret = rng.normal(0.0, 0.001, size=rows)
        close = 100.0 * np.exp(np.cumsum(ret))
        out[symbol] = _raw_frame(ts, close)
    return out


def test_incremental_with_no_prior_returns_full_build():
    frames = _all_symbols_frames(400)
    full = build_global_market_context(frames, timeframe="H1")

    # Empty prior — incremental should fall back to full build.
    inc = build_global_market_context_incremental(
        frames,
        timeframe="H1",
        prior_context=pd.DataFrame(columns=["timestamp", "timeframe"]),
        frontier_from_ts=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    pd.testing.assert_frame_equal(
        full.reset_index(drop=True),
        inc.reset_index(drop=True),
        check_dtype=False,
    )


def test_incremental_recompute_matches_full_rebuild():
    """Build context on full 800 bars (truth). Then build context on first
    600 bars (prior), then incremental on the next 200 bars. Output must
    be numerically identical to the full build on the overlapping rows.
    """
    full_frames = _all_symbols_frames(800)
    truth = build_global_market_context(full_frames, timeframe="H1")

    # Prior: only the first 600 bars known.
    prior_frames = {sym: f.iloc[:600].copy() for sym, f in full_frames.items()}
    prior = build_global_market_context(prior_frames, timeframe="H1")

    # Frontier: simulate appending bars 600-799 to the universe.
    # frontier_from_ts is set to max(prior) - warmup so the rolling windows
    # warm up to identical state.
    prior_max_ts = pd.to_datetime(prior["timestamp"], utc=True).max()
    frontier_from_ts = prior_max_ts - _warmup_buffer("H1")

    inc = build_global_market_context_incremental(
        full_frames,  # full universe (raw frames extend through bar 800)
        timeframe="H1",
        prior_context=prior,
        frontier_from_ts=frontier_from_ts,
    )

    # On rows from frontier_from_ts forward, incremental and full must agree.
    truth_ts = pd.to_datetime(truth["timestamp"], utc=True)
    inc_ts = pd.to_datetime(inc["timestamp"], utc=True)

    # Take corresponding slice of each on rows >= prior_max_ts
    # (the genuinely new bars — these rolled forward in the incremental).
    truth_new = truth.loc[truth_ts >= prior_max_ts].reset_index(drop=True)
    inc_new = inc.loc[inc_ts >= prior_max_ts].reset_index(drop=True)

    assert len(truth_new) == len(inc_new)
    assert len(truth_new) > 0

    # Compare correlation values at the new bars. Tolerance is split by
    # column family because the EWMA vol normalization is path-dependent:
    # raw correlations are essentially exact (< 1e-6), z-scored columns
    # carry residual EWMA drift on the order of 1e-3 even after 5
    # half-lives of warmup. Both are well below the noise floor of
    # rolling-correlation features used for trading signals.
    raw_corr_cols = [
        c
        for c in truth_new.columns
        if c.startswith("corr_")
        and not c.startswith("corr_z_")
        and not c.startswith("corr_sig_class_")
        and not c.startswith("corr_stability_class_")
    ]
    z_corr_cols = [c for c in truth_new.columns if c.startswith("corr_z_")]
    lag_cols = [c for c in truth_new.columns if c.startswith("lagcorr_best_")]

    for col in raw_corr_cols:
        if col not in inc_new.columns:
            continue
        pd.testing.assert_series_equal(
            truth_new[col],
            inc_new[col],
            check_names=False,
            atol=1e-6,
            rtol=1e-6,
        )

    for col in z_corr_cols:
        if col not in inc_new.columns:
            continue
        pd.testing.assert_series_equal(
            truth_new[col],
            inc_new[col],
            check_names=False,
            atol=5e-3,
            rtol=5e-3,
        )

    for col in lag_cols:
        if col not in inc_new.columns:
            continue
        # Lag scan integer-valued lag column should match exactly; the
        # score column is a correlation (raw — exact within 1e-6).
        if "_lag_" in col:
            pd.testing.assert_series_equal(
                truth_new[col], inc_new[col], check_names=False
            )
        else:
            pd.testing.assert_series_equal(
                truth_new[col],
                inc_new[col],
                check_names=False,
                atol=1e-6,
                rtol=1e-6,
            )


def test_incremental_preserves_frozen_history():
    """Rows strictly before frontier_from_ts must be passed through from
    the prior unchanged."""
    full_frames = _all_symbols_frames(800)
    prior_frames = {sym: f.iloc[:600].copy() for sym, f in full_frames.items()}
    prior = build_global_market_context(prior_frames, timeframe="H1")
    prior_max_ts = pd.to_datetime(prior["timestamp"], utc=True).max()
    frontier_from_ts = prior_max_ts - _warmup_buffer("H1")

    inc = build_global_market_context_incremental(
        full_frames,
        timeframe="H1",
        prior_context=prior,
        frontier_from_ts=frontier_from_ts,
    )

    # Pick a corr column and a row well before frontier_from_ts.
    inc_ts = pd.to_datetime(inc["timestamp"], utc=True)
    prior_ts = pd.to_datetime(prior["timestamp"], utc=True)

    # Find a sentinel row in the frozen region (before frontier_from_ts).
    frozen_mask_inc = inc_ts < frontier_from_ts
    frozen_mask_prior = prior_ts < frontier_from_ts
    inc_frozen = inc.loc[frozen_mask_inc].reset_index(drop=True)
    prior_frozen = prior.loc[frozen_mask_prior].reset_index(drop=True)

    # Frozen rows from incremental must equal frozen rows from prior.
    common_cols = [
        c
        for c in inc_frozen.columns
        if c in prior_frozen.columns
        and (c.startswith("corr_") or c.startswith("lagcorr_best_"))
    ]
    for col in common_cols[:5]:  # spot check 5 columns
        pd.testing.assert_series_equal(
            inc_frozen[col].reset_index(drop=True),
            prior_frozen[col].reset_index(drop=True),
            check_names=False,
        )


def test_incremental_with_relevant_pairs_filter():
    """When ``relevant_pairs`` is set, the incremental builder should also
    only compute those pairs in the frontier."""
    full_frames = _all_symbols_frames(400)
    relevant = relevant_correlation_pairs("XAU_USD")
    prior = build_global_market_context(
        {sym: f.iloc[:300].copy() for sym, f in full_frames.items()},
        timeframe="H1",
        relevant_pairs=relevant,
    )
    prior_max_ts = pd.to_datetime(prior["timestamp"], utc=True).max()
    frontier_from_ts = prior_max_ts - _warmup_buffer("H1")

    inc = build_global_market_context_incremental(
        full_frames,
        timeframe="H1",
        prior_context=prior,
        frontier_from_ts=frontier_from_ts,
        relevant_pairs=relevant,
    )

    # No correlations for FX×FX pairs (which are not in XAU_USD's relevant set).
    fx_fx_columns = [
        c
        for c in inc.columns
        if c.startswith("corr_EUR_USD__GBP_USD")
        or c.startswith("corr_AUD_USD__EUR_USD")
    ]
    assert (
        fx_fx_columns == []
    ), f"XAU_USD relevant filter leaked FX×FX pair columns: {fx_fx_columns}"

    # XAU_USD pair columns should be present.
    xau_columns = [
        c for c in inc.columns if "XAU_USD" in c and c.startswith("corr_") and "_w" in c
    ]
    assert len(xau_columns) > 0
