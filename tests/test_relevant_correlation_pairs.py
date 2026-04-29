"""Tests for Phase B: relevance-filtered correlation pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.features.cross_asset import (
    _iter_correlation_pairs,
    build_global_market_context,
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


def _all_symbols_frames(rows: int = 200) -> dict[str, pd.DataFrame]:
    """Synthetic close series for every CONTEXT_SYMBOL."""
    rng = np.random.default_rng(42)
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


# --- relevant_correlation_pairs unit tests ---


def test_relevant_pairs_xau_usd():
    pairs = relevant_correlation_pairs("XAU_USD")
    # XAU_USD pairs: 8 FX + USOIL + DXY = 10 pairs
    assert len(pairs) == 10
    assert ("AUD_USD", "XAU_USD") in pairs
    assert ("XAU_USD", "USOIL") in pairs
    assert ("DXY", "XAU_USD") in pairs


def test_relevant_pairs_dxy_minimal():
    pairs = relevant_correlation_pairs("DXY")
    assert pairs == frozenset({("DXY", "XAU_USD"), ("DXY", "USOIL")})


def test_relevant_pairs_fx_excludes_dxy():
    pairs = relevant_correlation_pairs("EUR_USD")
    # EUR_USD pairs: 7 other FX + XAU_USD + USOIL = 9 pairs (no DXY)
    assert len(pairs) == 9
    # DXY is not paired with FX in the canonical set.
    assert all("DXY" not in p for p in pairs)


def test_relevant_pairs_subset_of_canonical():
    canonical = set(_iter_correlation_pairs())
    for primary in ("XAU_USD", "USOIL", "DXY", "EUR_USD", "USD_JPY"):
        rel = relevant_correlation_pairs(primary)
        assert rel.issubset(canonical), f"{primary} produced non-canonical pairs"


def test_relevant_pairs_unknown_returns_empty():
    assert relevant_correlation_pairs("UNKNOWN_SYMBOL") == frozenset()


# --- build_global_market_context filtering ---


def test_build_market_context_default_full_matrix():
    """Without ``relevant_pairs``, all canonical pairs are computed."""
    frames = _all_symbols_frames(200)
    context = build_global_market_context(frames, timeframe="H1")

    # Count distinct pair keys among corr_X__Y_wN columns (excluding the
    # _z, _sig_class, _stability_class variants which use the same pair).
    plain_corr = [
        c
        for c in context.columns
        if c.startswith("corr_")
        and not c.startswith("corr_z_")
        and not c.startswith("corr_sig_class_")
        and not c.startswith("corr_stability_class_")
        and "_w" in c
    ]
    n_pairs_in_context = len({c.rsplit("__w", 1)[0] for c in plain_corr})
    n_canonical = len(_iter_correlation_pairs())
    assert n_pairs_in_context == n_canonical


def test_build_market_context_filtered_subset():
    """With ``relevant_pairs=relevant_correlation_pairs('XAU_USD')``, only
    XAU_USD-relevant pair columns exist."""
    frames = _all_symbols_frames(200)
    relevant = relevant_correlation_pairs("XAU_USD")
    filtered = build_global_market_context(
        frames, timeframe="H1", relevant_pairs=relevant
    )

    corr_cols = [c for c in filtered.columns if c.startswith("corr_") and "__" in c]
    pair_keys = {c.rsplit("__w", 1)[0] for c in corr_cols if "_w" in c}

    # Each relevant pair appears exactly once (in canonical ordering).
    expected_keys = {f"corr_{l}__{r}" for l, r in relevant}
    expected_keys |= {f"corr_z_{l}__{r}" for l, r in relevant}
    expected_keys |= {f"corr_sig_class_{l}__{r}" for l, r in relevant}
    expected_keys |= {f"corr_stability_class_{l}__{r}" for l, r in relevant}
    assert pair_keys == expected_keys


def test_filtered_context_correlation_values_match_full():
    """Correlations for the filtered subset must be numerically identical to
    those produced by the full-matrix path. No drift."""
    frames = _all_symbols_frames(200)
    full = build_global_market_context(frames, timeframe="H1")
    relevant = relevant_correlation_pairs("XAU_USD")
    filtered = build_global_market_context(
        frames, timeframe="H1", relevant_pairs=relevant
    )

    for left, right in relevant:
        for window in (24, 72, 168):
            col = f"corr_{left}__{right}__w{window}"
            assert col in filtered.columns
            assert col in full.columns
            pd.testing.assert_series_equal(
                full[col].reset_index(drop=True),
                filtered[col].reset_index(drop=True),
                check_names=False,
            )


def test_filtered_context_returns_columns_unchanged():
    """Return columns (ret_raw_, ret_vol_) are per-symbol, not per-pair, so
    filtering pairs should not affect them."""
    frames = _all_symbols_frames(200)
    full = build_global_market_context(frames, timeframe="H1")
    relevant = relevant_correlation_pairs("DXY")
    filtered = build_global_market_context(
        frames, timeframe="H1", relevant_pairs=relevant
    )

    for symbol in ("XAU_USD", "USOIL", "EUR_USD", "DXY"):
        for prefix in ("ret_raw_", "ret_vol_"):
            col = f"{prefix}{symbol}"
            assert col in filtered.columns
            pd.testing.assert_series_equal(
                full[col].reset_index(drop=True),
                filtered[col].reset_index(drop=True),
                check_names=False,
            )


def test_lag_scan_columns_unaffected_by_pair_filter():
    """``LAG_SCAN_PAIRS`` is a separate fixed set; it should always run."""
    frames = _all_symbols_frames(200)
    relevant = relevant_correlation_pairs("DXY")
    filtered = build_global_market_context(
        frames, timeframe="H1", relevant_pairs=relevant
    )
    lag_cols = [c for c in filtered.columns if c.startswith("lagcorr_best_")]
    assert any("XAU_USD__USOIL" in c for c in lag_cols)
    assert any("USOIL__DXY" in c for c in lag_cols)


@pytest.mark.parametrize(
    "primary,expected_pair_count",
    [
        ("XAU_USD", 10),
        ("USOIL", 10),
        ("DXY", 2),
        ("EUR_USD", 9),
        ("USD_JPY", 9),
    ],
)
def test_relevance_filter_drops_irrelevant_pairs(primary, expected_pair_count):
    frames = _all_symbols_frames(200)
    relevant = relevant_correlation_pairs(primary)
    assert len(relevant) == expected_pair_count

    filtered = build_global_market_context(
        frames, timeframe="H1", relevant_pairs=relevant
    )
    pair_count = len(
        {
            c.rsplit("__w", 1)[0].replace("corr_", "", 1)
            for c in filtered.columns
            if c.startswith("corr_")
            and "_w" in c
            and not c.startswith("corr_z_")
            and not c.startswith("corr_sig_class_")
            and not c.startswith("corr_stability_class_")
        }
    )
    assert pair_count == expected_pair_count
