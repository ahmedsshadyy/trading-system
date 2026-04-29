from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.features.cross_asset import (
    attach_cross_asset_context,
    build_global_market_context,
)


def _raw_frame(
    timestamps: pd.DatetimeIndex,
    close: np.ndarray,
) -> pd.DataFrame:
    close_series = pd.Series(close, dtype=float)
    open_ = close_series.shift(1).fillna(close_series.iloc[0] - 0.1)
    high = pd.concat([open_, close_series], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close_series], axis=1).min(axis=1) - 0.2
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close_series.to_numpy(),
            "volume": np.arange(len(timestamps), dtype=float) + 1000.0,
        }
    )


def test_h1_context_attaches_directly_and_excludes_dxy_fx_pairs() -> None:
    ts = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    eur = _raw_frame(ts, 1.10 + np.linspace(0.0, 0.04, len(ts)))
    gbp = _raw_frame(ts, 1.30 + np.linspace(0.0, 0.06, len(ts)))
    xau = _raw_frame(ts, 2000.0 + np.linspace(0.0, 10.0, len(ts)))
    oil = _raw_frame(ts, 70.0 + np.linspace(0.0, 4.0, len(ts)))
    dxy = _raw_frame(ts, 100.0 + np.linspace(0.0, 1.0, len(ts)))

    context = build_global_market_context(
        {
            "EUR_USD": eur,
            "GBP_USD": gbp,
            "XAU_USD": xau,
            "USOIL": oil,
            "DXY": dxy,
        },
        timeframe="H1",
    )
    attached = attach_cross_asset_context(
        eur,
        instrument="EUR_USD",
        timeframe="H1",
        market_context=context,
    )

    assert "corr_DXY__EUR_USD__w24" not in context.columns
    assert "corr_EUR_USD__DXY__w24" not in context.columns
    assert "xasset_corr_gbp_usd_w24" in attached.columns
    assert "xasset_corr_xau_usd_w24" in attached.columns
    assert "xasset_corr_usoil_w24" in attached.columns
    assert "xasset_corr_dxy_w24" not in attached.columns
    assert "xasset_corr_sig_class_gbp_usd_w24" in attached.columns
    assert "xasset_corr_stability_class_gbp_usd_w24" in attached.columns
    assert attached["xasset_corr_gbp_usd_w24"].notna().sum() > 0


def test_h4_context_aligns_commodities_by_session_end() -> None:
    ts_dxy = pd.date_range("2026-01-01", periods=20, freq="4h", tz="UTC")
    ts_xau = ts_dxy + pd.Timedelta(hours=1)
    dxy = _raw_frame(ts_dxy, 100.0 + np.linspace(0.0, 2.0, len(ts_dxy)))
    xau = _raw_frame(ts_xau, 2000.0 + np.linspace(0.0, 15.0, len(ts_xau)))
    oil = _raw_frame(ts_xau, 70.0 + np.linspace(0.0, 8.0, len(ts_xau)))

    context = build_global_market_context(
        {
            "DXY": dxy,
            "XAU_USD": xau,
            "USOIL": oil,
        },
        timeframe="H4",
    )
    attached = attach_cross_asset_context(
        xau,
        instrument="XAU_USD",
        timeframe="H4",
        market_context=context,
    )

    assert context["timestamp"].iloc[0] == ts_dxy[0]
    assert "xasset_corr_dxy_w6" in attached.columns
    assert attached["xasset_corr_dxy_w6"].notna().sum() > 0


def test_lag_scan_selects_signed_best_lag() -> None:
    # Use enough bars and non-periodic returns so that lag=1 is the unique
    # best peak rather than an alias of a periodic pattern.
    rng = np.random.default_rng(42)
    n = 80
    ts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    usoil_ret = rng.normal(0.0, 0.003, size=n)
    xau_ret = np.roll(usoil_ret, 1) * 0.95 + rng.normal(0.0, 0.0003, size=n)
    xau_ret[0] = 0.0
    usoil_close = 70.0 * np.exp(np.cumsum(usoil_ret))
    xau_close = 2000.0 * np.exp(np.cumsum(xau_ret))
    dxy_close = 100.0 * np.exp(np.cumsum(-0.5 * usoil_ret))

    context = build_global_market_context(
        {
            "XAU_USD": _raw_frame(ts, xau_close),
            "USOIL": _raw_frame(ts, usoil_close),
            "DXY": _raw_frame(ts, dxy_close),
        },
        timeframe="H1",
    )

    lag = context["lagcorr_best_lag_XAU_USD__USOIL__w24"].dropna().iloc[-1]
    score = context["lagcorr_best_XAU_USD__USOIL__w24"].dropna().iloc[-1]
    assert int(lag) == 1
    assert score > 0.8


def test_market_context_output_does_not_leak_raw_price_columns() -> None:
    ts = pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC")
    context = build_global_market_context(
        {
            "EUR_USD": _raw_frame(ts, 1.10 + np.linspace(0.0, 0.03, len(ts))),
            "GBP_USD": _raw_frame(ts, 1.30 + np.linspace(0.0, 0.05, len(ts))),
            "XAU_USD": _raw_frame(ts, 2000.0 + np.linspace(0.0, 12.0, len(ts))),
            "USOIL": _raw_frame(ts, 70.0 + np.linspace(0.0, 5.0, len(ts))),
            "DXY": _raw_frame(ts, 100.0 + np.linspace(0.0, 1.0, len(ts))),
        },
        timeframe="H1",
    )

    assert "EUR_USD" not in context.columns
    assert "GBP_USD" not in context.columns
    assert "ret_raw_EUR_USD" in context.columns
    assert "ret_vol_EUR_USD" in context.columns
    assert not any(
        column in {"open", "high", "low", "close"} for column in context.columns
    )


def test_attachment_rules_limit_columns_by_symbol_type() -> None:
    ts = pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC")
    context = build_global_market_context(
        {
            "EUR_USD": _raw_frame(ts, 1.10 + np.linspace(0.0, 0.03, len(ts))),
            "GBP_USD": _raw_frame(ts, 1.30 + np.linspace(0.0, 0.05, len(ts))),
            "XAU_USD": _raw_frame(ts, 2000.0 + np.linspace(0.0, 12.0, len(ts))),
            "USOIL": _raw_frame(ts, 70.0 + np.linspace(0.0, 5.0, len(ts))),
            "DXY": _raw_frame(ts, 100.0 + np.linspace(0.0, 1.0, len(ts))),
        },
        timeframe="H1",
    )

    fx_attached = attach_cross_asset_context(
        _raw_frame(ts, 1.10 + np.linspace(0.0, 0.03, len(ts))),
        instrument="EUR_USD",
        timeframe="H1",
        market_context=context,
    )
    dxy_attached = attach_cross_asset_context(
        _raw_frame(ts, 100.0 + np.linspace(0.0, 1.0, len(ts))),
        instrument="DXY",
        timeframe="H1",
        market_context=context,
    )

    assert "xasset_corr_dxy_w24" not in fx_attached.columns
    assert "xasset_corr_xau_usd_w24" in fx_attached.columns
    assert "xasset_corr_eur_usd_w24" not in dxy_attached.columns
    assert "xasset_corr_xau_usd_w24" in dxy_attached.columns
    assert "xasset_lagcorr_oil_dxy_best_w24" in dxy_attached.columns
