from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.pipelines.build_research import run_research_pipeline
from src.indicators.research.cross_asset_research import (
    build_cross_asset_correlation_audit,
    summarize_cross_asset_correlation_audit,
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


def _cross_asset_universe(rows: int = 96) -> dict[str, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=rows, freq="1h", tz="UTC")
    base = np.sin(np.linspace(0.0, 10.0, rows))
    return {
        "XAU_USD": _raw_frame(ts, 2000.0 + np.cumsum(base + 0.45)),
        "DXY": _raw_frame(ts, 100.0 + np.cumsum(-0.35 * base + 0.05)),
        "USD_JPY": _raw_frame(ts, 145.0 + np.cumsum(0.22 * base + 0.03)),
        "USOIL": _raw_frame(ts, 70.0 + np.cumsum(0.50 * base + 0.06)),
        "USD_CAD": _raw_frame(ts, 1.30 + np.cumsum(-0.02 * base + 0.002)),
        "EUR_USD": _raw_frame(ts, 1.10 + np.cumsum(0.01 * base + 0.001)),
        "GBP_USD": _raw_frame(ts, 1.28 + np.cumsum(0.015 * base + 0.001)),
    }


def test_cross_asset_research_emits_conditioned_long_form_tables() -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    runtime = run_research_pipeline(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    enriched = runtime.frame.copy()
    enriched["session_name"] = np.where(
        np.arange(len(enriched)) % 2 == 0, "Asia", "London"
    )
    enriched["regime"] = np.where(np.arange(len(enriched)) % 3 == 0, 0, 2)
    enriched["regime_context_caution"] = np.where(
        np.arange(len(enriched)) % 4 == 0, 1, 0
    )

    audit = build_cross_asset_correlation_audit(
        enriched,
        (
            runtime.market_context
            if runtime.market_context is not None
            else pd.DataFrame()
        ),
        instrument="XAU_USD",
        timeframe="H1",
    )
    matrix_long = audit["matrix_long"]
    assert not matrix_long.empty
    assert {"global", "session", "regime", "regime_stable"}.issubset(
        set(matrix_long["context_type"].astype(str))
    )
    assert {"raw", "vol_norm"}.issubset(set(matrix_long["return_mode"].astype(str)))
    assert {
        "contemporaneous",
        "lag_1",
        "lag_k",
        "best_lag",
        "stability_zscore",
    }.issubset(set(matrix_long["matrix_family"].astype(str)))
    assert {"hac_t_stat", "hac_p_value", "fdr_q_value"}.issubset(
        set(matrix_long["metric"].astype(str))
    )
    dxy_fx_mask = (
        matrix_long["left"].eq("DXY")
        & matrix_long["right"].isin(["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD"])
    ) | (
        matrix_long["right"].eq("DXY")
        & matrix_long["left"].isin(["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD"])
    )
    assert not dxy_fx_mask.any()


def test_cross_asset_research_summary_reports_candidates() -> None:
    universe = _cross_asset_universe()
    primary = universe["XAU_USD"]
    peers = {symbol: frame for symbol, frame in universe.items() if symbol != "XAU_USD"}
    runtime = run_research_pipeline(
        primary,
        instrument="XAU_USD",
        timeframe="H1",
        include_vp=False,
        include_avwap=False,
        include_cross_asset=True,
        peer_raw_frames=peers,
        raw_data_root=None,
    )
    audit = build_cross_asset_correlation_audit(
        runtime.frame,
        (
            runtime.market_context
            if runtime.market_context is not None
            else pd.DataFrame()
        ),
        instrument="XAU_USD",
        timeframe="H1",
    )
    summary = summarize_cross_asset_correlation_audit(audit)

    assert summary["matrix_rows"] > 0
    assert "vol_norm" in summary["return_modes"]
    assert "best_lag" in summary["matrix_families"]
